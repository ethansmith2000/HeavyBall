import random

import torch

from .utils import init_preconditioner, update_preconditioner, project, beta_debias, exp_avg_sq_, update_param_, \
    precond_schedule, set_, split_p_and_g_in_group, StatefulOptimizer, mul_, add_


class PrecondSchedulePaLMForeachSOAP(StatefulOptimizer):
    """
    Sources:
        Preconditioner Schedules:
            Preconditioned Stochastic Gradient Descent
            Xi-Lin Li, Omead Pooladzandi, Evan Walters
            https://arxiv.org/abs/1512.04202
            https://github.com/evanatyourservice/kron_torch
            https://github.com/lixilinx/psgd_torch

        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP

        Beta2 Schedule:
            PaLM: Scaling Language Modeling with Pathways
            Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel
            https://arxiv.org/abs/2204.02311
    """

    def __init__(self, params, lr: float = 3e-3, beta=0.9, shampoo_beta: float = 0.95, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 data_format: str = "channels_first", correct_bias: bool = True, warmup_steps: int = 1,
                 precond_scheduler=(1 / 3, 9), betas=(None, None), beta2_scale: float = 0.8, split: bool = False):
        if betas[0] is not None:
            beta = betas[0]
        defaults = {"lr": lr, "beta": beta, "shampoo_beta": shampoo_beta, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias, 'warmup_steps': warmup_steps, 'precond_scheduler': precond_scheduler,
                    'beta2_scale': beta2_scale, 'split': split}
        super().__init__(params, defaults)
        self._data_format = data_format
        self.rng = random.Random(0x120983109)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            vals = []
            step = 0

            max_precond_dim = group['max_precond_dim']
            precondition_1d = group['precondition_1d']

            for p, g in split_p_and_g_in_group(group):
                state = self.state_(p)
                step = state['step'] = state.get("step", -1) + 1

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(g, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(g, dtype=torch.float32)
                    init_preconditioner(g, state, max_precond_dim, precondition_1d)
                    update_preconditioner(g, state, max_precond_dim, precondition_1d, 0, True)
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = project(g, state['Q'], False)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                vals.append((p, g, grad_projected, exp_avg, exp_avg_sq))

            if not vals:
                continue

            p_list, grad, grad_projected, exp_avg, exp_avg_sq = zip(*vals)
            beta1 = group["beta"]

            beta2 = 1 - max(step, 1) ** -group['beta2_scale']
            old_debiased1 = beta_debias(beta1, step)
            old_debiased2 = beta_debias(beta2, step)

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            mul_(exp_avg, old_debiased1)
            add_(exp_avg, grad, alpha=1 - old_debiased1)
            denom = exp_avg_sq_(exp_avg_sq, grad_projected, old_debiased2, group['eps'])

            update_precond = precond_schedule(step, group['precond_scheduler'], self.rng)
            for p, g, ea, d in zip(p_list, grad, exp_avg, denom):
                state = self.state_(p)
                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                exp_avg_projected = project(ea, state['Q'], False)

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                # CANT DO /= HERE AS EXP_AVG MAY POINT TO THE BUFFER
                exp_avg_projected = exp_avg_projected / d
                set_(d, project(exp_avg_projected, state['Q'], True))

                update_preconditioner(g, state, max_precond_dim, precondition_1d, old_debiased2,
                                      update_precond)

            # Why does this have to be rebiased here?
            step_size = -group["lr"] * min(step / group['warmup_steps'], 1)
            update_param_(p_list, denom, step_size, group["weight_decay"])
        return loss
