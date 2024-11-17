import random

import torch

from .utils import init_preconditioner, update_preconditioner, project, set_, adaptive_gradient_clipping_, \
    exp_avg_sq_, beta_debias, schedule_free_, warmup, ScheduleFree, precond_schedule, split_p_and_g_in_group, div_, add_


class PrecondScheduleSFPaLMSOAP(ScheduleFree):
    """
    SFPaLMForeachSOAP with preconditioner schedules

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

        ScheduleFree:
            The Road Less Scheduled
            Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky
            https://arxiv.org/abs/2405.15682
            https://github.com/facebookresearch/schedule_free

        Beta2 Schedule:
            PaLM: Scaling Language Modeling with Pathways
            Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel
            https://arxiv.org/abs/2204.02311
    """

    def __init__(self, params, lr: float = 3e-3, beta=0.9, beta2_scale: float = 0.8, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 data_format: str = "channels_first", correct_bias: bool = True, warmup_steps: int = 1, r=0.0,
                 weight_lr_power=2.0, gradient_clip_val: float = 0.1, precond_scheduler=(1 / 3, 9),
                 betas=(None, None), split: bool = False):
        if betas[0] is not None:
            beta = betas[0]
        defaults = {"lr": lr, "beta": beta, "beta2_scale": beta2_scale, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias, 'warmup_steps': warmup_steps, 'r': r,
                    'weight_lr_power': weight_lr_power, 'train_mode': True, 'step': -1, 'weight_sum': 0,
                    'gradient_clip_val': gradient_clip_val, 'precond_scheduler': precond_scheduler, 'split': split}
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
            max_precond_dim = group['max_precond_dim']
            precondition_1d = group['precondition_1d']

            step = group['step'] = group.get("step", -1) + 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                vals.append((p, grad))

            p_list, grad = zip(*vals)
            vals = []

            # adaptive gradient clipping
            adaptive_gradient_clipping_(p_list, grad, group["gradient_clip_val"], eps=group["eps"])

            for p, g in split_p_and_g_in_group(group):
                state = self.state_(p)

                if "z" not in state:
                    state["z"] = torch.clone(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(g, dtype=torch.float32)
                    init_preconditioner(g, state, max_precond_dim, precondition_1d)
                    update_preconditioner(g, state, max_precond_dim, precondition_1d, 0, True)
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = project(g, state['Q'], False)
                z, exp_avg_sq = state["z"], state["exp_avg_sq"]
                vals.append((p, g, grad_projected, z, exp_avg_sq))

            if not vals:
                continue

            p_list, grad, grad_projected, z, exp_avg_sq = zip(*vals)
            del vals

            beta2 = 1 - max(step, 1) ** -group['beta2_scale']
            old_debiased2 = beta_debias(beta2, step)

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            denom = exp_avg_sq_(exp_avg_sq, grad_projected, old_debiased2, group['eps'])
            div_(grad_projected, denom)

            update_precond = precond_schedule(step, group['precond_scheduler'], self.rng)

            for p, g, gp in zip(p_list, grad, grad_projected):
                state = self.state_(p)
                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                set_(gp, project(gp, state['Q'], back=True))

                update_preconditioner(g, state, max_precond_dim, precondition_1d, old_debiased2,
                                      update_precond)

            # Weight decay calculated at y
            if group["weight_decay"] > 0:
                add_(grad, p_list, alpha=group["weight_decay"])

            lr = warmup(group['lr'], step, group['warmup_steps'])
            group['weight_sum'] = schedule_free_(lr, group['weight_lr_power'], group['weight_sum'], group['beta'],
                                                 p_list, z, grad_projected, group['r'], step)

        return loss
