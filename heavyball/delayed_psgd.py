"""
Originally from Evan Walters and Omead Pooladzandi, 2024
Modified under Creative Commons Attribution 4.0 International
Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
"""

import torch

from heavyball.utils import copy_stochastic_list_
from .utils import update_param_, warmup, psgd_precond_grad, init_Q_exprs, trust_region_clip_, PSGDBase, \
    precond_update_prob_schedule, split_p_and_g_in_group, triu_to_line, line_to_triu, set_, lerp_


class ForeachDelayedPSGD(PSGDBase):
    """
    Implements PSGD with off-by-one preconditioning (akin to ADOPT and SOAP)

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        b1 (float): Momentum parameter.
        weight_decay (float): Weight decay (L2 penalty).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs
            to have triangular preconditioners.
        memory_save_mode: (string, optional), None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        momentum_into_precond_update: (bool), whether to send momentum into preconditioner
            update instead of raw gradients.
    """

    def __init__(self, params, lr=0.001, beta=0.9, weight_decay=0.0, preconditioner_update_probability=None,
                 max_size_triangular=2048, min_ndim_triangular=2, memory_save_mode=None,
                 momentum_into_precond_update=True, warmup_steps: int = 1, merge_dims: bool = False,
                 split: bool = False, clip_fn: callable = None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()
        if clip_fn is None:
            clip_fn = lambda x: trust_region_clip_(x, 0.9, 1.5)
        self.preconditioner_update_probability = preconditioner_update_probability
        self.clip_fn = clip_fn

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, max_size_triangular=max_size_triangular,
                        min_ndim_triangular=min_ndim_triangular, memory_save_mode=memory_save_mode,
                        momentum_into_precond_update=momentum_into_precond_update, precond_lr=0.1,
                        # precond lr hardcoded to 0.1
                        precond_init_scale=1.0,  # precond init scale hardcoded to 1.0
                        step=0, warmup_steps=warmup_steps, merge_dims=merge_dims, split=split)
        super().__init__(params, defaults)

        self._prob_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # update preconditioners all together
        update_prob = self.preconditioner_update_probability
        if callable(update_prob):
            update_prob = update_prob(self._prob_step)
        do_update = self.rng.random() < update_prob
        self._prob_step += 1

        for group in self.param_groups:
            momentum_into_precond_update = group.get("momentum_into_precond_update", True)
            precond_init_scale = group['precond_init_scale']
            max_size_triangular = group['max_size_triangular']
            min_ndim_triangular = group['min_ndim_triangular']
            memory_save_mode = group['memory_save_mode']
            precond_lr = group['precond_lr']
            weight_decay = group['weight_decay']
            lr = group['lr']
            beta = group['beta']

            vals = []

            for p, g in split_p_and_g_in_group(group):
                state = self.state_(p)

                if 'Q' not in state:
                    state["exp_avg"] = torch.zeros_like(g)
                    Q, state["exprs"] = init_Q_exprs(p, precond_init_scale, max_size_triangular, min_ndim_triangular,
                                                     memory_save_mode, dtype=g.dtype)
                    state["Q"] = triu_to_line(Q)

                vals.append((p, g, state["exp_avg"], state["Q"]))

            if not vals:
                continue

            p_list, grad_list, exp_avg_list, Q_list = zip(*vals)
            del vals

            group["step"] += 1

            lerp_(exp_avg_list, grad_list, (1 - beta) / (1 - beta ** group["step"]))

            Q_list, exp_avg_list = list(Q_list), list(exp_avg_list)
            for i, (p, g) in enumerate(zip(p_list, grad_list)):
                q_orig = Q_list.pop(0)
                ea = exp_avg_list.pop(0)
                q = line_to_triu(q_orig)
                self.balance(do_update, [g], [q])
                new = psgd_precond_grad(q, self.state_(p)["exprs"], ea)

                if do_update:
                    self.do_update([p], [ea if momentum_into_precond_update else g], [q], precond_lr, [q_orig])
                set_(g, new)

            grad_list = self.clip_fn(grad_list)

            lr = -warmup(lr, group['step'], group['warmup_steps'])
            update_param_(p_list, grad_list, lr, weight_decay)

        return loss
