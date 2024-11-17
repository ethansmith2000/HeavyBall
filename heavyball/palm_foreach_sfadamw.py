import torch
import torch.optim

from .utils import schedule_free_, warmup, ScheduleFree, exp_avg_sq_, beta_debias, stoch_state_update, div_stoch_, add_stoch_


class PaLMForeachSFAdamW(ScheduleFree):
    def __init__(self, params, lr=0.0025, beta=0.9, betas=(None, None), eps=1e-8, weight_decay=0, warmup_steps=0, r=0.0,
                 weight_lr_power=2.0, beta2_scale: float = 0.8):
        if betas[0] is not None:
            beta = betas[0]
        defaults = dict(lr=lr, beta=beta, eps=eps, r=r, k=0, warmup_steps=warmup_steps, train_mode=True,
                        weight_sum=0.0, lr_max=-1.0, weight_lr_power=weight_lr_power, weight_decay=weight_decay,
                        beta2_scale=beta2_scale)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            decay = group['weight_decay']
            k = group['k']

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'z' not in self.state_(p):
                    self.state_(p)['z'] = torch.clone(p.data)
                    self.state_(p)['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

            y, grad, exp_avg_sq, z = zip(
                *[(p.data, p.grad.float(), self.state_(p)['exp_avg_sq'], self.state_(p)['z']) for p in active_p])

            # Decay the first moment running average coefficient
            beta2 = 1 - (k + 1) ** -group['beta2_scale']
            old_debiased = beta_debias(beta2, k + 1)

            # Decay the first and second moment running average coefficient
            denom = exp_avg_sq_(exp_avg_sq, grad, old_debiased, eps)

            # Normalize grad in-place for memory efficiency
            if stoch_state_update:
                div_stoch_(grad, denom)
            else:
                torch._foreach_div_(grad, denom)

            # Weight decay calculated at y
            if decay != 0:
                if stoch_state_update:
                    add_stoch_(grad, y, alpha=decay)
                else:
                    torch._foreach_add_(grad, y, alpha=decay)

            lr = warmup(group['lr'], k + 1, group['warmup_steps'])
            group['weight_sum'] = schedule_free_(lr, group['weight_lr_power'], group['weight_sum'], group['beta'],
                                                 y, z, grad, group['r'], k + 1)

            group['k'] = k + 1
        return loss
