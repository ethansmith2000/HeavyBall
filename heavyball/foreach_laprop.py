import torch
import torch.optim

from .utils import warmup, exp_avg_sq_, beta_debias, update_param_, StatefulOptimizer, stoch_state_update, mul_stoch_, addcdiv_stoch_


class ForeachLaProp(StatefulOptimizer):

    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, k=0, warmup_steps=warmup_steps, train_mode=True, weight_sum=0.0,
                        lr_max=-1.0, weight_decay=weight_decay)
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
                if 'exp_avg' not in self.state_(p):
                    self.state_(p)['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    self.state_(p)['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

            y, grad, exp_avg_sq, exp_avg = zip(
                *[(p.data, p.grad.float(), self.state_(p)['exp_avg_sq'], self.state_(p)['exp_avg']) for p in active_p])

            # Decay the first and second moment running average coefficient
            denom = exp_avg_sq_(exp_avg_sq, grad, beta_debias(group['betas'][1], k + 1), eps)
            beta1 = beta_debias(group['betas'][0], k + 1)
            if stoch_state_update:
                mul_stoch_(exp_avg, beta1)
                addcdiv_stoch_(exp_avg, grad, denom, 1 - beta1)
            else:
                torch._foreach_mul_(exp_avg, beta1)
                torch._foreach_addcdiv_(exp_avg, grad, denom, 1 - beta1)
            del grad

            # Normalize grad in-place for memory efficiency
            lr = -warmup(group['lr'], k + 1, group['warmup_steps'])
            update_param_(y, exp_avg, lr, decay)

            group['k'] = k + 1
        return loss
