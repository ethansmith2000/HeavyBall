import torch
import torch.optim

from heavyball.utils import copy_stochastic_list_
from .utils import warmup, exp_avg_sq_, beta_debias, update_param_, StatefulOptimizer, promote


@torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True, dynamic=True)
def _compilable_step_(y, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay):
    g32, exp_avg32, exp_avg_sq32 = [list(map(promote, x)) for x in [grad, exp_avg, exp_avg_sq]]

    torch._foreach_lerp_(exp_avg32, g32, 1 - beta_debias(beta1, step + 1))
    denom = list(exp_avg_sq_(exp_avg_sq32, g32, beta_debias(beta2, step + 1), eps))

    update_param_(y, exp_avg32, lr, decay, lambda p, e, l: p.addcdiv_(e, denom.pop(0), value=l))

    copy_stochastic_list_(exp_avg, exp_avg32)
    copy_stochastic_list_(exp_avg_sq, exp_avg_sq32)


class ForeachAdamW(StatefulOptimizer):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0,
                 foreach: bool = True, storage_dtype: str = 'float32'):
        defaults = dict(lr=lr, betas=betas, eps=eps, k=0, warmup_steps=warmup_steps, train_mode=True, weight_sum=0.0,
                        lr_max=-1.0, weight_decay=weight_decay, storage_dtype=storage_dtype)
        super().__init__(params, defaults, foreach)

    def _step(self, group):
        eps = group['eps']
        decay = group['weight_decay']
        k = group['k']

        if not group['train_mode']:
            raise Exception("Not in train mode!")

        active_p = [p for p in group['params'] if p.grad is not None]

        if not active_p:
            return

        storage_dtype = getattr(torch, group['storage_dtype'])

        for p in active_p:
            if 'exp_avg' not in self.state_(p):
                self.state_(p)['exp_avg'] = torch.zeros_like(p.data, dtype=storage_dtype)
                self.state_(p)['exp_avg_sq'] = torch.zeros_like(p.data, dtype=storage_dtype)

        y, grad, exp_avg_sq, exp_avg = zip(
            *[(p.data, p.grad, self.state_(p)['exp_avg_sq'], self.state_(p)['exp_avg']) for p in active_p])

        lr = -warmup(group['lr'], k + 1, group['warmup_steps'])
        lr = torch.empty((), dtype=torch.float32, device=y[0].device).fill_(lr)
        step = torch.empty((), dtype=torch.int32, device=y[0].device).fill_(k)
        _compilable_step_(y, grad, exp_avg_sq, exp_avg, group['betas'][0], group['betas'][1], step, lr, eps, decay)

        group['k'] = k + 1
