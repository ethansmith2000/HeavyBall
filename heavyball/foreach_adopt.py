import torch
import torch.optim

from .utils import warmup, beta_debias, update_param_, StatefulOptimizer, mul_, addcdiv_, addcmul_


class ForeachADOPT(StatefulOptimizer):

    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0):
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

            if k > 1:
                lr = -warmup(group['lr'], k - 1, group['warmup_steps'])

                update_param_(y, exp_avg, lr, decay)
            if k > 0:
                beta1 = beta_debias(group['betas'][0], k)
                denom = torch._foreach_sqrt(exp_avg_sq)
                torch._foreach_maximum_(denom, eps)                
                mul_(exp_avg, beta1)
                addcdiv_(exp_avg, grad, denom, 1 - beta1)

            beta2 = beta_debias(group['betas'][1], k + 1)
            mul_(exp_avg_sq, beta2)
            addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
            del grad

            group['k'] = k + 1
        return loss
