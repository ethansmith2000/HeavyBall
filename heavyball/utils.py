import functools
import gc
import math
import random
import string
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.backends import cudnn, opt_einsum
from torch.utils._pytree import tree_map

compile_mode = None
zeroth_power_mode = 'qr'  # 'qr' is baseline, 'newtonschulz' converges better and faster, 'eigh' is perfect but slow
stoch_state_update = False  # stochastic state update


def decorator(func):
    compiled = None

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if compile_mode is None:
            return func(*args, **kwargs)
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(func, fullgraph=True, dynamic=False, mode=compile_mode)
        return compiled(*args, **kwargs)

    return _fn


_einsum_base = string.ascii_lowercase + string.ascii_uppercase


def warmup(lr: float, step: int, warmup_steps: int):
    if step >= warmup_steps:  # if instead of min to guard against 0 div
        return lr
    return lr * step / warmup_steps


def schedule_free_(lr: float, weight_lr_power: float, weight_sum: float, beta1: float, parameters: List[torch.Tensor],
                   z: List[torch.Tensor], grad: list[torch.Tensor], r: float = 0.0, step: int = 0):
    weight = lr ** weight_lr_power * max(step, 1) ** r
    weight_sum = weight_sum + weight

    try:
        ckp1 = weight / weight_sum
    except ZeroDivisionError:
        ckp1 = 0

    # These operations update y in-place,
    # without computing x explicitly.
    p32 = [promote(p) for p in parameters]
    z32 = [promote(z_) for z_ in z]
    torch._foreach_lerp_(p32, z32, weight=ckp1)
    torch._foreach_add_(p32, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))
    copy_stochastic_list_(parameters, p32)

    # z step
    torch._foreach_sub_(z, grad, alpha=lr)
    copy_stochastic_list_(z, z32)
    return weight_sum


def promote_or_promote_list(x):
    if isinstance(x, list):
        return [promote(a) for a in x]
    return promote(x)


def two_arg_stoch_(fn, for_each_fn, x, y, **kwargs):
    if isinstance(x, torch.Tensor):
        x32 = promote(x)
        getattr(x32, fn)(promote(y), **kwargs)
        copy_stochastic_(x, x32)
    else:
        x32 = [promote(a) for a in x]
        y32 = [promote(a) for a in y]
        for_each_fn(x32, y32, **kwargs)
        copy_stochastic_list_(x, x32)

def three_arg_stoch_(fn, for_each_fn, x, y, z, **kwargs):
    if isinstance(x, torch.Tensor):
        x32 = promote(x)
        getattr(x32, fn)(promote(y), promote(z), **kwargs)
        copy_stochastic_(x, x32)
    else:
        x32 = [promote(a) for a in x]
        y32 = [promote(a) for a in y]
        # check if z[0] and y[0] are the same
        if z[0].data_ptr() == y[0].data_ptr():
            for_each_fn(x32, y32, y32, **kwargs)
        else:
            z32 = [promote(a) for a in z]
            for_each_fn(x32, y32, z32, **kwargs)
        copy_stochastic_list_(x, x32)


# stochastic ops
def mul_stoch_(x, y):
    two_arg_stoch_('mul_', torch._foreach_mul_, x, y)

def div_stoch_(x, y):
    two_arg_stoch_('div_', torch._foreach_div_, x, y)

def lerp_stoch_(x, y, weight=1):
    two_arg_stoch_('lerp_', torch._foreach_lerp_, x, y, weight=weight)

def addcmul_stoch_(x, y, z, value=1):
    three_arg_stoch_('addcmul_', torch._foreach_addcmul_, x, y, z, value=value)

def addcdiv_stoch_(x, y, z, value=1):
    three_arg_stoch_('addcdiv_', torch._foreach_addcdiv_, x, y, z, value=value)

def add_stoch_(x, y, alpha=1):
    two_arg_stoch_('add_', torch._foreach_add_, x, y, alpha=alpha)


def append_or_extend(base, new):
    if isinstance(new, list):
        base.extend(new)
    else:
        base.append(new)


def dim_merger(grad, max_precond_dim, split: bool = False):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done
    """
    shape = grad.shape
    new_shape = []

    curr_shape = 1

    for sh in shape[1:][::-1]:
        temp_shape = curr_shape * sh
        if temp_shape > max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape
    new_shape = [*shape[:1], *new_shape[::-1]]

    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)

    new_grad = grad.view(new_shape)
    if not split:
        return new_grad

    grads = [new_grad]
    for i, sh in reversed(list(enumerate(new_shape[:]))):
        if sh == 1:
            grads = [g.squeeze(dim=i) for g in grads]
            continue
        if sh <= max_precond_dim:
            continue
        grads = [a for g in grads for a in g.split(max_precond_dim, dim=i)]
    if len(grads) == 1:
        return new_grad
    new_grads = []
    for g in grads:
        append_or_extend(new_grads, dim_merger(g, max_precond_dim, split))
    return new_grads


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta ** step)


def exp_avg_sq_(state, grad, beta2, eps, out=None):
    if isinstance(state, torch.Tensor):
        if stoch_state_update:
            mul_stoch_(state, beta2)
            addcmul_stoch_(state, grad, grad, value=1 - beta2)
        else:
            state.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        return torch.sqrt(state, out=out).clamp_(min=eps)

    if stoch_state_update:
        mul_stoch_(state, beta2)
        addcmul_stoch_(state, grad, grad, value=1 - beta2)
    else:
        torch._foreach_mul_(state, beta2)
        torch._foreach_addcmul_(state, grad, grad, value=1 - beta2)

    denom = torch._foreach_sqrt(state)
    torch._foreach_maximum_(denom, eps)
    return denom



def adaptive_gradient_clipping_(parameters: List[torch.Tensor], gradients: List[torch.Tensor], clip_val: float,
                                minimum: float = 1e-3, eps: float = 1e-8):
    if clip_val <= 0:
        return
    p_norm = torch._foreach_norm(parameters)
    g_norm = torch._foreach_norm(gradients)
    torch._foreach_maximum_(p_norm, minimum)
    torch._foreach_maximum_(g_norm, eps)
    torch._foreach_div_(p_norm, g_norm)
    torch._foreach_mul_(p_norm, clip_val)
    torch._foreach_minimum_(p_norm, 1)
    torch._foreach_mul_(gradients, p_norm)


def set_(dst: torch.Tensor, src: torch.Tensor):
    if src.data_ptr() == dst.data_ptr():
        return
    if src.shape != dst.shape:
        src = src.reshape_as(dst)
    if src.is_contiguous() and dst.is_contiguous() and src.dtype == dst.dtype:
        dst.set_(src)
    else:
        dst.copy_(src)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


def set_torch():
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16
    opt_einsum.enabled = True
    opt_einsum.strategy = "auto-hq"


@decorator
def zeropower_via_newtonschulz5(G, init, steps=2, eps=1e-7):
    """
    Modified from "modded-nanogpt" under the MIT license:
    Original: https://github.com/KellerJordan/modded-nanogpt/blob/a0dcbfdd9a0617d091d5123cfc354745428e40d3/train_gpt2.py

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    init = init / (init.norm() + eps)  # ensure top singular value <= 1
    X = X / (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T  # preconditioner
        B = A @ init
        init = X = a * init + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


def ortho(x):
    if zeroth_power_mode == 'qr':
        return torch.linalg.qr(x).Q
    if zeroth_power_mode == 'svd':
        u, s, v = torch.linalg.svd(x)
        return u @ v.T
    raise NotImplementedError(f"Unknown zeroth_power_mode: {zeroth_power_mode}")


def get_orthogonal_matrix_QR(GG, Q, exp_avg_sq):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    for m, o in zip(GG, Q):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))
        else:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))

    indices = []

    for ind, (m, o, q) in enumerate(zip(matrix, orth_matrix, Q)):
        if len(m) == 0:
            indices.append(None)
            continue

        tmp = m @ o
        est_eig = torch.einsum('ij,ij->j', o, tmp)
        sort_idx = torch.argsort(est_eig, descending=True)
        indices.append(sort_idx)
        if zeroth_power_mode == 'eigh':
            set_(q, torch.linalg.eigh(m)[1])
        elif zeroth_power_mode.startswith('newtonschulz'):
            iterations = zeroth_power_mode[len('newtonschulz'):]
            if iterations == '':
                iterations = 10
            else:
                iterations = int(iterations)
            set_(q, zeropower_via_newtonschulz5(m, o[:, sort_idx], iterations))
        else:
            set_(q, ortho(tmp[:, sort_idx]))

    indices = tuple(slice(None) if ind is None else ind.view(*(1,) * i, -1, *(1,) * (exp_avg_sq.dim() - i - 1))  #
                    for i, ind in enumerate(indices))
    set_(exp_avg_sq, exp_avg_sq[indices])


def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(promote(m.data))
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue

        device, dtype = m.device, m.dtype
        for modifier in (None, torch.double, 'cpu'):
            if modifier is not None:
                m = m.to(modifier)
            try:
                Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))[1].to(device=device,
                                                                                                dtype=dtype)
                break
            except torch.OutOfMemoryError:
                pass
            except RuntimeError:  # failed to compute eigenvalues
                continue
            clean()
        else:
            raise RuntimeError("Failed to compute eigenvalues.")

        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)

    return final


@decorator
def compute_ggt(grad, GG, max_precond_dim, precondition_1d, beta):
    if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return



    for idx, sh in enumerate(grad.shape):
        if sh > max_precond_dim:
            continue
        b = _einsum_base[idx]
        g0 = _einsum_base[:grad.dim()]
        g1 = g0.replace(b, b.upper())
        if stoch_state_update:
            outer_product = torch.einsum(f'{g0},{g1}->{b + b.upper()}', promote(grad), promote(grad))
            GG32_idx = promote(GG[idx])
            GG32_idx.lerp_(outer_product, 1 - beta)
            copy_stochastic_(GG[idx], GG32_idx)
        else:
            outer_product = torch.einsum(f'{g0},{g1}->{b + b.upper()}', grad, grad)
            GG[idx].lerp_(promote(outer_product), 1 - beta)


def promote(x):
    if x is (torch.bfloat16, torch.float16):
        return torch.float32
    if hasattr(x, 'dtype'):
        if x.dtype in (torch.bfloat16, torch.float16):
            return x.float()
    return x


def update_preconditioner(grad, state, max_precond_dim, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    compute_ggt(grad, state['GG'], max_precond_dim, precondition_1d, beta)
    if state['Q'] is None:
        state['Q'] = get_orthogonal_matrix(state['GG'])
    if update_precond:
        get_orthogonal_matrix_QR(state['GG'], state['Q'], state['exp_avg_sq'])


def init_preconditioner(grad, state, max_precond_dim=10000, precondition_1d=False):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state['Q'] = None  # Will hold all the eigenbases of the preconditioner.
    state['GG'] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if not precondition_1d or grad.shape[0] > max_precond_dim:
            state['GG'].append([])
            return
        state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype))
        return

    for sh in grad.shape:
        if sh > max_precond_dim:
            state['GG'].append([])
        else:
            state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))


@decorator
def project(grad, Q, back: bool):
    """

    :param grad:
    :param Q:
    :param merge_dims:
    :param max_precond_dim:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    param = _einsum_base[:grad.dim()]
    preconditioners = ",".join([(g + g.upper())[::-1 if back else 1] for m, g in zip(Q, param) if len(m) > 0])
    if preconditioners:
        out = ''.join([c.upper() if c.upper() in preconditioners else c for c in param])
        if stoch_state_update:
            grad = torch.einsum(f'{param},{preconditioners}->{out}', promote(grad), *[promote(q) for q in Q if len(q) > 0])
        else:
            grad = torch.einsum(f'{param},{preconditioners}->{out}', grad, *[q for q in Q if len(q) > 0])
    return grad


class StatefulOptimizer(torch.optim.Optimizer):
    def state_(self, arg: torch.Tensor):
        return self.state[(arg.data_ptr(), tuple(arg.shape))]

    def state_size(self) -> int:
        total_bytes = 0

        def _add(x):
            nonlocal total_bytes
            if isinstance(x, torch.Tensor):
                total_bytes += x.numel() * x.element_size()

        for group in self.param_groups:
            for p, _ in split_p_and_g_in_group(group, skip_none=False):
                tree_map(_add, self.state_(p))
        return total_bytes


class ScheduleFree(StatefulOptimizer):
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        # Set p.data to x
                        z = promote(state['z'])
                        p32 = promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        copy_stochastic_(p.data, p32)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state_(p)
                    if 'z' in state:
                        z = promote(state['z'])
                        p32 = promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        copy_stochastic_(p.data, p32)
                group['train_mode'] = True

    def _step(self):
        raise NotImplementedError


def copy_stochastic_list_(target: List[torch.Tensor], source: List[torch.Tensor]):
    for t, s in zip(target, source):
        if t.dtype == torch.bfloat16:
            copy_stochastic_(t, s)
        else:
            set_(t, s)


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    if target.data_ptr() == source.data_ptr():
        return

    """Taken as-is from https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905"""
    # create a random 16 bit integer
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def update_param_(param: List[torch.Tensor], update: List[torch.Tensor], lr: float, decay: float,
                  add_fn: callable = None):
    param32 = [promote(p) for p in param]
    update32 = [promote(u.view(p.shape)) for u, p in zip(update, param)]
    if decay > 0:
        torch._foreach_mul_(param32, 1 - decay * lr)
    if add_fn is None:
        torch._foreach_add_(param32, update32, alpha=lr)
    else:
        add_fn(param32, update32, lr)
    copy_stochastic_list_(param, param32)


def precond_schedule(step, precond_scheduler, rng):
    precond_prob = max(step, 1) ** precond_scheduler[0]
    precond_prob = math.log10(precond_prob)
    precond_prob = precond_prob ** precond_scheduler[1] + 1
    precond_prob = 1 / precond_prob
    update_precond = rng.random() < precond_prob
    return update_precond


def init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape

    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
        return [Q, (exprA, tuple(exprGs), exprP)]

    # Tensor
    if len(shape) > 13:
        raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

    scale = scale ** (1 / len(shape))

    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == "one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [False for _ in shape]
        dim_diag[rev_sorted_dims[0]] = True
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                         "[None, 'one_diag', 'all_diag']")

    Q = []
    piece1A, piece2A, piece3A = ([], "", "")
    exprGs = []
    piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
    for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
            # use diagonal matrix as preconditioner for this dim
            Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

            piece1A.append(letters[i])
            piece2A = piece2A + letters[i]
            piece3A = piece3A + letters[i]

            piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
            exprGs.append(subscripts)

            piece1P.append(letters[i + 13])
            piece2P.append(letters[i + 13])
            piece3P = piece3P + letters[i + 13]
            piece4P = piece4P + letters[i + 13]
        else:
            # use triangular matrix as preconditioner for this dim
            Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

            piece1A.append(letters[i] + letters[i + 13])
            piece2A = piece2A + letters[i + 13]
            piece3A = piece3A + letters[i]

            piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            piece2 = "".join([(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))])
            subscripts = (piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26])
            exprGs.append(subscripts)

            a, b, c = (letters[i], letters[i + 13], letters[i + 26])
            piece1P.append(a + b)
            piece2P.append(a + c)
            piece3P = piece3P + c
            piece4P = piece4P + b

    exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
    exprP = (",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P)
    return [Q, (exprA, tuple(exprGs), exprP)]


@decorator
def psgd_balance_Q(Q_in):
    if stoch_state_update:
        Q_in_32 = [promote(q) for q in Q_in]
        # norms = torch._foreach_norm(Q_in_32, float("inf"))
        norms = torch.stack([q.norm(float("inf")) for q in Q_in_32])
        geometric_mean = norms.log().mean().exp()
        norms = geometric_mean / norms
        torch._foreach_mul_(Q_in_32, list(norms))
        copy_stochastic_list_(Q_in, Q_in_32)
    else:
        norms = torch.stack([q.norm(float("inf")) for q in Q_in])
        geometric_mean = norms.log().mean().exp()
        norms = geometric_mean / norms
        torch._foreach_mul_(Q_in, list(norms))


def psgd_calc_A_and_conjB(exprA, G, Q, V):
    A = torch.einsum(exprA, *Q, G)
    order = G.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        if q.dim() <= 1:
            conjB /= q
        else:
            unsqueeze = conjB.dim() <= 1
            if unsqueeze:
                conjB = conjB.unsqueeze(0)
            conjB = torch.linalg.solve_triangular(q, conjB, upper=True, left=False, out=conjB)
            if unsqueeze:
                conjB = conjB.squeeze(0)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    return A, conjB


def psgd_lb(A, max_abs):
    A /= max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)

    ah = A.H
    comp = value0 > value1
    x = torch.where(comp, A[:, i], A[j])
    x = x.conj()
    if x.dim() > 1:
        x = torch.where(comp, x, x.T)
    torch.matmul(x, torch.where(comp, A, A.T), out=x.view(1, -1))
    x /= torch.linalg.vector_norm(x)
    torch.matmul(x, torch.where(comp, ah, ah.T), out=x.view(1, -1))
    x = torch.linalg.vector_norm(x)
    x *= max_abs
    return x


def psgd_update_precond(Q, exprs, V, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs

    A, conjB = psgd_calc_A_and_conjB(exprA, G, Q, V)

    if stoch_state_update:
        A = promote(A)
        conjB = promote(conjB)

    for q, exprG in zip(Q, exprGs):
        if stoch_state_update:
            q_ = promote(q)
        else:
            q_ = q

        term1 = torch.einsum(exprG, A, A.conj())
        term2 = torch.einsum(exprG, conjB.conj(), conjB)

        term2 += term1  # a + b
        term1 *= 2  # 2a
        term1 -= term2  # 2a - (a + b) == a - b

        term1 *= step
        norm = term2.norm(float('inf'))
        if q_.dim() < 2:
            term1 *= q_
            q_.addcdiv_(term1, norm.clamp_(min=tiny), value=-1)
        else:
            torch.triu(term1, out=term1)
            term1 /= torch.where(norm > 0, psgd_lb(term2, norm), norm).clamp_(tiny)
            q_.addmm_(term1, q, alpha=-1)

        if stoch_state_update:
            copy_stochastic_(q, q_)


@decorator
def psgd_precond_grad(Q, exprs, G, inplace: bool = False):
    """Precondition gradient G with preconditioner Q."""
    if stoch_state_update:
        out32 = torch.einsum(exprs[-1], *[promote(q).conj() for q in Q], *[promote(q) for q in Q], promote(G))
        if inplace:
            copy_stochastic_(G, out32)
            return G
        return out32

    else:
        out = torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)
        if inplace:
            set_(G, out)
            return G
        return out


def norm_clip_(x, scale=None):
    norm = torch._foreach_norm(x)
    if scale is not None:
        torch._foreach_div_(norm, scale)
    torch._foreach_div_(x, norm)
    return x


def mu_law_compress(x, mu=127.0):
    """
    Foreach version of https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py

    μ-law compression
    Args:
        x: Input tensor
        mu: Compression parameter (default 127.0 for behavior similar to trust_region=1.5)
    """
    xa = torch._foreach_abs_(x)
    torch._foreach_mul_(xa, mu)
    torch._foreach_log1p_(xa)
    torch._foreach_div_(xa, math.log1p(mu))
    return [xa_.copysign_(x_) for x_, xa_ in zip(x, xa)]


def a_law_compress(x, A=87.6):
    """
    Foreach version of https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py

    A-law compression
    Args:
        x: Input tensor
        A: Compression parameter (default 87.6 - European PCM standard)
    """
    xa = torch._foreach_abs(x)
    torch._foreach_mul_(xa, A)
    [torch.where(x_ < 1, x_, 1 + torch.log_(x_), out=x_) for x_ in xa]
    [xa_.copysign(x_) for x_, xa_ in zip(x, xa)]
    torch._foreach_mul_(xa, 1 / (1 + math.log(A)))
    return xa

def identity(x):
    return x

def trust_region_clip_(grad, lerp: float = 0.9, scale: float = 1.5):
    torch._foreach_mul_(grad, 1 / scale)
    tanh = torch._foreach_tanh(grad)
    torch._foreach_abs_(grad)
    torch._foreach_log1p_(grad)
    grad = [p.copysign_(t) for t, p in zip(tanh, grad)]  # torch doesn't have a foreach copysign
    torch._foreach_lerp_(grad, tanh, lerp)  # sgn(x) * log(1 + |x|) * 0.1 + tanh(x) * 0.9
    torch._foreach_mul_(grad, scale)

    torch._foreach_maximum_(grad, -2)
    torch._foreach_minimum_(grad, 2)
    return grad


@decorator
def triu_to_line(Q_list: List[torch.Tensor]):
    out = []
    for q in Q_list:
        if q.dim() < 2:
            out.append((None, q))
        else:
            out.append((q.shape, q[*torch.triu_indices(*q.shape)]))
    return out


def _triu_shape(numel):
    n = int((2 * numel) ** 0.5)
    assert n * (n + 1) == 2 * numel
    return n, n


@decorator
def line_to_triu(Q_list: List[Tuple[Optional[List[int]], torch.Tensor]]):
    new = []
    for shape, q in Q_list:
        if shape is not None:
            shape = _triu_shape(q.numel())
            x = torch.zeros(shape, device=q.device, dtype=q.dtype)
            x[*torch.triu_indices(*shape, device=q.device)] = q
            q = x
        new.append(q)
    return new


def update_triu_(q_state, materialised):
    for (shape0, q), (shape1, m) in zip(q_state, triu_to_line(materialised)):
        assert shape0 == shape1
        set_(q, m)


class PSGDBase(StatefulOptimizer):
    def __init__(self, parameters, groups):
        super().__init__(parameters, groups)
        self.rng = random.Random(0x1923213)
        self._tiny = torch.finfo(torch.bfloat16).tiny

    def balance(self, do_update, grad_list, Q_list):
        if not do_update or self.rng.random() > 0.01:
            return

        for g, q in zip(grad_list, Q_list):
            if g.dim() > 1:
                psgd_balance_Q(q)

    def do_update(self, p_list, grad_list, q_list, precond_lr, original_q: Optional[List] = None):
        for i, (p, grad, Q) in enumerate(zip(p_list, grad_list, q_list)):
            psgd_update_precond(Q, self.state_(p)["exprs"], torch.randn_like(grad), grad, precond_lr, self._tiny)
            if original_q:
                update_triu_(original_q[i], Q)


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=250):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    def _schedule(n):
        if n < flat_start:  # higher numerical stability
            return max_prob

        n -= flat_start
        prob = max_prob * math.exp(-decay * (n - flat_start))
        return max(min_prob, min(max_prob, prob))

    return _schedule


def merge_group(group, *tensors):
    if not group.get('merge_dims', False):
        return tensors
    if isinstance(tensors[0], list):
        return [merge_group(group, *t) for t in tensors]

    out = []
    for t in tensors:
        append_or_extend(out, dim_merger(t, group['max_size_triangular'] if 'max_size_triangular' in group else group[
            'max_precond_dim'], group.get('split', False)))
    return out


def split_p_and_g_in_group(group: dict, skip_none: bool = True):
    for p in group["params"]:
        if skip_none and p.grad is None:
            continue

        grad = None if p.grad is None else promote(p.grad)
        p.grad = None

        p_views = merge_group(group, p)
        if grad is not None:
            grad = merge_group(group, grad)
        if isinstance(p_views, torch.Tensor):
            yield p_views, grad
            continue
        if grad is None:
            yield from zip(p_views, [None] * len(p_views))
            continue
        yield from zip(p_views, grad)
