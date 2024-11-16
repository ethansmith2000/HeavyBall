import pytest
import torch
from torch import nn

import heavyball
import heavyball.utils
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


expected_memory = {'adamw': {'after': 4, 'peak': 5.1}, 'soap': {'after': 7, 'peak': 14},
                   'psgd': {'after': 4, 'peak': 11.5}, 'padam': {'after': 5, 'peak': 11.4}}


@pytest.mark.parametrize("opt", ['ForeachPSGDKron', 'ForeachPaLMPAdam', 'ForeachPurePSGD', 'ForeachDelayedPSGD'])
@pytest.mark.parametrize("method",
                         ['norm_clip_', 'mu_law_compress', 'a_law_compress', 'trust_region_clip_', 'identity'])
@pytest.mark.parametrize("size,depth", [(128, 1), (16, 4)])
def test_clip(opt, method, size, depth: int, iterations: int = 100, outer_iterations: int = 3):
    set_torch()

    for k, v in expected_memory.items():
        if k in opt.lower():
            break
    else:
        raise ValueError(f'Unknown optimizer {opt}')

    opt = getattr(heavyball, opt)

    for i in range(outer_iterations):
        model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_accumulated_memory_stats()

        model_allocated = get_memory()
        o = get_optim(opt, model.parameters(), lr=1e-3, clip_fn=getattr(heavyball.utils, method))
        losses = torch.zeros((iterations,), device='cuda')
        for itr in range(iterations):
            src = torch.randn((4, size), device='cuda')
            tgt = src
            loss = (model(src) - tgt).square().mean()
            loss.backward()
            o.step()

            opt_allocated = get_memory()
            o.zero_grad()
            losses[itr] = loss

        del model, o

        arange = torch.arange(iterations, device='cuda', dtype=torch.float32)
        lwma_bwd = (losses @ torch.flip(arange, [0])).item()
        lwma_fwd = (losses @ arange).item()
        assert lwma_bwd > lwma_fwd

        peak = torch.cuda.memory_stats()['allocated_bytes.all.peak']

        print(f'Peak: {peak / model_allocated:.2f}x | Opt: {opt_allocated / model_allocated:.2f}x')
        if i > 0:
            assert peak / model_allocated < v['peak']
            assert opt_allocated / model_allocated < v['after']
