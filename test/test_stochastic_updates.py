import heavyball
import heavyball.utils
import pytest
import torch
from benchmark.utils import get_optim
from heavyball.utils import clean, set_torch, PSGDBase
from torch import nn


def get_memory():
    clean()
    torch.cuda.synchronize()
    clean()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("opt", heavyball.__all__)
@pytest.mark.parametrize("size,depth", [(128, 2)])
def test_foreach(opt, size, depth: int, iterations: int = 1024, outer_iterations: int = 3):
    set_torch()

    opt = getattr(heavyball, opt)
    if not issubclass(opt, PSGDBase):
        raise pytest.skip('Only PSGD is supported')

    peaks = []
    losses = []

    for stochastic in [False, True]:
        torch.manual_seed(0x2131290)
        peaks.append([])
        losses.append([])

        for i in range(outer_iterations):
            model = nn.Sequential(*[nn.Linear(size, size) for _ in range(depth)]).cuda()
            o = get_optim(opt, model.parameters(), lr=1e-3, stochastic_schedule=stochastic)

            for _ in range(iterations):
                loss = model(torch.randn((128, size)).cuda()).square().mean()
                loss.backward()
                o.step()
                o.zero_grad()
                losses[-1].append(loss.detach())

            del model, o
            clean()

    stochastic = sum([l.item() for l in losses[1]])
    deterministic = sum([l.item() for l in losses[0]])
    print(f"{deterministic=}, {stochastic=}")
    assert deterministic < stochastic