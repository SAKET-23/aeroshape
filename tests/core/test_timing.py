import time

import torch

from aeroshape.core.seeding import rand_on
from aeroshape.core.timing import Timer, peak_memory, reset_peak


def test_timer(device):
    x = torch.ones(256, 256, device=device)
    with Timer(device) as t:
        x @ x
        time.sleep(0.01)
    assert t.elapsed >= 0.01
    reset_peak(device)
    assert peak_memory(device) >= 0


def test_rand_on_identical_across_devices(device):
    a = rand_on(device, 3, 4, seed=1)
    b = rand_on(torch.device("cpu"), 3, 4, seed=1)
    assert a.device.type == device.type
    torch.testing.assert_close(a.cpu(), b)
