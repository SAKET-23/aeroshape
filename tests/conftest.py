import pytest
import torch


def available_devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.fixture(params=available_devices())
def device(request):
    return torch.device(request.param)


@pytest.fixture
def f64():
    return torch.float64


def assert_close(a, b, rtol, atol, msg=None):
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=msg)


def pytest_collection_modifyitems(config, items):
    has_gpu = len(available_devices()) > 1
    skip = pytest.mark.skip(reason="no GPU available")
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip)
