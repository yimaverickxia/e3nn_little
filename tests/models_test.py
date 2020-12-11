# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import pytest
import torch
from e3nn_little.nn.models import Network


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test(dtype):
    torch.set_default_dtype(dtype)

    i = torch.arange(1000)
    z = i % 100
    pos = torch.randn(len(z), 3)
    batch = i % 50

    out = Network()(z, pos, batch)

    assert out.shape == (50, 1)


def test_scalar():
    i = torch.arange(1000)
    z = i % 100
    pos = torch.randn(len(z), 3)
    batch = i % 50

    out = Network(muls=(16,), lmax=0)(z, pos, batch)

    assert out.shape == (50, 1)
