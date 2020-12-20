# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import pytest
import torch

from e3nn_little import o3
from e3nn_little.nn import (WeightedTensorProduct,
                            GroupedWeightedTensorProduct, Identity,
                            FullyConnectedWeightedTensorProduct)


def test():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    in1 = [(mul, ir, 1.0) for mul, ir in irreps_in1]
    in2 = [(mul, ir, 1.0) for mul, ir in irreps_in2]
    out = [(mul, ir, 1.0) for mul, ir in irreps_out]
    instr = [
        (1, 1, 1, 'uvw', True, 1.0),
    ]
    m = WeightedTensorProduct(in1, in2, out, instr)
    x1 = torch.randn(irreps_in1.dim)
    x2 = torch.randn(irreps_in2.dim)
    m(x1, x2)


def test_wtp():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = FullyConnectedWeightedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    print(m)
    m(torch.randn(irreps_in1.dim), torch.randn(irreps_in2.dim))


def test_gwtp():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = GroupedWeightedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    print(m)
    m(torch.randn(irreps_in1.dim), torch.randn(irreps_in2.dim))


def test_id():
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = Identity(irreps_in, irreps_out)
    print(m)
    m(torch.randn(irreps_in.dim))


@pytest.mark.parametrize('sc', [False, True])
def test_variance(sc):
    n = 1000
    tol = 1.2

    m = WeightedTensorProduct(
        [(12, (0, 1), 1.0)],
        [(3, (0, 1), 1.0)],
        [(7, (0, 1), 1.0)],
        [
            (0, 0, 0, 'uvw', True, 1.0)
        ],
        normalization='component',
        internal_weights=True,
        _specialized_code=sc,
    )
    x = m(torch.randn(n, 12), torch.randn(n, 3)).var(0)
    assert x.mean().log10().abs() < torch.tensor(tol).log10()

    m = WeightedTensorProduct(
        [(12, (0, 1), 1.0), (79, (0, 1), 1.0)],
        [(3, (0, 1), 1.0)],
        [(7, (0, 1), 1.0)],
        [
            (0, 0, 0, 'uvw', True, 1.0),
            (1, 0, 0, 'uvw', True, 2.5),
        ],
        normalization='component',
        internal_weights=True,
        _specialized_code=sc,
    )
    x = m(torch.randn(n, 12 + 79), torch.randn(n, 3)).var(0)
    assert x.mean().log10().abs() < torch.tensor(tol).log10()

    m = WeightedTensorProduct(
        [(12, (0, 1), 1.0), (79, (1, 1), 1.0)],
        [(3, (0, 1), 1.0), (10, (1, 1), 1.0)],
        [(7, (0, 1), 1.0)],
        [
            (0, 0, 0, 'uvw', True, 1.0),
            (1, 1, 0, 'uvw', True, 1.5),
        ],
        normalization='component',
        internal_weights=True,
        _specialized_code=sc,
    )
    x = m(torch.randn(n, 12 + 3 * 79), torch.randn(n, 3 + 10 * 3)).var(0)
    assert x.mean().log10().abs() < torch.tensor(tol).log10()

    m = WeightedTensorProduct(
        [(12, (0, 1), 1.0), (79, (1, 1), 1.0)],
        [(3, (1, 1), 1.0), (10, (1, 1), 1.0)],
        [(7, (1, 1), 1.0)],
        [
            (0, 0, 0, 'uvw', True, 1.0),
            (1, 1, 0, 'uvw', True, 1.5),
        ],
        normalization='component',
        internal_weights=True,
        _specialized_code=sc,
    )
    x = m(torch.randn(n, 12 + 3 * 79), torch.randn(n, 3 * 3 + 10 * 3)).var(0)
    assert x.mean().log10().abs() < torch.tensor(tol).log10()

    m = WeightedTensorProduct(
        [(12, (0, 1), 1.0), (79, (1, 1), 1.0)],
        [(3, (1, 1), 1.0), (10, (2, 1), 3.0)],
        [(7, (1, 1), 2.0)],
        [
            (0, 0, 0, 'uvw', True, 1.0),
            (1, 1, 0, 'uvw', True, 1.5),
        ],
        normalization='component',
        internal_weights=True,
        _specialized_code=sc,
    )
    y = torch.randn(n, 3 * 3 + 10 * 5)
    y[:, 3 * 3:].mul_(3**0.5)
    x = m(torch.randn(n, 12 + 3 * 79), y).var(0) / 2
    assert x.mean().log10().abs() < torch.tensor(tol).log10()

    m = WeightedTensorProduct(
        [(12, (0, 1), 1.0), (79, (1, 1), 1.0)],
        [(3, (1, 1), 1.0), (10, (2, 1), 3.0)],
        [(7, (1, 1), 0.5)],
        [
            (0, 0, 0, 'uvw', True, 1.0),
            (1, 1, 0, 'uvw', True, 1.5),
        ],
        normalization='norm',
        internal_weights=True,
        _specialized_code=sc,
    )
    x = torch.randn(n, 12 + 3 * 79)
    x[:, 12:].div_(3**0.5)

    y = torch.randn(n, 3 * 3 + 10 * 5)
    y[:, :3 * 3].div_(3**0.5)
    y[:, 3 * 3:].div_(5**0.5)
    y[:, 3 * 3:].mul_(3**0.5)

    x = m(x, y).var(0) / 0.5 * 3
    assert x.mean().log10().abs() < torch.tensor(tol).log10()
