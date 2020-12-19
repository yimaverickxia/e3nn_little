# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
from e3nn_little.nn import CustomWeightedTensorProduct, WeightedTensorProduct, GroupedWeightedTensorProduct, Identity
from e3nn_little import o3


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
    m = CustomWeightedTensorProduct(in1, in2, out, instr)
    x1 = torch.randn(irreps_in1.dim)
    x2 = torch.randn(irreps_in2.dim)
    m(x1, x2)


def test_wtp():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = WeightedTensorProduct(irreps_in1, irreps_in2, irreps_out)
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
