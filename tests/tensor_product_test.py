# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
from e3nn_little.nn import CustomWeightedTensorProduct, WeightedTensorProduct, GroupedWeightedTensorProduct, Identity
from e3nn_little import o3


def test():
    Rs_in1 = o3.IrList([0, 1, 2])
    Rs_in2 = o3.IrList([0, 1, 2])
    Rs_out = o3.IrList([0, 1, 2])
    in1 = [(mul, ir, 1.0) for mul, ir in Rs_in1]
    in2 = [(mul, ir, 1.0) for mul, ir in Rs_in2]
    out = [(mul, ir, 1.0) for mul, ir in Rs_out]
    instr = [
        (1, 1, 1, 'uvw', True, 1.0),
    ]
    m = CustomWeightedTensorProduct(in1, in2, out, instr)
    x1 = torch.randn(Rs_in1.dim)
    x2 = torch.randn(Rs_in2.dim)
    m(x1, x2)


def test_wtp():
    Rs_in1 = o3.IrList([0, 1, 2])
    Rs_in2 = o3.IrList([0, 1, 2])
    Rs_out = o3.IrList([0, 1, 2])
    m = WeightedTensorProduct(Rs_in1, Rs_in2, Rs_out)
    print(m)
    m(torch.randn(Rs_in1.dim), torch.randn(Rs_in2.dim))


def test_gwtp():
    Rs_in1 = o3.IrList([0, 1, 2])
    Rs_in2 = o3.IrList([0, 1, 2])
    Rs_out = o3.IrList([0, 1, 2])
    m = GroupedWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out)
    print(m)
    m(torch.randn(Rs_in1.dim), torch.randn(Rs_in2.dim))


def test_id():
    Rs_in = o3.IrList([0, 1, 2])
    Rs_out = o3.IrList([0, 1, 2])
    m = Identity(Rs_in, Rs_out)
    print(m)
    m(torch.randn(Rs_in.dim))
