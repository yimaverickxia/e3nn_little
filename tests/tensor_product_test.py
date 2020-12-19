# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
from e3nn_little.nn import CustomWeightedTensorProduct
from e3nn_little import o3


def test():
    Rs_in1 = o3.IrList([0, 1, 2])
    Rs_in2 = o3.IrList([0, 1, 2])
    Rs_out = o3.IrList([0, 1, 2])
    instr = [
        (1, 1, 1, 'uvw', 1.0),
    ]
    m = CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr)
    x1 = torch.randn(Rs_in1.dim)
    x2 = torch.randn(Rs_in2.dim)
    m(x1, x2)
