# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
from e3nn_little.nn import CustomWeightedTensorProduct
from e3nn_little import o3


def test():
    Rs_in1 = [0, 1, 2]
    Rs_in2 = [0, 1, 2]
    Rs_out = [0, 1, 2]
    instr = [
        (1, 1, 1, 'uvw', 1.0),
    ]
    m = CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr)
    x1 = torch.randn(o3.dim(Rs_in1))
    x2 = torch.randn(o3.dim(Rs_in2))
    m(x1, x2)
