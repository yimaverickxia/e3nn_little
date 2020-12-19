# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from e3nn_little import group, o3
from e3nn_little import math


def reduce_tensor(formula, eps=1e-9, has_parity=True, **kw_Rs):
    """reduce a tensor with symmetries into irreducible representations
    Usage
    Rs, Q = rs.reduce_tensor('ijkl=jikl=ikjl=ijlk', i=[(1, 1)])
    Rs = 0,2,4
    Q = tensor of shape [15, 3, 3, 3, 3]
    """
    gr = group.O3() if has_parity else group.SO3()

    kw_representations = {}

    def rep(Rs):
        def re(g):
            return o3.rep(Rs, *g)
        return re

    for i in kw_Rs:
        if callable(kw_Rs[i]):
            kw_representations[i] = lambda g: kw_Rs[i](*g)
        else:
            kw_representations[i] = rep(kw_Rs[i])

    rs_out, Q = group.reduce_tensor(gr, formula, eps, **kw_representations)

    Rs = o3.IrList(rs_out)

    return Rs, Q
