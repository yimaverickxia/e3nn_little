# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from e3nn_little import group, o3
from e3nn_little import math


def reduce_tensor(formula, eps=1e-9, has_parity=None, **kw_Rs):
    """reduce a tensor with symmetries into irreducible representations
    Usage
    Rs, Q = rs.reduce_tensor('ijkl=jikl=ikjl=ijlk', i=[(1, 1)])
    Rs = 0,2,4
    Q = tensor of shape [15, 3, 3, 3, 3]
    """
    for i in kw_Rs:
        if not callable(kw_Rs[i]):
            Rs = o3.convention(kw_Rs[i])
            if has_parity is None:
                has_parity = any(p != 0 for _, _, p in Rs)
            if not has_parity and not all(p == 0 for _, _, p in Rs):
                raise RuntimeError(f'{o3.format_Rs(Rs)} parity has to be specified everywhere or nowhere')
            if has_parity and any(p == 0 for _, _, p in Rs):
                raise RuntimeError(f'{o3.format_Rs(Rs)} parity has to be specified everywhere or nowhere')
            kw_Rs[i] = Rs

    if has_parity is None:
        raise RuntimeError(f'please specify the argument `has_parity`')

    gr = group.O3() if has_parity else group.SO3()

    kw_representations = {}

    def rep(Rs):
        def re(g):
            return math.direct_sum(*[
                gr.irrep((l, p) if has_parity else l)(g)
                for mul, l, p in Rs
                for _ in range(mul)
            ])
        return re

    for i in kw_Rs:
        if callable(kw_Rs[i]):
            kw_representations[i] = lambda g: kw_Rs[i](*g)
        else:
            kw_representations[i] = rep(kw_Rs[i])

    rs_out, Q = group.reduce_tensor(gr, formula, eps, **kw_representations)

    if has_parity:
        Rs = [(mul, l, p) for mul, (l, p) in rs_out]
    else:
        Rs = [(mul, l, 0) for mul, l in rs_out]

    return Rs, Q
