# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
from e3nn_little import group, o3


def reduce_tensor(formula, eps=1e-9, has_parity=True, **kw_irreps):
    """reduce a tensor with symmetries into irreducible representations
    Usage
    irreps, Q = rs.reduce_tensor('ijkl=jikl=ikjl=ijlk', i=[(1, 1)])
    irreps = 0,2,4
    Q = tensor of shape [15, 3, 3, 3, 3]
    """
    gr = group.O3() if has_parity else group.SO3()

    kw_representations = {}

    for i in kw_irreps:
        if callable(kw_irreps[i]):
            kw_representations[i] = lambda g: kw_irreps[i](*g)
        else:
            kw_representations[i] = lambda g: o3.Irreps(kw_irreps[i]).D(*g)

    irreps, Q = group.reduce_tensor(gr, formula, eps, **kw_representations)

    if has_parity:
        irreps = o3.Irreps(irreps)
    else:
        irreps = o3.Irreps([(mul, l, 1) for mul, l in irreps])

    return irreps, Q
