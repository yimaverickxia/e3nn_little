# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import itertools

import torch
from e3nn_little import o3, perm
from e3nn_little.group import O3, SO3, is_representation, reduce
from e3nn_little.math import direct_sum, kron
from e3nn_little.util import torch_default_dtype


def reduce_tensor(formula, eps=1e-9, has_parity=None, **kw_Rs):
    """
    Usage
    Rs, Q = rs.reduce_tensor('ijkl=jikl=ikjl=ijlk', i=[(1, 1)])
    Rs = 0,2,4
    Q = tensor of shape [15, 81]
    """
    dtype = torch.get_default_dtype()
    with torch_default_dtype(torch.float64):
        # reformat `formulas` and make checks
        formulas = [
            (-1 if f.startswith('-') else 1, f.replace('-', ''))
            for f in formula.split('=')
        ]
        s0, f0 = formulas[0]
        assert s0 == 1

        for _s, f in formulas:
            if len(set(f)) != len(f) or set(f) != set(f0):
                raise RuntimeError(f'{f} is not a permutation of {f0}')
            if len(f0) != len(f):
                raise RuntimeError(f'{f0} and {f} don\'t have the same number of indices')

        # `formulas` is a list of (sign, permutation of indices)
        # each formula can be viewed as a permutation of the original formula
        formulas = {(s, tuple(f.index(i) for i in f0)) for s, f in formulas}  # set of generators (permutations)

        # they can be composed, for instance if you have ijk=jik=ikj
        # you also have ijk=jki
        # applying all possible compositions creates an entire group
        while True:
            n = len(formulas)
            formulas = formulas.union([(s, perm.inverse(p)) for s, p in formulas])
            formulas = formulas.union([
                (s1 * s2, perm.compose(p1, p2))
                for s1, p1 in formulas
                for s2, p2 in formulas
            ])
            if len(formulas) == n:
                break  # we break when the set is stable => it is now a group \o/

        # lets clean the `kw_Rs` before checking that they are compatible with the formulas
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

        group = O3() if has_parity else SO3()

        # here we check that each index has one and only one representation
        for _s, p in formulas:
            f = "".join(f0[i] for i in p)
            for i, j in zip(f0, f):
                if i in kw_Rs and j in kw_Rs and kw_Rs[i] != kw_Rs[j]:
                    raise RuntimeError(f'Rs of {i} (Rs={o3.format_Rs(kw_Rs[i])}) and {j} (Rs={o3.format_Rs(kw_Rs[j])}) should be the same')
                if i in kw_Rs:
                    kw_Rs[j] = kw_Rs[i]
                if j in kw_Rs:
                    kw_Rs[i] = kw_Rs[j]

        for i in f0:
            if i not in kw_Rs:
                raise RuntimeError(f'index {i} has not Rs associated to it')

        ide = group.identity()
        dims = {i: len(kw_Rs[i](*ide)) if callable(kw_Rs[i]) else o3.dim(kw_Rs[i]) for i in f0}  # dimension of each index
        full_base = list(itertools.product(*(range(dims[i]) for i in f0)))  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
        # len(full_base) degrees of freedom in an unconstrained tensor

        # but there is constraints given by the group `formulas`
        # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
        base = set()
        for x in full_base:
            # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
            # if x and y are related by a formula
            xs = {(s, tuple(x[i] for i in p)) for s, p in formulas}
            # s * T[x] are all equal for all (s, x) in xs
            # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
            if not (-1, x) in xs:
                # the sign is arbitrary, put both possibilities
                base.add(frozenset({
                    frozenset(xs),
                    frozenset({(-s, x) for s, x in xs})
                }))

        # len(base) is the number of degrees of freedom in the tensor.
        # Now we want to decompose these degrees of freedom into irreps

        base = sorted([sorted([sorted(xs) for xs in x]) for x in base])  # requested for python 3.7 but not for 3.8 (probably a bug in 3.7)

        # First we compute the change of basis (projection) between full_base and base
        d_sym = len(base)
        d = len(full_base)
        Q = torch.zeros(d_sym, d)

        for i, x in enumerate(base):
            x = max(x, key=lambda xs: sum(s for s, x in xs))
            for s, e in x:
                j = full_base.index(e)
                Q[i, j] = s / len(x)**0.5

        assert torch.allclose(Q @ Q.T, torch.eye(d_sym))

        if d_sym == 0:
            return [], torch.zeros(d_sym, d).to(dtype=dtype)

        # We project the representation on the basis `base`
        def representation(g):
            def re(r):
                if callable(r):
                    return r(*g)
                return o3.rep(r, *g)

            m = kron(*(re(kw_Rs[i]) for i in f0))
            return Q @ m @ Q.T

        # And check that after this projection it is still a representation
        assert is_representation(group, representation, eps)

        # The rest of the code simply extract the irreps present in this representation
        Rs_out = []
        A = Q.clone()
        for r in group.irrep_indices():
            if group.irrep(r)(ide).shape[0] > d_sym - o3.dim(Rs_out):
                break

            mul, B, representation = reduce(group, representation, group.irrep(r), eps)
            A = direct_sum(torch.eye(d_sym - B.shape[0]), B) @ A
            A = _round_sqrt(A, eps)

            if has_parity:
                Rs_out += [(mul,) + r]
            else:
                Rs_out += [(mul, r)]

            if o3.dim(Rs_out) == d_sym:
                break

        if o3.dim(Rs_out) != d_sym:
            raise RuntimeError(f'unable to decompose into irreducible representations')

        return o3.simplify(Rs_out), A.to(dtype=dtype)


def _round_sqrt(x, eps):
    # round off x assuming it contains terms of the form +-1/sqrt(N)
    x[x.abs() < eps] = 0
    x = x.sign() / x.pow(2)
    x = x.div(eps).round().mul(eps)
    x = x.sign() / x.abs().sqrt()
    x[torch.isnan(x)] = 0
    x[torch.isinf(x)] = 0
    return x
