# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch

from e3nn_little import o3
from e3nn_little.group import reduce_tensor


def test_reduce_tensor_Levi_Civita_symbol():
    torch.set_default_dtype(torch.float64)

    Rs, Q = reduce_tensor('ijk=-ikj=-jik', i=[(1, 1)])
    assert Rs == [(1, 0, 0)]
    r = o3.rand_angles()
    D = o3.irrep(1, *r)
    Q = Q.reshape(3, 3, 3)
    Q1 = torch.einsum('li,mj,nk,ijk', D, D, D, Q)
    assert (Q1 - Q).abs().max() < 1e-10


def test_reduce_tensor_antisymmetric_L2():
    torch.set_default_dtype(torch.float64)

    Rs, Q = reduce_tensor('ijk=-ikj=-jik', i=[(1, 2)])
    assert Rs[0] == (1, 1, 0)
    q = Q[:3].reshape(3, 5, 5, 5)

    r = o3.rand_angles()
    D1 = o3.irrep(1, *r)
    D2 = o3.irrep(2, *r)
    Q1 = torch.einsum('il,jm,kn,zijk->zlmn', D2, D2, D2, q)
    Q2 = torch.einsum('yz,zijk->yijk', D1, q)

    assert (Q1 - Q2).abs().max() < 1e-10
    assert (q + q.transpose(1, 2)).abs().max() < 1e-10
    assert (q + q.transpose(1, 3)).abs().max() < 1e-10
    assert (q + q.transpose(3, 2)).abs().max() < 1e-10


def test_reduce_tensor_elasticity_tensor():
    Rs, _Q = reduce_tensor('ijkl=jikl=klij', i=[(1, 1)])
    assert o3.dim(Rs) == 21


def test_reduce_tensor_elasticity_tensor_parity():
    Rs, _Q = reduce_tensor('ijkl=jikl=klij', i=[(1, 1, -1)])
    assert all(p == 1 for (_, _, p) in Rs)
    assert o3.dim(Rs) == 21
