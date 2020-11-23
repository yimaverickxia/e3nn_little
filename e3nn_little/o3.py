# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math
import os
from functools import lru_cache
from typing import List, Tuple

import torch
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from e3nn_little.eval_code import eval_code

_Jd, _W3j = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))


def _z_rot_mat(angle, l):
    """
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    M = torch.zeros(2 * l + 1, 2 * l + 1, dtype=torch.float64)
    inds = torch.arange(0, 2 * l + 1, 1)
    reversed_inds = torch.arange(2 * l, -1, -1)
    frequencies = torch.arange(l, -l - 1, -1, dtype=torch.float64)
    M[inds, reversed_inds] = torch.sin(frequencies * angle)
    M[inds, inds] = torch.cos(frequencies * angle)
    return M


def irrep(l, alpha, beta, gamma):
    J = _Jd[l]
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def wigner_3j(l1, l2, l3):
    assert abs(l2 - l3) <= l1 <= l2 + l3

    if l1 <= l2 <= l3:
        return _W3j[(l1, l2, l3)].clone()
    if l1 <= l3 <= l2:
        return _W3j[(l1, l3, l2)].transpose(1, 2).mul((-1) ** (l1 + l2 + l3)).clone()
    if l2 <= l1 <= l3:
        return _W3j[(l2, l1, l3)].transpose(0, 1).mul((-1) ** (l1 + l2 + l3)).clone()
    if l3 <= l2 <= l1:
        return _W3j[(l3, l2, l1)].transpose(0, 2).mul((-1) ** (l1 + l2 + l3)).clone()
    if l2 <= l3 <= l1:
        return _W3j[(l2, l3, l1)].transpose(0, 2).transpose(1, 2).clone()
    if l3 <= l1 <= l2:
        return _W3j[(l3, l1, l2)].transpose(0, 2).transpose(0, 1).clone()


def direct_sum(*matrices):
    """
    Direct sum of matrices, put them in the diagonal
    """
    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(*total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x
        i += m
        j += n
    return out


def rep(Rs, alpha, beta, gamma, parity=None):
    """
    Representation of O(3). Parity applied (-1)**parity times.
    """
    Rs = simplify(Rs)
    if parity is None:
        return direct_sum(*[irrep(l, alpha, beta, gamma) for mul, l, _ in Rs for _ in range(mul)])
    else:
        assert all(parity != 0 for _, _, parity in Rs)
        return direct_sum(*[(p ** parity) * irrep(l, alpha, beta, gamma) for mul, l, p in Rs for _ in range(mul)])



def convention(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: conventional version of the same list which always includes parity
    """
    if isinstance(Rs, int):
        return [(1, Rs, 0)]

    out = []
    for r in Rs:
        if isinstance(r, int):
            mul, l, p = 1, r, 0
        elif len(r) == 2:
            (mul, l), p = r, 0
        elif len(r) == 3:
            mul, l, p = r

        assert isinstance(mul, int) and mul >= 0
        assert isinstance(l, int) and l >= 0
        assert p in [0, 1, -1]

        out.append((mul, l, p))
    return out


def simplify(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: An equivalent list with parity = {-1, 0, 1} and neighboring orders consolidated into higher multiplicity.

    Note that simplify does not sort the Rs.
    >>> simplify([(1, 1), (1, 1), (1, 0)])
    [(2, 1, 0), (1, 0, 0)]

    Same order Rs which are seperated from each other are not combined
    >>> simplify([(1, 1), (1, 1), (1, 0), (1, 1)])
    [(2, 1, 0), (1, 0, 0), (1, 1, 0)]

    Parity is normalized to {-1, 0, 1}
    >>> simplify([(1, 1, -1), (1, 1, 50), (1, 0, 0)])
    [(1, 1, -1), (1, 1, 1), (1, 0, 0)]
    """
    out = []
    Rs = convention(Rs)
    for mul, l, p in Rs:
        if out and out[-1][1:] == (l, p):
            out[-1] = (out[-1][0] + mul, l, p)
        elif mul > 0:
            out.append((mul, l, p))
    return out


def dim(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: dimention of the representation
    """
    Rs = convention(Rs)
    return sum(mul * (2 * l + 1) for mul, l, _ in Rs)


def lmax(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: maximum l present in the signal
    """
    return max(l for mul, l, _ in convention(Rs) if mul > 0)


def format_Rs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: simplified version of the same list with the parity
    """
    Rs = convention(Rs)
    d = {
        0: "",
        1: "e",
        -1: "o",
    }
    return ",".join("{}{}{}".format("{}x".format(mul) if mul > 1 else "", l, d[p]) for mul, l, p in Rs if mul > 0)


def cut(features, *Rss, dim_=-1):
    """
    Cut `feaures` according to the list of Rs
    ```
    x = rs.randn(10, Rs1 + Rs2)
    x1, x2 = cut(x, Rs1, Rs2)
    ```
    """
    index = 0
    outputs = []
    for Rs in Rss:
        n = dim(Rs)
        yield features.narrow(dim_, index, n)
        index += n
    assert index == features.shape[dim_]
    return outputs


def spherical_harmonics(Rs, pos, normalization='none'):
    """
    spherical harmonics

    :param Rs: list of L's
    :param pos: tensor of shape [..., 3]
    :return: tensor of shape [..., m]
    """
    Rs = simplify(Rs)
    *size, _ = pos.shape
    pos = pos.reshape(-1, 3)
    d = torch.norm(pos, 2, dim=1)
    pos = pos[d > 0]
    pos = pos / d[d > 0, None]

    # if z > x, rotate x-axis with z-axis
    s = pos[:, 2].abs() > pos[:, 0].abs()
    pos[s] = pos[s] @ pos.new_tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    alpha = torch.atan2(pos[:, 1], pos[:, 0])
    z = pos[:, 2]
    y = pos[:, :2].norm(dim=1)

    sh = _spherical_harmonics_alpha_z_y(Rs, alpha, z, y)

    # rotate back
    sh[s] = sh[s] @ _rep_zx(tuple(Rs), pos.dtype, pos.device)

    if len(d) > len(sh):
        out = sh.new_zeros(len(d), sh.shape[1])
        out[d == 0] = math.sqrt(1 / (4 * math.pi)) * torch.cat([sh.new_ones(1) if l == 0 else sh.new_zeros(2 * l + 1) for mul, l, p in Rs for _ in range(mul)])
        out[d > 0] = sh
        sh = out

    if normalization == 'component':
        sh.mul_(math.sqrt(4 * math.pi))
    if normalization == 'norm':
        sh.mul_(torch.cat([math.sqrt(4 * math.pi / (2 * l + 1)) * sh.new_ones(2 * l + 1) for mul, l, p in Rs for _ in range(mul)]))
    return sh.reshape(*size, sh.shape[1])


@lru_cache()
def _rep_zx(Rs, dtype, device):
    return rep(Rs, 0, -math.pi / 2, 0).to(device=device, dtype=dtype)


def _spherical_harmonics_alpha_z_y(Rs, alpha, z, y):
    """
    spherical harmonics
    """
    Rs = simplify(Rs)
    sha = _spherical_harmonics_alpha(lmax(Rs), alpha.flatten())  # [z, m]
    shz = _spherical_harmonics_z(Rs, z.flatten(), y.flatten())  # [z, l * m]
    out = mul_m_lm(Rs, sha, shz)
    return out.reshape(alpha.shape + (shz.shape[1],))


@torch.jit.script
def _spherical_harmonics_alpha(l: int, alpha: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    the alpha (x, y) component of the spherical harmonics
    (useful to perform fourier transform)

    :param alpha: tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    alpha = alpha.unsqueeze(-1)  # [..., 1]

    m = torch.arange(1, l + 1, dtype=alpha.dtype, device=alpha.device)  # [1, 2, 3, ..., l]
    cos = torch.cos(m * alpha)  # [..., m]

    m = torch.arange(l, 0, -1, dtype=alpha.dtype, device=alpha.device)  # [l, l-1, l-2, ..., 1]
    sin = torch.sin(m * alpha)  # [..., m]

    out = torch.cat([
        math.sqrt(2) * sin,
        torch.ones_like(alpha),
        math.sqrt(2) * cos,
    ], dim=alpha.ndim-1)

    return out  # [..., m]


def _spherical_harmonics_z(Rs, z, y=None):
    """
    the z component of the spherical harmonics
    (useful to perform fourier transform)

    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    Rs = simplify(Rs)
    for _, l, p in Rs:
        assert p in [0, (-1)**l]
    ls = [l for mul, l, _ in Rs]
    return legendre(ls, z, y)  # [..., l * m]


def legendre(ls, z, y=None):
    """
    associated Legendre polynomials

    :param ls: list
    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    if y is None:
        y = (1 - z**2).relu().sqrt()

    return _legendre_genjit(tuple(ls))(z, y)


_legendre_code = """
import torch
from e3nn import rsh

@torch.jit.script
def main(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = z.new_zeros(z.shape + (lsize,))

# fill out

    return out
"""


@lru_cache()
def _legendre_genjit(ls):
    ls = list(ls)
    fill = ""
    i = 0
    for l in ls:
        for m in range(l + 1):
            p = _poly_legendre(l, m)
            formula = " + ".join("{:.25f} * z**{} * y**{}".format(c, zn, yn) for (zn, yn), c in p.items())
            fill += "    l{} = {}\n".format(m, formula)

        for m in range(-l, l + 1):
            fill += "    out[..., {}] = l{}\n".format(i, abs(m))
            i += 1

    code = _legendre_code
    code = code.replace("lsize", str(sum(2 * l + 1 for l in ls)))
    code = code.replace("# fill out", fill)
    return eval_code(code).main


def _poly_legendre(l, m):
    """
    polynomial coefficients of legendre

    y = sqrt(1 - z^2)
    """
    z, y = symbols('z y', real=True)
    return Poly(_sympy_legendre(l, m), domain='R', gens=(z, y)).as_dict()


def _sympy_legendre(l, m):
    """
    en.wikipedia.org/wiki/Associated_Legendre_polynomials
    - remove two times (-1)^m
    - use another normalization such that P(l, -m) = P(l, m)

    y = sqrt(1 - z^2)
    """
    l = Integer(l)
    m = Integer(abs(m))
    z, y = symbols('z y', real=True)
    ex = 1 / (2**l * factorial(l)) * y**m * diff((z**2 - 1)**l, z, l + m)
    ex *= (-1)**l * sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return ex


@torch.jit.script
def mul_m_lm(Rs: List[Tuple[int, int, int]], x_m: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    multiply tensor [..., l * m] by [..., m]
    """
    l_max = x_m.shape[-1] // 2
    out = []
    i = 0
    for mul, l, _ in Rs:
        d = mul * (2 * l + 1)
        x1 = x_lm[..., i: i + d]  # [..., mul * m]
        x1 = x1.reshape(x1.shape[:-1] + (mul, 2 * l + 1))  # [..., mul, m]
        x2 = x_m[..., l_max - l: l_max + l + 1]  # [..., m]
        x2 = x2.reshape(x2.shape[:-1] + (1, 2 * l + 1))  # [..., mul=1, m]
        x = x1 * x2
        x = x.reshape(x.shape[:-2] + (d,))
        out.append(x)
        i += d
    return torch.cat(out, dim=-1)
