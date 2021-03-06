# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import itertools
import math
import random
from abc import ABC, abstractmethod

import torch
from e3nn_little import o3


class LieGroup(ABC):  # pragma: no cover
    @abstractmethod
    def irrep_indices(self):
        while False:
            yield None

    @abstractmethod
    def irrep(self, r):
        return NotImplemented

    def irrep_dim(self, r):
        return self.irrep(r)(self.identity()).shape[0]

    @abstractmethod
    def compose(self, g1, g2):
        return NotImplemented

    @abstractmethod
    def random(self):
        return NotImplemented

    @abstractmethod
    def identity(self):
        return NotImplemented

    @abstractmethod
    def inverse(self, g):
        return NotImplemented

    @abstractmethod
    def haar(self, g):
        return NotImplemented


class SO3(LieGroup):
    def irrep_indices(self):
        for l in itertools.count():
            yield l

    def irrep(self, r):
        def f(g):
            return o3.wigner_D(r, *g)
        return f

    def irrep_dim(self, r):
        return 2 * r + 1

    def compose(self, g1, g2):
        return o3.compose_angles(*g1, *g2)

    def random(self):
        return o3.rand_angles()

    def identity(self):
        return (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

    def inverse(self, g):
        a, b, c = g
        return (-c, -b, -a)

    def haar(self, g):
        return o3.angles_to_angle(*g)


class O3(LieGroup):
    def irrep_indices(self):
        for l in itertools.count():
            yield o3.Irrep(l, (-1)**l)
            yield o3.Irrep(l, -(-1)**l)

    def irrep(self, r):
        l, p = r
        def f(g):
            *abc, k = g
            return o3.wigner_D(l, *abc) * p**k
        return f

    def irrep_dim(self, r):
        l, _ = r
        return 2 * l + 1

    def compose(self, g1, g2):
        *abc1, p1 = g1
        *abc2, p2 = g2
        return o3.compose_angles(*abc1, *abc2) + ((p1 + p2) % 2,)

    def random(self):
        return o3.rand_angles() + (random.choice([0, 1]),)

    def identity(self):
        return (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), 0)

    def inverse(self, g):
        a, b, c, k = g
        return (-c, -b, -a, k)

    def haar(self, g):
        a, b, c, k = g
        return o3.angles_to_angle(a, b, c) if k % 2 == 0 else math.inf


def is_representation(group: LieGroup, D, eps):
    e = group.identity()
    I = D(e)

    if not torch.allclose(I, torch.eye(len(I), dtype=I.dtype)):
        return False

    for _ in range(4):
        g1 = group.random()
        g2 = group.random()

        g12 = group.compose(g1, g2)
        D12 = D(g12)

        D1D2 = D(g1) @ D(g2)

        if (D12 - D1D2).abs().max().item() > eps * D12.abs().max().item():
            return False
    return True


def is_group(g: LieGroup, eps) -> bool:
    e = g.identity()
    g1 = g.random()
    g2 = g.random()
    g3 = g.random()

    g4 = g.compose(e, g1)
    if not g.haar(g.compose(g4, g.inverse(g1))) < eps:
        return False

    g4 = g.compose(g.compose(g1, g2), g3)
    g5 = g.compose(g1, g.compose(g2, g3))
    if not g.haar(g.compose(g4, g.inverse(g5))) < eps:
        return False

    return True
