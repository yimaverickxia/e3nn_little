# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import operator

from e3nn_little import o3
from e3nn_little.math import direct_sum


class Irrep(tuple):
    def __new__(self, l, p=None):
        if isinstance(l, Irrep):
            return l

        if isinstance(l, str) and p is None:
            name = l.strip()
            l = int(name[:-1])
            assert l >= 0
            p = {
                'e': 1,
                'o': -1,
                'y': (-1)**l,
            }[name[-1]]

        if isinstance(l, tuple) and p is None:
            l, p = l

        assert isinstance(l, int) and l >= 0
        assert p in [-1, 1]
        return tuple.__new__(self, (l, p))

    def __repr__(self):
        p = {+1: 'e', -1: 'o'}[self.p]
        return f"{self.l}{p}"

    def D(self, alpha, beta, gamma, parity=0):
        return o3.wigner_D(self.l, alpha, beta, gamma) * self.p**parity


Irrep.l = property(operator.itemgetter(0))
Irrep.p = property(operator.itemgetter(1))


class Irreps(tuple):
    def __new__(self, irreps):
        if isinstance(irreps, Irreps):
            return irreps

        out = []
        if isinstance(irreps, Irrep):
            out.append((1, Irrep(irreps)))
        elif isinstance(irreps, str):
            for mul_ir in irreps.split('+'):
                if 'x' in mul_ir:
                    mul, ir = mul_ir.split('x')
                    mul = int(mul)
                    ir = Irrep(ir)
                else:
                    mul = 1
                    ir = Irrep(mul_ir)

                assert isinstance(mul, int) and mul >= 0
                out.append((mul, ir))
        else:
            for mul_ir in irreps:
                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)
                elif len(mul_ir) == 3:
                    mul, l, p = mul_ir
                    ir = Irrep(l, p)
                else:
                    mul = None
                    ir = None

                assert isinstance(mul, int) and mul >= 0
                assert ir is not None

                out.append((mul, ir))
        return tuple.__new__(self, out)

    @staticmethod
    def spherical_harmonics(lmax):
        return Irreps([(1, l, (-1)**l) for l in range(lmax + 1)])

    def __getitem__(self, i):
        x = tuple.__getitem__(self, i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __add__(self, other):
        return Irreps(tuple.__add__(self, other))

    def __radd__(self, other):
        return Irreps(tuple.__add__(other, self))

    def simplify(self):
        """
        :param irreps: list of triplet (multiplicity, representation order, [parity])
        :return: An equivalent list with parity = {-1, 0, 1} and neighboring orders consolidated into higher multiplicity.

        Note that simplify does not sort the irreps.
        >>> simplify([(1, 1), (1, 1), (1, 0)])
        [(2, 1, 0), (1, 0, 0)]

        Same order irreps which are seperated from each other are not combined
        >>> simplify([(1, 1), (1, 1), (1, 0), (1, 1)])
        [(2, 1, 0), (1, 0, 0), (1, 1, 0)]

        Parity is normalized to {-1, 0, 1}
        >>> simplify([(1, 1, -1), (1, 1, 50), (1, 0, 0)])
        [(1, 1, -1), (1, 1, 1), (1, 0, 0)]
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    @property
    def dim(self):
        """
        :param irreps: list of triplet (multiplicity, representation order, [parity])
        :return: dimention of the representation
        """
        return sum(mul * (2 * l + 1) for mul, (l, _) in self)

    @property
    def num_irreps(self):
        """
        :param irreps: list of triplet (multiplicity, representation order, [parity])
        :return: number of multiplicities of the representation
        """
        return sum(mul for mul, _ in self)

    @property
    def ls(self):
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self):
        """
        :param irreps: list of triplet (multiplicity, representation order, [parity])
        :return: maximum l present in the signal
        """
        return max(self.ls)

    def __repr__(self):
        """
        :param irreps: list of triplet (multiplicity, representation order, [parity])
        :return: simplified version of the same list with the parity
        """
        return "+".join("{}{}".format(f"{mul}x" if mul > 1 else "", ir) for mul, ir in self if mul > 0)

    def D(self, alpha, beta, gamma, parity=0):
        """
        Representation of O(3). Parity applied (-1)**parity times.
        """
        return direct_sum(*[ir.D(alpha, beta, gamma, parity) for mul, ir in self for _ in range(mul)])
