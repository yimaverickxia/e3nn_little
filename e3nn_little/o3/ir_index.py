# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import operator


class IrRep(tuple):
    def __new__(self, l, p):
        assert isinstance(l, int) and l >= 0
        assert p in [-1, 1]
        return tuple.__new__(self, (l, p))

    def __repr__(self):
        p = {+1: 'e', -1: 'o'}[self.p]
        return f"{self.l}{p}"


IrRep.l = property(operator.itemgetter(0))
IrRep.p = property(operator.itemgetter(1))


class IrList(tuple):
    def __new__(self, irs):
        if isinstance(irs, IrList):
            return irs

        out = []
        if isinstance(irs, int):
            out.append((1, IrRep(irs, 1)))
        elif isinstance(irs, IrRep):
            out.append((1, irs))
        else:
            for r in irs:
                if isinstance(r, int):
                    mul, l, p = 1, r, 1
                elif isinstance(r, IrRep):
                    mul = 1
                    l, p = r
                elif len(r) == 2:
                    mul, l = r
                    if isinstance(l, int):
                        p = 1
                    elif len(l) == 2:
                        l, p = l
                    else:
                        p = None
                elif len(r) == 3:
                    mul, l, p = r
                else:
                    mul = None
                    l = None
                    p = None

                assert isinstance(mul, int) and mul >= 0

                out.append((mul, IrRep(l, p)))
        return tuple.__new__(self, out)

    def __getitem__(self, i):
        x = tuple.__getitem__(self, i)
        if isinstance(i, slice):
            return IrList(x)
        return x

    def __add__(self, other):
        return IrList(tuple.__add__(self, other))

    def __radd__(self, other):
        return IrList(tuple.__add__(other, self))

    def simplify(self):
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
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return IrList(out)

    @property
    def dim(self):
        """
        :param Rs: list of triplet (multiplicity, representation order, [parity])
        :return: dimention of the representation
        """
        return sum(mul * (2 * l + 1) for mul, (l, _) in self)

    @property
    def mul_dim(self):
        """
        :param Rs: list of triplet (multiplicity, representation order, [parity])
        :return: number of multiplicities of the representation
        """
        return sum(mul for mul, _ in self)

    @property
    def lmax(self):
        """
        :param Rs: list of triplet (multiplicity, representation order, [parity])
        :return: maximum l present in the signal
        """
        return max(l for mul, (l, _) in self if mul > 0)

    def __repr__(self):
        """
        :param Rs: list of triplet (multiplicity, representation order, [parity])
        :return: simplified version of the same list with the parity
        """
        return ",".join("{}{}".format(f"{mul}x" if mul > 1 else "", ir) for mul, ir in self if mul > 0)


o0 = IrRep(0, -1)
e0 = IrRep(0, 1)
o1 = IrRep(1, -1)
e1 = IrRep(1, 1)
o2 = IrRep(2, -1)
e2 = IrRep(2, 1)
o3 = IrRep(3, -1)
e3 = IrRep(3, 1)
o4 = IrRep(4, -1)
e4 = IrRep(4, 1)
o5 = IrRep(5, -1)
e5 = IrRep(5, 1)
o6 = IrRep(6, -1)
e6 = IrRep(6, 1)
o7 = IrRep(7, -1)
e7 = IrRep(7, 1)
o8 = IrRep(8, -1)
e8 = IrRep(8, 1)
o9 = IrRep(9, -1)
e9 = IrRep(9, 1)
o01 = IrRep(10, -1)
e01 = IrRep(10, 1)
o11 = IrRep(11, -1)
e11 = IrRep(11, 1)
