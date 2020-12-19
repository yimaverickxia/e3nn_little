# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math
from typing import List, Tuple

import torch
from e3nn_little import o3
from e3nn_little.util import eval_code


def WeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, normalization='component', own_weight=True, weight_batch=False):
    Rs_in1 = Rs_in1.simplify()
    Rs_in2 = Rs_in2.simplify()
    Rs_out = Rs_out.simplify()

    instr = [
        (i_1, i_2, i_out, 'uvw', True)
        for i_1, (_, l_1, p_1) in enumerate(Rs_in1)
        for i_2, (_, l_2, p_2) in enumerate(Rs_in2)
        for i_out, (_, l_out, p_out) in enumerate(Rs_out)
        if abs(l_1 - l_2) <= l_out <= l_1 + l_2 and p_1 * p_2 == p_out
    ]
    return CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr, normalization, own_weight, weight_batch)


def GroupedWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, groups=math.inf, normalization='component', own_weight=True, weight_batch=False):
    groups = min(groups, min(mul for mul, _, _ in Rs_in1), min(mul for mul, _, _ in Rs_out))

    Rs_in1 = [(mul // groups + (g < mul % groups), l, p) for mul, l, p in Rs_in1 for g in range(groups)]
    Rs_out = [(mul // groups + (g < mul % groups), l, p) for mul, l, p in Rs_out for g in range(groups)]

    instr = [
        (i_1, i_2, i_out, 'uvw', True)
        for i_1, (_, l_1, p_1) in enumerate(Rs_in1)
        for i_2, (_, l_2, p_2) in enumerate(Rs_in2)
        for i_out, (_, l_out, p_out) in enumerate(Rs_out)
        if abs(l_1 - l_2) <= l_out <= l_1 + l_2 and p_1 * p_2 == p_out
        if i_1 % groups == i_out % groups
    ]
    return CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr, normalization, own_weight, weight_batch)


def ElementwiseTensorProduct(Rs_in1, Rs_in2, normalization='component'):
    Rs_in1 = Rs_in1.simplify()
    Rs_in2 = Rs_in2.simplify()

    assert Rs_in1.mul_dim == Rs_in2.mul_dim

    Rs_in1 = list(Rs_in1)
    Rs_in2 = list(Rs_in2)

    i = 0
    while i < len(Rs_in1):
        mul_1, (l_1, p_1) = Rs_in1[i]
        mul_2, (l_2, p_2) = Rs_in2[i]

        if mul_1 < mul_2:
            Rs_in2[i] = (mul_1, (l_2, p_2))
            Rs_in2.insert(i + 1, (mul_2 - mul_1, (l_2, p_2)))

        if mul_2 < mul_1:
            Rs_in1[i] = (mul_2, (l_1, p_1))
            Rs_in1.insert(i + 1, (mul_1 - mul_2, (l_1, p_1)))
        i += 1

    Rs_out = []
    instr = []
    for i, ((mul, (l_1, p_1)), (mul_2, (l_2, p_2))) in enumerate(zip(Rs_in1, Rs_in2)):
        assert mul == mul_2
        for l in list(range(abs(l_1 - l_2), l_1 + l_2 + 1)):
            i_out = len(Rs_out)
            Rs_out.append((mul, l, p_1 * p_2))
            instr += [
                (i, i, i_out, 'uuu', False)
            ]

    return CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr, normalization, own_weight=False)


class Identity(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out):
        super().__init__()

        self.Rs_in = Rs_in.simplify()
        self.Rs_out = Rs_out.simplify()

        assert self.Rs_in == self.Rs_out

        output_mask = torch.cat([
            torch.ones(mul * (2 * l + 1))
            for mul, l, p in self.Rs_out
        ])
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.Rs_in} -> {self.Rs_out})"

    def forward(self, features):
        """
        :param features: [..., dim(Rs_in)]
        :return: [..., dim(Rs_out)]
        """
        return features


class Linear(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, normalization: str = 'component'):
        super().__init__()

        self.Rs_in = Rs_in.simplify()
        self.Rs_out = Rs_out.simplify()

        instr = [
            (i_in, 0, i_out, 'uvw', True)
            for i_in, (_, (l_in, p_in)) in enumerate(self.Rs_in)
            for i_out, (_, (l_out, p_out)) in enumerate(self.Rs_out)
            if l_in == l_out and p_in == p_out
        ]
        self.tp = CustomWeightedTensorProduct(self.Rs_in, [(1, 0, 1)], self.Rs_out, instr, normalization, own_weight=True)

        output_mask = torch.cat([
            torch.ones(mul * (2 * l + 1))
            if any(l_in == l and p_in == p for _, (l_in, p_in) in self.Rs_in)
            else torch.zeros(mul * (2 * l + 1))
            for mul, (l, p) in self.Rs_out
        ])
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.Rs_in} -> {self.Rs_out} {self.tp.weight_numel} weights)"

    def forward(self, features):
        """
        :param features: [..., dim(Rs_in)]
        :return: [..., dim(Rs_out)]
        """
        ones = features.new_ones(features.shape[:-1] + (1,))
        return self.tp(features, ones)


class CustomWeightedTensorProduct(torch.nn.Module):
    def __init__(
            self,
            Rs_in1,
            Rs_in2,
            Rs_out,
            instr: List[Tuple[int, int, int, str, bool]],
            normalization: str = 'component',
            own_weight: bool = True,
            weight_batch: bool = False,
            _specialized_code=True,
        ):
        """
        Create a Tensor Product operation that has each of his path weighted by a parameter.
        `instr` is a list of instructions.
        An instruction if of the form (i_1, i_2, i_out, mode, weight)
        it means "Put `Rs_in1[i_1] otimes Rs_in2[i_2] into Rs_out[i_out]"
        `mode` determines the way the multiplicities are treated.
        `weight` determines if weights are learned.
        The default mode should be 'uvw', meaning that all paths are created.
        """

        super().__init__()

        assert normalization in ['component', 'norm'], normalization
        self.Rs_in1 = o3.IrList(Rs_in1)
        self.Rs_in2 = o3.IrList(Rs_in2)
        self.Rs_out = o3.IrList(Rs_out)

        self.weight_batch = weight_batch
        z = 'z' if self.weight_batch else ''

        code = f"""
from typing import List

import torch

@torch.jit.script
def main(x1: torch.Tensor, x2: torch.Tensor, ws: List[torch.Tensor], w3j: List[torch.Tensor]) -> torch.Tensor:
    batch = x1.shape[0]
    out = x1.new_zeros((batch, {self.Rs_out.dim}))
    ein = torch.einsum
"""

        wshapes = []
        wigners = []
        count = [0 for _ in range(self.Rs_out.dim)]

        instr = sorted(instr)  # for optimization

        for i_1, (mul_1, (l_1, p_1)) in enumerate(self.Rs_in1):
            index_1 = self.Rs_in1[:i_1].dim
            dim_1 = mul_1 * (2 * l_1 + 1)
            code += f"    x1_{i_1} = x1[:, {index_1}:{index_1+dim_1}].reshape(batch, {mul_1}, {2 * l_1 + 1})\n"
        code += f"\n"

        for i_2, (mul_2, (l_2, p_2)) in enumerate(self.Rs_in2):
            index_2 = self.Rs_in2[:i_2].dim
            dim_2 = mul_2 * (2 * l_2 + 1)
            code += f"    x2_{i_2} = x2[:, {index_2}:{index_2+dim_2}].reshape(batch, {mul_2}, {2 * l_2 + 1})\n"
        code += f"\n"

        last_ss = None

        for i_1, i_2, i_out, mode, weight in instr:
            mul_1, (l_1, p_1) = self.Rs_in1[i_1]
            mul_2, (l_2, p_2) = self.Rs_in2[i_2]
            mul_out, (l_out, p_out) = self.Rs_out[i_out]
            dim_1 = mul_1 * (2 * l_1 + 1)
            dim_2 = mul_2 * (2 * l_2 + 1)
            dim_out = mul_out * (2 * l_out + 1)
            index_1 = self.Rs_in1[:i_1].dim
            index_2 = self.Rs_in2[:i_2].dim
            index_out = self.Rs_out[:i_out].dim

            assert p_1 * p_2 == p_out
            assert abs(l_1 - l_2) <= l_out <= l_1 + l_2

            if dim_1 == 0 or dim_2 == 0 or dim_out == 0:
                continue

            code += (
                f"    with torch.autograd.profiler.record_function("
                f"'{self.Rs_in1[i_1:i_1+1]} x {self.Rs_in2[i_2:i_2+1]} "
                f"= {self.Rs_out[i_out:i_out+1]} {mode} {weight}'):\n"
            )
            code += f"        s1 = x1_{i_1}\n"
            code += f"        s2 = x2_{i_2}\n"

            assert mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

            if _specialized_code:
                # optimized code for special cases:
                # 0 x 0 = 0
                # 0 x L = L
                # L x 0 = L
                # L x L = 0
                # 1 x 1 = 1

                if (l_1, l_2, l_out) == (0, 0, 0) and mode in ['uvw', 'uvu'] and normalization in ['component', 'norm'] and weight:
                    code += f"        s1 = s1.reshape(batch, {mul_1})\n"
                    code += f"        s2 = s2.reshape(batch, {mul_2})\n"

                    if mode == 'uvw':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uvw,zu,zv->zw', ws[{len(wshapes)}], s1, s2)\n"
                        code += "\n"

                        wshapes += [(mul_1, mul_2, mul_out)]

                        for pos in range(index_out, index_out + dim_out):
                            count[pos] += mul_1 * mul_2
                    if mode == 'uvu':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,zu,zv->zu', ws[{len(wshapes)}], s1, s2)\n"
                        code += "\n"

                        wshapes += [(mul_1, mul_2)]

                        for pos in range(index_out, index_out + dim_out):
                            count[pos] += mul_2

                    continue

                if l_1 == 0 and l_2 == l_out and mode in ['uvw', 'uvu'] and normalization == 'component' and weight:
                    code += f"        s1 = s1.reshape(batch, {mul_1})\n"

                    if mode == 'uvw':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uvw,zu,zvi->zwi', ws[{len(wshapes)}], s1, s2).reshape(batch, {dim_out})\n"
                        code += "\n"

                        wshapes += [(mul_1, mul_2, mul_out)]

                        for pos in range(index_out, index_out + dim_out):
                            count[pos] += mul_1 * mul_2
                    if mode == 'uvu':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,zu,zvi->zui', ws[{len(wshapes)}], s1, s2).reshape(batch, {dim_out})\n"
                        code += "\n"

                        wshapes += [(mul_1, mul_2)]

                        for pos in range(index_out, index_out + dim_out):
                            count[pos] += mul_2

                    continue

                if l_2 == 0 and l_1 == l_out and mode in ['uvw', 'uvu'] and normalization == 'component' and weight:
                    code += f"        s2 = s2.reshape(batch, {mul_2})\n"

                    if mode == 'uvw':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uvw,zui,zv->zwi', ws[{len(wshapes)}], s1, s2).reshape(batch, {dim_out})\n"
                        code += "\n"

                        wshapes += [(mul_1, mul_2, mul_out)]

                        for pos in range(index_out, index_out + dim_out):
                            count[pos] += mul_1 * mul_2
                    if mode == 'uvu':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,zui,zv->zui', ws[{len(wshapes)}], s1, s2).reshape(batch, {dim_out})\n"
                        code += "\n"

                        wshapes += [(mul_1, mul_2)]

                        for pos in range(index_out, index_out + dim_out):
                            count[pos] += mul_2

                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uvw' and normalization == 'component' and weight:
                    # Cl_l_0 = eye(3) / sqrt(2L+1)
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uvw,zui,zvi->zw', ws[{len(wshapes)}], s1, s2).reshape(batch, {dim_out})\n"
                    code += "\n"

                    wshapes += [(mul_1, mul_2, mul_out)]

                    for pos in range(index_out, index_out + dim_out):
                        count[pos] += mul_1 * mul_2
                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uvu' and normalization == 'component' and weight:
                    # Cl_l_0 = eye(3) / sqrt(2L+1)
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,zui,zvi->zu', ws[{len(wshapes)}], s1, s2).reshape(batch, {dim_out})\n"
                    code += "\n"

                    wshapes += [(mul_1, mul_2)]

                    for pos in range(index_out, index_out + dim_out):
                        count[pos] += mul_2
                    continue

                if (l_1, l_2, l_out) == (1, 1, 1) and mode == 'uvw' and normalization == 'component' and weight:
                    # C1_1_1 = levi-civita / sqrt(2)
                    code += f"        s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                    code += f"        s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                    code += f"        s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uvw,zuvi->zwi', ws[{len(wshapes)}], torch.cross(s1, s2, dim=3)).reshape(batch, {dim_out})\n"
                    code += "\n"

                    wshapes += [(mul_1, mul_2, mul_out)]

                    for pos in range(index_out, index_out + dim_out):
                        count[pos] += mul_1 * mul_2
                    continue

                if (l_1, l_2, l_out) == (1, 1, 1) and mode == 'uvu' and normalization == 'component' and weight:
                    # C1_1_1 = levi-civita / sqrt(2)
                    code += f"        s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                    code += f"        s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                    code += f"        s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,zuvi->zui', ws[{len(wshapes)}], torch.cross(s1, s2, dim=3)).reshape(batch, {dim_out})\n"
                    code += "\n"

                    wshapes += [(mul_1, mul_2)]

                    for pos in range(index_out, index_out + dim_out):
                        count[pos] += mul_2
                    continue

            if last_ss != (i_1, i_2, mode[:2]):
                if mode[:2] == 'uv':
                    code += f"        ss = ein('zui,zvj->zuvij', s1, s2)\n"
                if mode[:2] == 'uu':
                    code += f"        ss = ein('zui,zuj->zuij', s1, s2)\n"
                last_ss = (i_1, i_2, mode[:2])

            if (l_1, l_2, l_out) in wigners:
                index_w3j = wigners.index((l_1, l_2, l_out))
            else:
                index_w3j = len(wigners)
                wigners += [(l_1, l_2, l_out)]

            if mode == 'uvw':
                assert weight
                code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uvw,ijk,zuvij->zwk', ws[{len(wshapes)}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                wshapes += [(mul_1, mul_2, mul_out)]
                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_1 * mul_2

            if mode == 'uvu':
                assert mul_1 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,ijk,zuvij->zuk', ws[{len(wshapes)}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                    wshapes += [(mul_1, mul_2)]
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('ijk,zuvij->zuk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_2

            if mode == 'uvv':
                assert mul_2 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,ijk,zuvij->zvk', ws[{len(wshapes)}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                    wshapes += [(mul_1, mul_2)]
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('ijk,zuvij->zvk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_1

            if mode == 'uuw':
                assert mul_1 == mul_2
                assert weight
                code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uw,ijk,zuij->zwk', sw[{len(wshapes)}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                wshapes += [(mul_1, mul_out)]

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_1

            if mode == 'uuu':
                assert mul_1 == mul_2 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}u,ijk,zuij->zuk', sw[{len(wshapes)}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                    wshapes += [(mul_1,)]
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('ijk,zuij->zuk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += 1

            if mode == 'uvuv':
                assert mul_1 * mul_2 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('{z}uv,ijk,zuvij->zuvk', sw[{len(wshapes)}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                    wshapes += [(mul_1, mul_2)]
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += ein('ijk,zuvij->zuvk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += 1

            code += "\n"

        if count:
            ilast = 0
            clast = count[0]
            for i, c in enumerate(count):
                if clast != c:
                    if clast > 1:
                        code += f"    out[:, {ilast}:{i}].div_({clast ** 0.5})\n"
                    clast = c
                    ilast = i
            if clast > 1:
                code += f"    out[:, {ilast}:].div_({clast ** 0.5})\n"

        code += f"    return out"

        self.code = code
        self.main = eval_code(self.code).main

        # w3j
        self.wigners = wigners
        for i, (l_1, l_2, l_out) in enumerate(self.wigners):
            wig = o3.wigner_3j(l_1, l_2, l_out)

            if normalization == 'component':
                wig *= (2 * l_out + 1) ** 0.5
            if normalization == 'norm':
                wig *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5

            self.register_buffer(f"C{i}", wig)

        # weights
        self.weight_shapes = wshapes
        self.weight_numel = sum(math.prod(shape) for shape in self.weight_shapes)
        weight_infos = [
            (i_1, i_2, i_out, mode, shape)
            for (i_1, i_2, i_out, mode), shape in zip(
                [
                    (i_1, i_2, i_out, mode)
                    for i_1, i_2, i_out, mode, weight in instr
                    if weight
                ],
                wshapes
            )
        ]

        if own_weight:
            assert not self.weight_batch, "weight_batch and own_weight are incompatible"
            self.weight = torch.nn.ParameterDict()
            for i, (i_1, i_2, i_out, mode, shape) in enumerate(weight_infos):
                mul_1, (l_1, p_1) = self.Rs_in1[i_1]
                mul_2, (l_2, p_2) = self.Rs_in2[i_2]
                mul_out, (l_out, p_out) = self.Rs_out[i_out]
                self.weight[f'{i} l1={l_1} l2={l_2} lout={l_out}'] = torch.nn.Parameter(torch.randn(shape))

        self.to(dtype=torch.get_default_dtype())

    def __repr__(self):
        return "{name}({Rs_in1} x {Rs_in2} -> {Rs_out} {nw} weights)".format(
            name=self.__class__.__name__,
            Rs_in1=self.Rs_in1.simplify(),
            Rs_in2=self.Rs_in2.simplify(),
            Rs_out=self.Rs_out.simplify(),
            nw=self.weight_numel,
        )

    def forward(self, features_1, features_2, weight=None):
        """
        :return:         tensor [..., channel]
        """
        with torch.autograd.profiler.record_function(repr(self)):
            *size, n = features_1.size()
            features_1 = features_1.reshape(-1, n)
            assert n == self.Rs_in1.dim, f"{n} is not {self.Rs_in1.dim}"
            *size2, n = features_2.size()
            features_2 = features_2.reshape(-1, n)
            assert n == self.Rs_in2.dim, f"{n} is not {self.Rs_in2.dim}"
            assert size == size2

            if self.weight_numel:
                if weight is None:
                    weight = list(self.weight.values())
                if torch.is_tensor(weight):
                    ws = []
                    i = 0
                    for shape in self.weight_shapes:
                        d = math.prod(shape)
                        if self.weight_batch:
                            ws += [weight[:, i:i+d].reshape((-1,) + shape)]
                        else:
                            ws += [weight[i:i+d].reshape(shape)]
                        i += d
                    weight = ws
            else:
                weight = []

            wigners = [getattr(self, f"C{i}") for i in range(len(self.wigners))]

            if features_1.shape[0] == 0:
                return torch.zeros(*size, self.Rs_out.dim)

            features = self.main(features_1, features_2, weight, wigners)

            return features.reshape(*size, -1)
