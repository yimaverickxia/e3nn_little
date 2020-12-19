# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math
from typing import List, Tuple

import torch
from e3nn_little import o3
from e3nn_little.util import eval_code


def WeightedTensorProduct(irreps_in1, irreps_in2, irreps_out, normalization='component', internal_weights=True, shared_weights=True):
    irreps_in1 = irreps_in1.simplify()
    irreps_in2 = irreps_in2.simplify()
    irreps_out = irreps_out.simplify()

    in1 = [(mul, ir, 1.0) for mul, ir in irreps_in1]
    in2 = [(mul, ir, 1.0) for mul, ir in irreps_in2]
    out = [(mul, ir, 1.0) for mul, ir in irreps_out]

    instr = [
        (i_1, i_2, i_out, 'uvw', True, 1.0)
        for i_1, (_, (l_1, p_1)) in enumerate(irreps_in1)
        for i_2, (_, (l_2, p_2)) in enumerate(irreps_in2)
        for i_out, (_, (l_out, p_out)) in enumerate(irreps_out)
        if abs(l_1 - l_2) <= l_out <= l_1 + l_2 and p_1 * p_2 == p_out
    ]
    return CustomWeightedTensorProduct(in1, in2, out, instr, normalization, internal_weights, shared_weights)


def GroupedWeightedTensorProduct(irreps_in1, irreps_in2, irreps_out, groups=math.inf, normalization='component', internal_weights=True, shared_weights=True):
    groups = min(groups, min(mul for mul, _ in irreps_in1), min(mul for mul, _ in irreps_out))

    irreps_in1 = [(mul // groups + (g < mul % groups), (l, p)) for mul, (l, p) in irreps_in1 for g in range(groups)]
    irreps_out = [(mul // groups + (g < mul % groups), (l, p)) for mul, (l, p) in irreps_out for g in range(groups)]

    in1 = [(mul, ir, 1.0) for mul, ir in irreps_in1]
    in2 = [(mul, ir, 1.0) for mul, ir in irreps_in2]
    out = [(mul, ir, 1.0) for mul, ir in irreps_out]

    instr = [
        (i_1, i_2, i_out, 'uvw', True, 1.0)
        for i_1, (_, (l_1, p_1)) in enumerate(irreps_in1)
        for i_2, (_, (l_2, p_2)) in enumerate(irreps_in2)
        for i_out, (_, (l_out, p_out)) in enumerate(irreps_out)
        if abs(l_1 - l_2) <= l_out <= l_1 + l_2 and p_1 * p_2 == p_out
        if i_1 % groups == i_out % groups
    ]
    return CustomWeightedTensorProduct(in1, in2, out, instr, normalization, internal_weights, shared_weights)


def ElementwiseTensorProduct(irreps_in1, irreps_in2, normalization='component'):
    irreps_in1 = irreps_in1.simplify()
    irreps_in2 = irreps_in2.simplify()

    assert irreps_in1.num_irreps == irreps_in2.num_irreps

    irreps_in1 = list(irreps_in1)
    irreps_in2 = list(irreps_in2)

    i = 0
    while i < len(irreps_in1):
        mul_1, (l_1, p_1) = irreps_in1[i]
        mul_2, (l_2, p_2) = irreps_in2[i]

        if mul_1 < mul_2:
            irreps_in2[i] = (mul_1, (l_2, p_2))
            irreps_in2.insert(i + 1, (mul_2 - mul_1, (l_2, p_2)))

        if mul_2 < mul_1:
            irreps_in1[i] = (mul_2, (l_1, p_1))
            irreps_in1.insert(i + 1, (mul_1 - mul_2, (l_1, p_1)))
        i += 1

    irreps_out = []
    instr = []
    for i, ((mul, (l_1, p_1)), (mul_2, (l_2, p_2))) in enumerate(zip(irreps_in1, irreps_in2)):
        assert mul == mul_2
        for l in list(range(abs(l_1 - l_2), l_1 + l_2 + 1)):
            i_out = len(irreps_out)
            irreps_out.append((mul, (l, p_1 * p_2)))
            instr += [
                (i, i, i_out, 'uuu', False, 1.0)
            ]

    in1 = [(mul, ir, 1.0) for mul, ir in irreps_in1]
    in2 = [(mul, ir, 1.0) for mul, ir in irreps_in2]
    out = [(mul, ir, 1.0) for mul, ir in irreps_out]

    return CustomWeightedTensorProduct(in1, in2, out, instr, normalization, internal_weights=False)


class Identity(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()

        self.irreps_in = irreps_in.simplify()
        self.irreps_out = irreps_out.simplify()

        assert self.irreps_in == self.irreps_out

        output_mask = torch.cat([
            torch.ones(mul * (2 * l + 1))
            for mul, (l, _p) in self.irreps_out
        ])
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        """
        :param features: [..., dim(irreps_in)]
        :return: [..., dim(irreps_out)]
        """
        return features


class Linear(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, normalization: str = 'component'):
        super().__init__()

        self.irreps_in = irreps_in.simplify()
        self.irreps_out = irreps_out.simplify()

        instr = [
            (i_in, 0, i_out, 'uvw', True, 1.0)
            for i_in, (_, (l_in, p_in)) in enumerate(self.irreps_in)
            for i_out, (_, (l_out, p_out)) in enumerate(self.irreps_out)
            if l_in == l_out and p_in == p_out
        ]
        in1 = [(mul, ir, 1.0) for mul, ir in self.irreps_in]
        out = [(mul, ir, 1.0) for mul, ir in self.irreps_out]
        self.tp = CustomWeightedTensorProduct(in1, [(1, (0, 1), 1.0)], out, instr, normalization, internal_weights=True)

        output_mask = torch.cat([
            torch.ones(mul * (2 * l + 1))
            if any(l_in == l and p_in == p for _, (l_in, p_in) in self.irreps_in)
            else torch.zeros(mul * (2 * l + 1))
            for mul, (l, p) in self.irreps_out
        ])
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} {self.tp.weight_numel} weights)"

    def forward(self, features):
        """
        :param features: [..., dim(irreps_in)]
        :return: [..., dim(irreps_out)]
        """
        ones = features.new_ones(features.shape[:-1] + (1,))
        return self.tp(features, ones)


class CustomWeightedTensorProduct(torch.nn.Module):
    def __init__(
            self,
            in1: List[Tuple[int, Tuple[int, int], float]],
            in2: List[Tuple[int, Tuple[int, int], float]],
            out: List[Tuple[int, Tuple[int, int], float]],
            instr: List[Tuple[int, int, int, str, bool, float]],
            normalization: str = 'component',
            internal_weights: bool = True,
            shared_weights: bool = True,
            _specialized_code=True,
        ):
        """Tensor Product with parametrizable paths

        Parameters
        ----------
        in1
            List of inputs (multiplicity, (l, p), variance).
        in2
            List of inputs (multiplicity, (l, p), variance).
        out
            List of outputs (multiplicity, (l, p), variance).
        instr
            List of instructions (i_1, i_2, i_out, mode, train, path_weight)
            it means: Put `in1[i_1]` otimes `in2[i_2]` into `out[i_out]`
            - mode: determines the way the multiplicities are treated, "uvw" is fully connected
            - train: is this path trained?
            - path weight: how much this path should contribute to the output
        """

        super().__init__()

        assert normalization in ['component', 'norm'], normalization
        self.irreps_in1 = o3.Irreps([(mul, (l, p)) for mul, (l, p), _var in in1])
        self.irreps_in2 = o3.Irreps([(mul, (l, p)) for mul, (l, p), _var in in2])
        self.irreps_out = o3.Irreps([(mul, (l, p)) for mul, (l, p), _var in out])

        in1_var = [var for _, _, var in in1]
        in2_var = [var for _, _, var in in2]
        out_var = [var for _, _, var in out]

        self.shared_weights = shared_weights
        z = '' if self.shared_weights else 'z'

        code = f"""
from typing import List

import torch

@torch.jit.script
def main(x1: torch.Tensor, x2: torch.Tensor, ws: List[torch.Tensor], w3j: List[torch.Tensor]) -> torch.Tensor:
    batch = x1.shape[0]
    out = x1.new_zeros((batch, {self.irreps_out.dim}))
    ein = torch.einsum
"""

        wshapes = []
        wigners = []

        for i_1, (mul_1, (l_1, p_1)) in enumerate(self.irreps_in1):
            index_1 = self.irreps_in1[:i_1].dim
            dim_1 = mul_1 * (2 * l_1 + 1)
            code += f"    x1_{i_1} = x1[:, {index_1}:{index_1+dim_1}].reshape(batch, {mul_1}, {2 * l_1 + 1})\n"
        code += f"\n"

        for i_2, (mul_2, (l_2, p_2)) in enumerate(self.irreps_in2):
            index_2 = self.irreps_in2[:i_2].dim
            dim_2 = mul_2 * (2 * l_2 + 1)
            code += f"    x2_{i_2} = x2[:, {index_2}:{index_2+dim_2}].reshape(batch, {mul_2}, {2 * l_2 + 1})\n"
        code += f"\n"

        last_ss = None

        for i_1, i_2, i_out, mode, weight, path_weight in instr:
            mul_1, (l_1, p_1) = self.irreps_in1[i_1]
            mul_2, (l_2, p_2) = self.irreps_in2[i_2]
            mul_out, (l_out, p_out) = self.irreps_out[i_out]
            dim_1 = mul_1 * (2 * l_1 + 1)
            dim_2 = mul_2 * (2 * l_2 + 1)
            dim_out = mul_out * (2 * l_out + 1)
            index_1 = self.irreps_in1[:i_1].dim
            index_2 = self.irreps_in2[:i_2].dim
            index_out = self.irreps_out[:i_out].dim

            assert p_1 * p_2 == p_out
            assert abs(l_1 - l_2) <= l_out <= l_1 + l_2

            if dim_1 == 0 or dim_2 == 0 or dim_out == 0:
                continue

            # TODO test variance
            alpha = out_var[i_out] / sum(path_weight_ * in1_var[i_1_] * in2_var[i_2_] for i_1_, i_2_, i_out_, _, _, path_weight_ in instr if i_out_ == i_out)

            code += (
                f"    with torch.autograd.profiler.record_function("
                f"'{self.irreps_in1[i_1:i_1+1]} x {self.irreps_in2[i_2:i_2+1]} "
                f"= {self.irreps_out[i_out:i_out+1]} {mode} {weight}'):\n"
            )
            code += f"        s1 = x1_{i_1}\n"
            code += f"        s2 = x2_{i_2}\n"

            assert mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

            c = math.sqrt(alpha * path_weight / {
                'uvw': (mul_1 * mul_2),
                'uvu': mul_2,
                'uvv': mul_1,
                'uuw': mul_1,
                'uuu': 1,
                'uvuv': 1,
            }[mode])

            index_w = len(wshapes)
            if weight:
                wshapes.append({
                    'uvw': (mul_1, mul_2, mul_out),
                    'uvu': (mul_1, mul_2),
                    'uvv': (mul_1, mul_2),
                    'uuw': (mul_1, mul_out),
                    'uuu': (mul_1,),
                    'uvuv': (mul_1, mul_2),
                }[mode])

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
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uvw,zu,zv->zw', ws[{index_w}], s1, s2)\n\n"
                    if mode == 'uvu':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,zu,zv->zu', ws[{index_w}], s1, s2)\n\n"

                    continue

                if l_1 == 0 and l_2 == l_out and mode in ['uvw', 'uvu'] and normalization == 'component' and weight:
                    code += f"        s1 = s1.reshape(batch, {mul_1})\n"

                    if mode == 'uvw':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uvw,zu,zvi->zwi', ws[{index_w}], s1, s2).reshape(batch, {dim_out})\n\n"
                    if mode == 'uvu':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,zu,zvi->zui', ws[{index_w}], s1, s2).reshape(batch, {dim_out})\n\n"

                    continue

                if l_2 == 0 and l_1 == l_out and mode in ['uvw', 'uvu'] and normalization == 'component' and weight:
                    code += f"        s2 = s2.reshape(batch, {mul_2})\n"

                    if mode == 'uvw':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uvw,zui,zv->zwi', ws[{index_w}], s1, s2).reshape(batch, {dim_out})\n\n"
                    if mode == 'uvu':
                        code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,zui,zv->zui', ws[{index_w}], s1, s2).reshape(batch, {dim_out})\n\n"

                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uvw' and normalization == 'component' and weight:
                    # Cl_l_0 = eye(3) / sqrt(2L+1)
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uvw,zui,zvi->zw', ws[{index_w}], s1, s2).reshape(batch, {dim_out})\n\n"
                    continue

                if l_1 == l_2 and l_out == 0 and mode == 'uvu' and normalization == 'component' and weight:
                    # Cl_l_0 = eye(3) / sqrt(2L+1)
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,zui,zvi->zu', ws[{index_w}], s1, s2).reshape(batch, {dim_out})\n\n"
                    continue

                if (l_1, l_2, l_out) == (1, 1, 1) and mode == 'uvw' and normalization == 'component' and weight:
                    # C1_1_1 = levi-civita / sqrt(2)
                    code += f"        s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                    code += f"        s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                    code += f"        s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uvw,zuvi->zwi', ws[{index_w}], torch.cross(s1, s2, dim=3)).reshape(batch, {dim_out})\n\n"
                    continue

                if (l_1, l_2, l_out) == (1, 1, 1) and mode == 'uvu' and normalization == 'component' and weight:
                    # C1_1_1 = levi-civita / sqrt(2)
                    code += f"        s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                    code += f"        s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                    code += f"        s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,zuvi->zui', ws[{index_w}], torch.cross(s1, s2, dim=3)).reshape(batch, {dim_out})\n\n"
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
                code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uvw,ijk,zuvij->zwk', ws[{index_w}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
            if mode == 'uvu':
                assert mul_1 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,ijk,zuvij->zuk', ws[{index_w}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('ijk,zuvij->zuk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
            if mode == 'uvv':
                assert mul_2 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,ijk,zuvij->zvk', ws[{index_w}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('ijk,zuvij->zvk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
            if mode == 'uuw':
                assert mul_1 == mul_2
                assert weight
                code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uw,ijk,zuij->zwk', sw[{index_w}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
            if mode == 'uuu':
                assert mul_1 == mul_2 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}u,ijk,zuij->zuk', sw[{index_w}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('ijk,zuij->zuk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
            if mode == 'uvuv':
                assert mul_1 * mul_2 == mul_out
                if weight:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('{z}uv,ijk,zuvij->zuvk', sw[{index_w}], w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
                else:
                    code += f"        out[:, {index_out}:{index_out+dim_out}] += {c} * ein('ijk,zuvij->zuvk', w3j[{index_w3j}], ss).reshape(batch, {dim_out})\n"
            code += "\n"

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
        self.weight_infos = [
            (i_1, i_2, i_out, mode, path_weight, shape)
            for (i_1, i_2, i_out, mode, path_weight), shape in zip(
                [
                    (i_1, i_2, i_out, mode, path_weight)
                    for i_1, i_2, i_out, mode, weight, path_weight in instr
                    if weight
                ],
                wshapes
            )
        ]

        if internal_weights:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.ParameterDict()
            for i, (i_1, i_2, i_out, mode, path_weight, shape) in enumerate(self.weight_infos):
                mul_1, (l_1, p_1) = self.irreps_in1[i_1]
                mul_2, (l_2, p_2) = self.irreps_in2[i_2]
                mul_out, (l_out, p_out) = self.irreps_out[i_out]
                self.weight[f'{i} l1={l_1} l2={l_2} lout={l_out}'] = torch.nn.Parameter(torch.randn(shape))

        self.to(dtype=torch.get_default_dtype())

    def __repr__(self):
        return "{name}({irreps_in1} x {irreps_in2} -> {irreps_out} {nw} weights)".format(
            name=self.__class__.__name__,
            irreps_in1=self.irreps_in1.simplify(),
            irreps_in2=self.irreps_in2.simplify(),
            irreps_out=self.irreps_out.simplify(),
            nw=self.weight_numel,
        )

    def forward(self, features_1, features_2, weight=None):
        """
        :return:         tensor [..., channel]
        """
        with torch.autograd.profiler.record_function(repr(self)):
            *size, n = features_1.size()
            features_1 = features_1.reshape(-1, n)
            assert n == self.irreps_in1.dim, f"{n} is not {self.irreps_in1.dim}"
            *size2, n = features_2.size()
            features_2 = features_2.reshape(-1, n)
            assert n == self.irreps_in2.dim, f"{n} is not {self.irreps_in2.dim}"
            assert size == size2

            if self.weight_numel:
                if weight is None:
                    weight = list(self.weight.values())
                if torch.is_tensor(weight):
                    ws = []
                    i = 0
                    for shape in self.weight_shapes:
                        d = math.prod(shape)
                        if not self.shared_weights:
                            ws += [weight[:, i:i+d].reshape((-1,) + shape)]
                        else:
                            ws += [weight[i:i+d].reshape(shape)]
                        i += d
                    weight = ws
                if isinstance(weight, list):
                    if not self.shared_weights:
                        weight = [w.reshape(-1, *shape) for w, shape in zip(weight, self.weight_shapes)]
                    else:
                        weight = [w.reshape(*shape) for w, shape in zip(weight, self.weight_shapes)]
            else:
                weight = []

            wigners = [getattr(self, f"C{i}") for i in range(len(self.wigners))]

            if features_1.shape[0] == 0:
                return torch.zeros(*size, self.irreps_out.dim)

            features = self.main(features_1, features_2, weight, wigners)

            return features.reshape(*size, -1)
