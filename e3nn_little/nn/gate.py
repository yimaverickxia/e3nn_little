# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, unbalanced-tuple-unpacking, abstract-method
import torch

from e3nn_little import nn, o3
from e3nn_little.math import normalize2mom


class Activation(torch.nn.Module):
    def __init__(self, irreps, acts):
        '''
        Can be used only with scalar fields

        :param acts: list of tuple (multiplicity, activation)
        '''
        super().__init__()

        irreps = irreps.simplify()

        n1 = sum(mul for mul, _ in irreps)
        n2 = sum(mul for mul, _ in acts if mul > 0)

        # normalize the second moment
        acts = [(mul, normalize2mom(act)) for mul, act in acts]

        for i, (mul, act) in enumerate(acts):
            if mul == -1:
                acts[i] = (n1 - n2, act)
                assert n1 - n2 >= 0

        assert n1 == sum(mul for mul, _ in acts)

        irreps = list(irreps)
        i = 0
        while i < len(irreps):
            mul_r, (l, p_r) = irreps[i]
            mul_a, act = acts[i]

            if mul_r < mul_a:
                acts[i] = (mul_r, act)
                acts.insert(i + 1, (mul_a - mul_r, act))

            if mul_a < mul_r:
                irreps[i] = (mul_a, (l, p_r))
                irreps.insert(i + 1, (mul_r - mul_a, (l, p_r)))
            i += 1

        x = torch.linspace(0, 10, 256)

        irreps_out = []
        for (mul, (l, p_in)), (mul_a, act) in zip(irreps, acts):
            assert mul == mul_a

            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                p_act = 1
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                p_act = -1
            else:
                p_act = 0

            p = p_act if p_in == -1 else p_in
            irreps_out.append((mul, (0, p)))

            if p_in != 0 and p == 0:
                raise ValueError("warning! the parity is violated")

        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.acts = acts

    def forward(self, features, dim=-1):
        '''
        :param features: [..., channels, ...]
        '''
        with torch.autograd.profiler.record_function(repr(self)):
            output = []
            index = 0
            for mul, act in self.acts:
                output.append(act(features.narrow(dim, index, mul)))
                index += mul

            if output:
                return torch.cat(output, dim=dim)
            else:
                size = list(features.size())
                size[dim] = 0
                return features.new_zeros(*size)


class Sortcut(torch.nn.Module):
    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(irreps.simplify() for irreps in irreps_outs)
        def key(mul_ir):
            _mul, (l, p) = mul_ir
            return (l, p)
        self.irreps_in = o3.Irreps(sorted((x for irreps in self.irreps_outs for x in irreps), key=key)).simplify()

    def forward(self, x):
        outs = tuple(x.new_zeros(x.shape[:-1] + (irreps.dim,)) for irreps in self.irreps_outs)
        i_in = 0
        for _, (l_in, p_in) in self.irreps_in:
            for irreps_out, out in zip(self.irreps_outs, outs):
                i_out = 0
                for mul_out, (l_out, p_out) in irreps_out:
                    d = mul_out * (2 * l_out + 1)
                    if (l_in, p_in) == (l_out, p_out):
                        out[..., i_out:i_out + d] = x[..., i_in:i_in + d]
                        i_in += d
                    i_out += d
        return outs


class Gate(torch.nn.Module):
    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_nonscalars):
        super().__init__()

        self.sc = Sortcut(irreps_scalars, irreps_gates)
        self.irreps_scalars, self.irreps_gates = self.sc.irreps_outs
        self.irreps_nonscalars = irreps_nonscalars.simplify()
        self.irreps_in = self.sc.irreps_in + self.irreps_nonscalars

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = nn.ElementwiseTensorProduct(irreps_nonscalars, irreps_gates)
        irreps_nonscalars = self.mul.irreps_out

        self.irreps_out = irreps_scalars + irreps_nonscalars

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_scalars} + {self.irreps_gates} + {self.irreps_nonscalars} -> {self.irreps_out})"

    def forward(self, features):
        """
        input of shape [..., dim(self.irreps_in)]
        """
        with torch.autograd.profiler.record_function(repr(self)):
            scalars, gates = self.sc(features)
            nonscalars = features[..., scalars.shape[-1] + gates.shape[-1]:]

            scalars = self.act_scalars(scalars)
            if gates.shape[-1]:
                gates = self.act_gates(gates)
                nonscalars = self.mul(nonscalars, gates)
                features = torch.cat([scalars, nonscalars], dim=-1)
            else:
                features = scalars
            return features
