# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import torch

from e3nn_little.math import normalize2mom


class FC(torch.nn.Module):
    def __init__(self, hs, act, in_scale='norm', out_scale='zero', out_act=False):
        super().__init__()
        assert isinstance(hs, tuple)
        self.hs = hs
        weights = []

        for h1, h2 in zip(self.hs, self.hs[1:]):
            weights.append(torch.nn.Parameter(torch.randn(h1, h2)))

        self.weights = torch.nn.ParameterList(weights)
        self.act = normalize2mom(act)
        self.in_scale = in_scale
        self.out_scale = out_scale
        self.out_act = out_act
        assert self.in_scale in ['component', 'norm']
        assert self.out_scale in ['component', 'zero']

    def __repr__(self):
        return f"{self.__class__.__name__}{self.hs}"

    def forward(self, x):
        with torch.autograd.profiler.record_function(repr(self)):
            for i, W in enumerate(self.weights):
                if i == 0:  # first layer
                    if self.in_scale == 'component':
                        W = W / x.shape[1]**0.5
                if i > 0:  # not first layer
                    W = W / x.shape[1]**0.5
                if i == len(self.weights) - 1:  # last layer
                    if self.out_scale == 'zero' and not self.out_act:
                        W = W / x.shape[1]**0.5

                x = x @ W

                if i < len(self.weights) - 1:  # not last layer
                    x = self.act(x)
                if i == len(self.weights) - 1 and self.out_act:  # last layer
                    x = self.act(x)

                if i == len(self.weights) - 1:  # last layer
                    if self.out_scale == 'zero' and self.out_act:
                        x = x / W.shape[0]**0.5


            return x
