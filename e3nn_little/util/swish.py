# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, arguments-differ
import torch


@torch.jit.script
def _swish_jit_fwd(x):
    return x * torch.sigmoid(x)


@torch.jit.script
def _swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class _SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        x = ctx.saved_tensors[0]
        return _swish_jit_bwd(x, grad_output)


def swish(x):
    return _SwishJitAutoFn.apply(x)
