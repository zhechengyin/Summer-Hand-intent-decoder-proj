"""First-order FP32 custom autograd operators; no silent fallback."""
import torch
from torch.autograd.function import once_differentiable

from .runtime import runtime


def check(*values):
    if not values or any(not x.is_cuda or x.dtype != torch.float32 for x in values):
        raise ValueError("Phase19 custom operators require CUDA FP32")
    if any(x.device != values[0].device for x in values):
        raise ValueError("All operator tensors must be on the same CUDA device")


class ResidualReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, convolved, residual):
        check(convolved, residual)
        if (convolved.ndim != 3 or residual.ndim != 3
                or convolved.shape[:2] != residual.shape[:2]
                or convolved.shape[2] < residual.shape[2]):
            raise ValueError("Expected B,C,L convolution and B,C,T residual with L >= T")
        convolved, residual = convolved.contiguous(), residual.contiguous()
        b, c, t = residual.shape
        length = convolved.shape[2]
        output = torch.empty_like(residual)
        runtime(output.device).call("residual_forward", [convolved, residual, output],
                                    [output.numel(), t, length], output.numel())
        ctx.save_for_backward(output)
        ctx.length = length
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        output, = ctx.saved_tensors
        b, c, t = output.shape
        dc = output.new_empty((b, c, ctx.length))
        dx = torch.empty_like(output)
        runtime(output.device).call("residual_backward",
                                    [output, grad.contiguous(), dc, dx],
                                    [dc.numel(), t, ctx.length], dc.numel())
        return dc, dx


class LayerNormReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        check(x, weight, bias)
        if x.ndim != 3 or x.shape[1] != 64 or weight.shape != (64,) or bias.shape != (64,):
            raise ValueError("Phase19 normalization is specialized for 64 channels")
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        b, _, t = x.shape
        y = torch.empty_like(x)
        mean = x.new_empty((b, t))
        invstd = torch.empty_like(mean)
        runtime(x.device).call("norm_forward", [x, weight, bias, y, mean, invstd],
                               [b, t], b * t * 32)
        ctx.save_for_backward(x, weight, y, mean, invstd)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, y, mean, invstd = ctx.saved_tensors
        b, _, t = x.shape
        dx, dw, db = (torch.empty_like(x) for _ in range(3))
        runtime(x.device).call("norm_backward",
                               [x, weight, y, mean, invstd, grad.contiguous(), dx, dw, db],
                               [b, t], b * t * 32)
        # Keep parameter reductions native and deterministic; no atomic additions.
        return dx, dw.sum(dim=(0, 2)), db.sum(dim=(0, 2))


class GRUGates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, previous):
        check(u, v, previous)
        if previous.ndim != 2:
            raise ValueError("Expected B,H recurrent state")
        b, h = previous.shape
        if u.shape != (b, 3 * h) or v.shape != u.shape:
            raise ValueError("Expected packed reset/update/candidate projections")
        u, v, previous = u.contiguous(), v.contiguous(), previous.contiguous()
        result = torch.empty_like(previous)
        gates = torch.empty_like(u)
        runtime(u.device).call("gate_forward", [u, v, previous, result, gates],
                               [b, h], b * h)
        ctx.save_for_backward(v, previous, gates)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        v, previous, gates = ctx.saved_tensors
        b, h = previous.shape
        du, dv = torch.empty_like(v), torch.empty_like(v)
        dh = torch.empty_like(previous)
        runtime(v.device).call("gate_backward",
                               [v, previous, gates, grad.contiguous(), du, dv, dh],
                               [b, h], b * h)
        return du, dv, dh
