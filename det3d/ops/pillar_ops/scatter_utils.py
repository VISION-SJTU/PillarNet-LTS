import torch
from torch import Tensor
from torch.autograd import Function, Variable
from . import pillar_cuda


class ScatterMaxFunction(Function):
    @staticmethod
    def forward(ctx, src:Tensor, index:Tensor, M:int):
        """
        Args:
            src: (L, C)
            index: (L, )
        Returns:
            out: (M, C)
        """
        assert index.is_contiguous()
        assert src.is_contiguous()
        L, C = src.size()

        arg = index.new_full((M, C), -1, requires_grad=False)
        out = src.new_zeros(M ,C)
        pillar_cuda.scatter_max_wrapper(index, src, arg, out)

        ctx.for_backwards = (L, C, arg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        L, C, arg = ctx.for_backwards
        grad_src = grad_out.new_zeros(L, C)
        grad_out_data = grad_out.data

        pillar_cuda.scatter_max_grad_wrapper(grad_out_data, arg, grad_src.data)
        return grad_src, None, None

scatter_max = ScatterMaxFunction.apply
