import torch
from torch import Tensor
from torch.autograd import Function, Variable
from . import pillar_cuda


class ScatterMaxFunction(Function):
    @staticmethod
    def forward(ctx, src:Tensor, index:Tensor, M:int):
        """
        Args:
            src: (C, L)
            index: (L, )
        Returns:c
            out: (C, M)
        """
        assert index.is_contiguous()
        assert src.is_contiguous()
        C, L = src.size()

        arg = index.new_full([C, M], -1, requires_grad=False)
        out = src.new_zeros([C, M])
        pillar_cuda.scatter_max_wrapper(index, src, arg, out)

        ctx.for_backwards = (C, L, arg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        C, L, arg = ctx.for_backwards
        grad_src = Variable(torch.cuda.FloatTensor(C, L).zero_())
        grad_out_data = grad_out.data.contiguous()

        pillar_cuda.scatter_max_grad_wrapper(arg, grad_out_data, grad_src)
        return grad_src, None, None

scatter_max = ScatterMaxFunction.apply
