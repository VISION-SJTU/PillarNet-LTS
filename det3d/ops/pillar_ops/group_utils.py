from torch import Tensor
from torch.autograd import Function
from . import pillar_cuda


def gather_indice(indices: Tensor, index: Tensor):
    assert index.is_contiguous()
    assert indices.is_contiguous()
    
    outs = indices.new_zeros((index.shape[0], ))
    pillar_cuda.gather_indice_wrapper(index, indices, outs)
    
    return outs


class GatherFeature(Function):
    @staticmethod
    def forward(ctx, features: Tensor, index: Tensor):
        assert index.is_contiguous()
        assert features.is_contiguous()
        
        outs = features.new_zeros((index.shape[0], features.shape[1]))
        pillar_cuda.gather_feature_wrapper(index, features, outs)
        ctx.for_backwards = (features.shape[0], features.shape[1], index)
        return outs

    @staticmethod
    def backward(ctx, grad_outs):
        N, C, index = ctx.for_backwards
        grad_features = grad_outs.new_zeros(N, C)
        
        grad_outs_data = grad_outs.data.contiguous()
        pillar_cuda.gather_feature_grad_wrapper(index, grad_outs_data, grad_features.data)
        
        return grad_features, None

gather_feature = GatherFeature.apply
