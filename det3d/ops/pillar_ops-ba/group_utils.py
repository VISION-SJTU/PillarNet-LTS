import torch
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import Function
from . import pillar_cuda


class FlattenIndices(Function):
    @staticmethod
    def forward(ctx, indice_pairs:Tensor):
        """
        Args:
            indice_pairs: (N, K) -1[[none]

        Returns:
            first_indices: (L, )  the flatten indices for the first dimension of indice_pairs.
            second_indices: (L, ) the flatten indices for the second dimension of indice_pairs.
        """
        mask = indice_pairs.view(-1) > -1
        position = torch.cumsum(mask, 0).int()
        L = position[-1].item()
        position = position * mask - 1

        first_indices = indice_pairs.new_zeros(L, requires_grad=False)
        second_indices = indice_pairs.new_zeros(L, requires_grad=False)

        pillar_cuda.flatten_indice_pairs_wrapper(indice_pairs, position, first_indices, second_indices)

        return first_indices, second_indices

    @staticmethod
    def backward(ctx, a=None):
        return None

flatten_indices = FlattenIndices.apply


class GatherFeature(Function):
    @staticmethod
    def forward(ctx, features:Tensor, set_indices:Tensor):
        new_features = features.new_zeros((set_indices.shape[0], features.shape[1]))
        pillar_cuda.gather_feature_wrapper(set_indices, features, new_features)

        ctx.for_backwards = (features.shape[0], features.shape[1], set_indices)
        return new_features

    @staticmethod
    def backward(ctx, grad_out):
        N, C, set_indices = ctx.for_backwards
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())
        grad_out_data = grad_out.data.contiguous()

        pillar_cuda.gather_feature_grad_wrapper(set_indices, grad_out_data, grad_features)
        return grad_features, None

gather_feature = GatherFeature.apply