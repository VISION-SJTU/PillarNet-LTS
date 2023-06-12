import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from .group_utils import gather_feature, flatten_indices
from . import pillar_cuda


@torch.no_grad()
def generate_pillar_indices(bev_size, point_cloud_range, point_batch_cnt, points):
    pillars, pillar_bev_indices = gen_pillar_indices(points, point_batch_cnt, bev_size, point_cloud_range)
    return pillars, pillar_bev_indices


def bev_spatial_shape(point_cloud_range, pillar_size):
    W = round((point_cloud_range[3] - point_cloud_range[0]) / pillar_size)
    H = round((point_cloud_range[4] - point_cloud_range[1]) / pillar_size)
    return int(H), int(W)


class PillarQueryAndGroup(nn.Module):
    def __init__(self, pillar_size, point_cloud_range):
        super().__init__()

        self.pillar_size = pillar_size
        self.spatial_shape = bev_spatial_shape(point_cloud_range, pillar_size)
        self.point_cloud_range = point_cloud_range

    def forward(self, xy, xy_batch_cnt, pts_features):
        """ batch-wise operation
        Args:
            xy: (N1+N2..., 2)  relative coordinates
            xy_batch_cnt: (N1+N2...)
            pts_features: (N1+N2..., C)
        Return:
            pillar_indices: indices for resulting sparse pillar tensor
            group_features: (L1+L2..., C)
        """
        pillars, pillar_centers, indice_pairs = gen_indice_pairs(
            xy, xy_batch_cnt, self.pillar_size, self.spatial_shape)

        point_set_indices, pillar_set_indices = flatten_indices(indice_pairs)
        group_pts_features = gather_feature(pts_features, point_set_indices)  # (L, C)
        group_pts_xy = gather_feature(xy, point_set_indices)  # (L, 2) [xy]

        group_pillar_centers = gather_feature(pillar_centers, pillar_set_indices)  # (L, 2)  [xy]
        group_pillar_centers = group_pts_xy - group_pillar_centers

        group_features = torch.cat([group_pillar_centers.detach(), 
                                    group_pts_features], dim=1) # (L, C)

        return pillars, pillar_set_indices, group_features


class GenPillarsIndices(Function):
    @staticmethod
    def forward(ctx, xy: Tensor, xy_batch_cnt:Tensor, pillar_size, spatial_shape):
        B = xy_batch_cnt.numel()
        H, W = spatial_shape

        device = xy.device
        pillar_mask = torch.zeros([B, H, W], dtype=torch.bool, device=device)

        pillar_cuda.create_pillar_indices_stack_wrapper(pillar_size, xy, xy_batch_cnt, pillar_mask)

        location = torch.cumsum(pillar_mask.view(-1), 0).int()
        M = location[-1].item()
        pillar_indices = location.view(B, H, W) * pillar_mask - 1
        # create indices (M, 3) [byx]
        pillars = torch.zeros([M, 3], dtype=torch.int32, device=device)
        pillar_cuda.create_pillar_indices_wrapper(pillar_indices, pillars)

        return pillars, pillar_indices

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

gen_pillar_indices = GenPillarsIndices.apply


class GenIndicePairs(Function):
    @staticmethod
    def forward(ctx, xy:Tensor, xy_batch_cnt:Tensor, pillar_size, spatial_shape):
        """
        Args:
            xy: (N1+N2..., 2)
            xy_batch_cnt: (N1, N2, ...)

        Returns:
            pillars: (M1+M2..., 3) [byx]
            pillar_bev_indices: (B, H, W) none(-1)
            pillar_centers: by using pillars yx to calculate centers
            indice_pairs: (N1+N2..., K) neighboring pillars for each point
        """
        assert xy.is_contiguous()
        assert xy_batch_cnt.is_contiguous()
        assert xy.shape[1] == 2
        assert xy_batch_cnt.dtype == torch.int32

        B = xy_batch_cnt.numel()
        H, W = spatial_shape

        pillar_mask = xy.new_zeros([B, H, W], dtype=torch.bool)
        pillar_cuda.create_pillar_indices_stack_wrapper(pillar_size, xy, xy_batch_cnt, pillar_mask)
        location = torch.cumsum(pillar_mask.view(-1), 0).int()
        M = location[-1].item()
        pillar_indices = location.view(B, H, W) * pillar_mask - 1
        # create indices (M, 3) [byx]
        pillars = xy_batch_cnt.new_zeros([M, 3])
        pillar_cuda.create_pillar_indices_wrapper(pillar_indices, pillars)

        indice_pairs = xy_batch_cnt.new_full([xy.shape[0], 1], -1)

        # create pillar center [x y]
        pillar_centers = xy.new_zeros([pillars.shape[0], 2], requires_grad=False)
        pillar_centers[:, 0] = (pillars[:, 2] + 0.5) * pillar_size
        pillar_centers[:, 1] = (pillars[:, 1] + 0.5) * pillar_size

        pillar_cuda.create_pillar_indice_pairs_stack_wrapper(pillar_size, xy, xy_batch_cnt,
                                                             pillar_indices, indice_pairs)

        return pillars, pillar_centers, indice_pairs

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None, None


gen_indice_pairs = GenIndicePairs.apply
