import torch
import torch.nn as nn
from .group_utils import gather_feature, gather_indice
from . import pillar_cuda


def bev_spatial_shape(pillar_size, point_cloud_range):
    W = round((point_cloud_range[3] - point_cloud_range[0]) / pillar_size)
    H = round((point_cloud_range[4] - point_cloud_range[1]) / pillar_size)
    return int(H), int(W)


class PillarQueryAndGroup(nn.Module):
    def __init__(self, pillar_size, point_cloud_range):
        super().__init__()

        self.pillar_size = pillar_size
        self.spatial_shape = bev_spatial_shape(pillar_size, point_cloud_range)
        self.x_offset = pillar_size / 2. + point_cloud_range[0]
        self.y_offset = pillar_size / 2. + point_cloud_range[1]

    def forward(self, pts_xy, pts_batch_cnt, pts_features):
        """ batch-wise operation
        Args:
            pts_xy: (N1+N2..., 2)  relative coordinates
            pts_batch_cnt: (N1+N2...)
            pts_features: (N1+N2..., C)
        Return:
            pillar_indices: indices for resulting sparse pillar tensor
            point_pillar_indices: the pillar index of valid points, has shape
                (L1+L2...)
            point_pillar_features: (L1+L2..., C)
        """
        batch = pts_batch_cnt.numel()
        height, width = self.spatial_shape
        
        num_points = pts_xy.shape[0]
        point_pillar_index = pts_batch_cnt.new_full((num_points,), -1)
        pillars_mask = pts_batch_cnt.new_zeros((batch, height, width), dtype=torch.bool)
        pillar_cuda.create_point_pillar_index_stack_wrapper(
            pts_xy, pts_batch_cnt, pillars_mask, point_pillar_index)
        
        pillars_position = torch.cumsum(pillars_mask.view(-1), dim=0, dtype=torch.int32)
        out_dim = pillars_position[-1].item()
        pillars_position = pillars_position.view(batch, height, width) * pillars_mask - 1
        
        pillar_indices = pts_batch_cnt.new_zeros(out_dim, 3)
        pillar_cuda.create_pillar_indices_wrapper(pillars_position, pillar_indices)
        
        point_pillar_indices = gather_indice(pillars_position.view(-1), point_pillar_index)
        pillar_centers = torch.stack((pillar_indices[:, 2] * self.pillar_size + self.x_offset, 
                                      pillar_indices[:, 1] * self.pillar_size + self.y_offset), dim=1)
        point_pillar_centers = gather_feature(pillar_centers, point_pillar_indices)
        point_pillar_centers = pts_features[:, :2] - point_pillar_centers
        
        point_pillar_features = torch.cat([point_pillar_centers, pts_features], dim=1) # (L, C)
        
        return pillar_indices, point_pillar_indices, point_pillar_features