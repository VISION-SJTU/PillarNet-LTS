import torch.nn as nn
from typing import List
import spconv.pytorch as spconv
from .pillar_utils import bev_spatial_shape, PillarQueryAndGroup
from .scatter_utils import scatter_max


class PillarMaxPooling(nn.Module):
  def __init__(self, 
               mlps: List[int], 
               pillar_size: float,
               point_cloud_range: List[float], 
               activation='relu'):
    super().__init__()

    self.width, self.height = bev_spatial_shape(point_cloud_range, pillar_size)
    self.groups = PillarQueryAndGroup(pillar_size, point_cloud_range)

    if activation == 'relu':
        activation = nn.ReLU
    elif activation == 'silu':
        activation = nn.SiLU
    else:
      raise NotImplementedError

    shared_mlp = []
    for k in range(len(mlps) - 1):
        shared_mlp.extend([
            nn.Linear(mlps[k], mlps[k + 1], bias=False),
            nn.BatchNorm1d(mlps[k + 1], momentum=0.01, eps=1e-3),
            activation()]
        )
    self.shared_mlps = nn.Sequential(*shared_mlp)

    self.init_weights(weight_init='kaiming')

  def init_weights(self, weight_init='xavier'):
    if weight_init == 'kaiming':
        init_func = nn.init.kaiming_normal_
    elif weight_init == 'xavier':
        init_func = nn.init.xavier_normal_
    elif weight_init == 'normal':
        init_func = nn.init.normal_
    else:
        raise NotImplementedError

    for m in self.modules():
        if isinstance(m, nn.Linear):
            if weight_init == 'normal':
                init_func(m.weight, mean=0, std=0.001)
            else:
                init_func(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

  def forward(self, xyz, xyz_batch_cnt, pts_feature):
    """
    Args:
      xyz: (N1+N2..., 3)
      xyz_batch_cnt: (N1, N2, ...)
      pts_feature: (N1+N2..., C)
    Return:
      pillars: (M1+M2..., 3) [byx]
      pillar_features: (M, C)
    """
    batch = xyz_batch_cnt.shape[0]
    pillar_indices, pillar_set_indices, group_features = self.groups(
        xyz, xyz_batch_cnt, pts_feature)

    group_features = self.shared_mlps(group_features)  # (L, C)
    group_features = group_features.transpose(1, 0).contiguous() # (C, L)

    pillar_features = scatter_max(group_features, pillar_set_indices, pillar_indices.shape[0])   # (C, M)
    pillar_features = pillar_features.transpose(1, 0)   # (M, C)

    return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.height, self.width), batch)
