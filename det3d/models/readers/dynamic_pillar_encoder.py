import torch
from torch import nn
from det3d.ops.pillar_ops.pillar_modules import PillarMaxPooling

from ..registry import READERS


@READERS.register_module
class DynamicPFE(nn.Module):
    def __init__(self,
                 in_channels=5,
                 num_filters=(32, ),
                 pillar_size=0.1,
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        super().__init__()
        self.pillar_size = pillar_size
        self.pc_range = pc_range
        
        assert len(num_filters) > 0
        # only use the relative distance to its pillar centers
        num_filters = [2 + in_channels] + list(num_filters)
        self.pfn_layers = PillarMaxPooling(
            mlps=num_filters,
            pillar_size=pillar_size,
            point_cloud_range=pc_range
        )
        self.height, self.width = self.pfn_layers.height, self.pfn_layers.width

    def forward(self, data, **kwargs):
        pts_xy = []
        pts_batch_cnt = []
        pts_features = []
        for points in data["points"]:
            coors_x = ((points[:, 0] - self.pc_range[0]) / self.pillar_size).floor().int()
            coors_y = ((points[:, 1] - self.pc_range[1]) / self.pillar_size).floor().int()
            
            mask = (coors_x >= 0) & (coors_x < self.width) & \
                (coors_y >= 0) & (coors_y < self.height)
            coors_xy = torch.stack((coors_x[mask], coors_y[mask]), dim=1)
    
            pts_xy.append(coors_xy)
            pts_features.append(points[mask])
            pts_batch_cnt.append(len(coors_xy))

        pts_xy = torch.cat(pts_xy)
        pts_batch_cnt = pts_xy.new_tensor(pts_batch_cnt, dtype=torch.int32)
        pts_features = torch.cat(pts_features)
        
        sparse_tensor = self.pfn_layers(pts_xy, pts_batch_cnt, pts_features)
        return sparse_tensor
