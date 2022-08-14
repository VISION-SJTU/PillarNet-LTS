import torch
from torch import nn
from ..registry import READERS
from ..utils import build_norm_layer
from det3d.ops.pillar_ops.pillar_modules import PillarMaxPooling


@READERS.register_module
class DynamicPillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features=2,
        num_filters=(32,),
        pillar_size=0.1,
        virtual=False,
        pc_range=(0, -40, -3, 70.4, 40, 1),
        **kwargs
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param pillar_size: (<float>: 3). Size of pillars.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.pc_range = pc_range
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_filters = [6 + num_input_features] + list(num_filters)
        self.pfn_layers = PillarMaxPooling(
            mlps=num_filters,
            pillar_size=pillar_size,
            point_cloud_range=pc_range
        )

        self.virtual = virtual

    @torch.no_grad()
    def absl_to_relative(self, absolute):
        relative = absolute.detach().clone()
        relative[..., 0] -= self.pc_range[0]
        relative[..., 1] -= self.pc_range[1]
        relative[..., 2] -= self.pc_range[2]

        return relative

    def forward(self, example, **kwargs):
        points_list = example.pop("points")
        device = points_list[0].device

        if self.virtual:
            # virtual_point_mask = features[..., -2] == -1
            # virtual_points = features[virtual_point_mask]
            # virtual_points[..., -2] = 1
            # features[..., -2] = 0
            # features[virtual_point_mask] = virtual_points
            raise NotImplementedError

        xyz = []
        xyz_batch_cnt = []
        for points in points_list:
            points = self.absl_to_relative(points)

            xyz_batch_cnt.append(len(points))
            xyz.append(points[:, :3])

        xyz = torch.cat(xyz, dim=0).contiguous()
        pt_features = torch.cat(points_list, dim=0).contiguous()
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)

        sp_tensor = self.pfn_layers(xyz, xyz_batch_cnt, pt_features)
        return sp_tensor
