import torch
from torch import nn
import numpy as np
from spconv.pytorch import SparseSequential, SparseConv2d, SparseConvTensor

from ..registry import SECOND_STAGE
from ..utils import build_norm_layer

from det3d.core.bbox.box_torch_ops import (
    center_to_grid_box2d
)
from det3d.core.utils.center_utils import (
    bilinear_interpolate_torch,
)

    
@SECOND_STAGE.register_module
class BEVFeature(nn.Module):
    def __init__(self, feature_sources, pillar_size, pc_range, out_stride=4, grid_size=7,
                 in_channels=256, share_channels=64, backbone_channels=None, backbone_strides=None):
        super().__init__()
        self.pillar_size = pillar_size
        self.point_cloud_range = pc_range
        self.grid_size = grid_size

        self.lat_conv = nn.ModuleList()
        self.lat_conv_name = []
        self.lat_tensor_type = []

        opt_strides = [1, 2, 4, 8]
        opt_out_stage_name = ['conv1', 'conv2', 'conv3', 'conv4']
        opt_out_channels = [32, 64, 128, 256]
        
        assert out_stride in opt_strides
        out_name = opt_out_stage_name[opt_strides.index(out_stride)]
        out_channels = opt_out_channels[opt_strides.index(out_stride)]
        assert out_channels <= backbone_channels[out_name]

        c_in = 0
        stride = int(backbone_strides['conv4'] / out_stride)
        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.top_down_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, stride, stride=stride, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU()
        )
        c_in += out_channels

        for src_name in feature_sources:
            if src_name not in ['conv1', 'conv2', 'conv3', 'conv4']: 
                continue

            in_channels = backbone_channels[src_name]
            stride = backbone_strides[src_name] / out_stride

            if stride > 1 or (out_stride == 8 and stride == 1):
                stride = int(stride)
                norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, stride, stride=stride, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU()
                )
                self.lat_tensor_type.append('dense')
            else:
                stride = int(np.round(1 / stride))
                norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)
                deblock = SparseSequential(
                    SparseConv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=True),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU()
                )
                self.lat_tensor_type.append('sparse')

            c_in += out_channels
            self.lat_conv.append(deblock)
            self.lat_conv_name.append(src_name)

        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c_in, share_channels, 3, stride=1, padding=1, bias=True),
            build_norm_layer(norm_cfg, share_channels)[1],
            nn.ReLU(),
        )
        self.out_stride = out_stride
        
    def get_pooling_points(self, rois):
        batch_size, num_rois, _ = rois.shape
        rois = rois.view(-1, rois.shape[-1])

        center2d = rois[:, :2]
        # height = rois[:, 2:3]
        dim2d = rois[:, 3:5]
        rotation_y = rois[:, -1]
        
        grid_points = center_to_grid_box2d(
            center2d, dim2d, rotation_y, (self.grid_size, self.grid_size))

        return grid_points.view(batch_size, num_rois, -1, grid_points.shape[-1])

    def interpolate_from_bev_features(self, pooling_points, bev_features, bev_stride):
        """
        Args:
            pooling_points: (batch_size, num_rois, 7x7, 2)
            bev_features: (batch, channels, height, weight)

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        batch_size, num_rois, grid_size = pooling_points.shape[:3]
        x_idxs = (pooling_points[..., 0] - self.point_cloud_range[0]) / (bev_stride * self.pillar_size)
        y_idxs = (pooling_points[..., 1] - self.point_cloud_range[1]) / (bev_stride * self.pillar_size)

        pts_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k].view(-1)
            cur_y_idxs = y_idxs[k].view(-1)
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            pts_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            pts_bev_features_list.append(pts_bev_features.view(num_rois, grid_size, -1))

        pts_bev_features = torch.stack(pts_bev_features_list, dim=0)  # (batch_size, num_rois, 7x7, C)
        return pts_bev_features

    def forward(self, example):
        batch_size, num_rois = example['rois'].shape[:2]

        # step 1: aggregate multi-scale bev features with stride <= 4
        top_down_features = self.top_down_conv(example['bev_feature'])
        multi_scale_features = [top_down_features]

        for k, src_name in enumerate(self.lat_conv_name):
            cur_features = example['backbone_features'][src_name]
            if self.lat_tensor_type[k] == 'dense' and isinstance(cur_features, SparseConvTensor):
                cur_features = cur_features.dense()
            cur_features = self.lat_conv[k](cur_features)

            if isinstance(cur_features, SparseConvTensor):
                cur_features = cur_features.dense()
            multi_scale_features.append(cur_features)

        multi_scale_features = torch.cat(multi_scale_features, dim=1)
        multi_scale_features = self.fusion_conv(multi_scale_features)

        # step 2: bi-linear interpolation
        # batch_dict['rois'] = batch_dict['roi_targets_dict']['gt_of_rois_src'][..., :7].contiguous()
        # batch_dict['rois'] = batch_dict['gt_boxes'][..., :7]
        pooling_points = self.get_pooling_points(example['rois'])
        
        pts_interp_features = self.interpolate_from_bev_features(
            pooling_points, multi_scale_features, bev_stride=self.out_stride
        )

        example['roi_features'] = pts_interp_features.view(batch_size, num_rois, -1)
        example['point_features'] = pts_interp_features  # (B, N, 7x7, C)
        example['point_coords'] = pooling_points  # [(N1, 2), (N2, 2), ...]
        
        return example


@SECOND_STAGE.register_module
class BEVStrideFeature(nn.Module):
    def __init__(self, feature_sources, pillar_size, pc_range, out_stride=4, grid_size=7, 
                 in_channels=128, share_channels=64, backbone_channels=None, backbone_strides=None):
        super().__init__()
        self.pillar_size = pillar_size
        self.point_cloud_range = pc_range
        self.grid_size = grid_size

        self.lat_conv = nn.ModuleList()
        self.lat_conv_name = []
        self.lat_tensor_type = []

        opt_strides = [1, 2, 4]
        opt_out_stage_name = ['conv1', 'conv2', 'conv3']
        opt_out_channels = [32, 64, 128]
        
        assert out_stride in opt_strides
        out_name = opt_out_stage_name[opt_strides.index(out_stride)]
        out_channels = opt_out_channels[opt_strides.index(out_stride)]
        assert out_channels <= backbone_channels[out_name]
        # out_bev_channels = max(self.model_cfg.SHARE_FEATURES, out_channels)

        c_in = 0
        stride = int(backbone_strides['conv3'] / out_stride)
        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.top_down_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, stride, stride=stride, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU()
        )
        c_in += out_channels

        for src_name in feature_sources:
            if src_name not in ['conv1', 'conv2', 'conv3', 'conv4']: 
                continue

            in_channels = backbone_channels[src_name]
            stride = backbone_strides[src_name] / out_stride

            if stride >= 1:
                stride = int(stride)
                norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, stride, stride=stride, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU()
                )
                self.lat_tensor_type.append('dense')
            else:
                stride = int(np.round(1 / stride))
                norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)
                deblock = SparseSequential(
                    SparseConv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=True),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU()
                )
                self.lat_tensor_type.append('sparse')

            c_in += out_channels
            self.lat_conv.append(deblock)
            self.lat_conv_name.append(src_name)

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c_in, share_channels, 3, stride=1, padding=1, bias=True),
            build_norm_layer(norm_cfg, share_channels)[1],
            nn.ReLU()
        )
        self.out_stride = out_stride

    def get_pooling_points(self, rois):
        batch_size, num_rois, _ = rois.shape
        rois = rois.view(-1, rois.shape[-1])

        center2d = rois[:, :2]
        # height = rois[:, 2:3]
        dim2d = rois[:, 3:5]
        rotation_y = rois[:, -1]
        
        grid_points = center_to_grid_box2d(
            center2d, dim2d, rotation_y, (self.grid_size, self.grid_size))

        return grid_points.view(batch_size, num_rois, -1, grid_points.shape[-1])
    
    def interpolate_from_bev_features(self, pooling_points, bev_features, bev_stride):
        """
        Args:
            pooling_points: (batch_size, num_rois, 7x7, 2)
            bev_features: (batch, channels, height, weight)

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        batch_size, num_rois, grid_size = pooling_points.shape[:3]
        x_idxs = (pooling_points[..., 0] - self.point_cloud_range[0]) / (bev_stride * self.pillar_size)
        y_idxs = (pooling_points[..., 1] - self.point_cloud_range[1]) / (bev_stride * self.pillar_size)

        pts_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k].view(-1)
            cur_y_idxs = y_idxs[k].view(-1)
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            pts_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            pts_bev_features_list.append(pts_bev_features.view(num_rois, grid_size, -1))

        pts_bev_features = torch.stack(pts_bev_features_list, dim=0)  # (batch_size, num_rois, 7x7, C)
        return pts_bev_features

    def forward(self, example):
        batch_size, num_rois = example['rois'].shape[:2]

        # step 1: aggregate multi-scale bev features with stride <= 4
        top_down_features = self.top_down_conv(example['bev_feature'])
        multi_scale_features = [top_down_features]

        for k, src_name in enumerate(self.lat_conv_name):
            cur_features = example['backbone_features'][src_name]
            if self.lat_tensor_type[k] == 'dense' and isinstance(cur_features, SparseConvTensor):
                cur_features = cur_features.dense()
            cur_features = self.lat_conv[k](cur_features)

            if isinstance(cur_features, SparseConvTensor):
                cur_features = cur_features.dense()
            multi_scale_features.append(cur_features)

        multi_scale_features = torch.cat(multi_scale_features, dim=1)
        multi_scale_features = self.fusion_conv(multi_scale_features)

        # step 2: bi-linear interpolation
        # example['rois'] = example['roi_targets_dict']['gt_of_rois_src'][..., :7].contiguous()
        # example['rois'] = example['gt_box'][0][..., :7]
        pooling_points = self.get_pooling_points(example['rois'])
        
        # from tools.visual import draw_scenes
        # roi = example['rois'][0, 0]
        # pts2d = pooling_points[0, 0]
        # pts3d = torch.cat((pts2d, pts2d.new_full((pts2d.shape[0], 1), roi[2].item())), dim=-1)
        # draw_scenes(pts3d.cpu().numpy(), gt_boxes=roi.view(1, -1).cpu().numpy())
        
        pts_interp_features = self.interpolate_from_bev_features(
            pooling_points, multi_scale_features, bev_stride=self.out_stride
        )
        example['roi_features'] = pts_interp_features.view(batch_size, num_rois, -1)
        example['point_features'] = pts_interp_features  # (B, N, 7x7, C)
        example['point_coords'] = pooling_points  # [(N1, 2), (N2, 2), ...]
        
        return example
