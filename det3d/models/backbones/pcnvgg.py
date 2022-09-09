import torch
from torch import nn

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import spconv, post_act_block, post_act_block_dense


@BACKBONES.register_module
class SpMiddlePillarEncoderVgg(nn.Module):
    def __init__(
        self, in_planes=32, name="SpMiddlePillarEncoderVgg", **kwargs):
        super(SpMiddlePillarEncoderVgg, self).__init__()
        self.name = name

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(32, 32, 3, padding=1, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 32)[1],
            block(32, 32, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 64, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 128, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(128, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(256, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )
