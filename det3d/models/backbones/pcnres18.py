import torch
from torch import nn

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import spconv, SparseConv2d, Sparse2DBasicBlock, Sparse2DBasicBlockV, post_act_block_dense


@BACKBONES.register_module
class SpMiddlePillarEncoder(nn.Module):
    def __init__(self,
                 in_planes=32, name="SpMiddlePillarEncoder", **kwargs):
        super(SpMiddlePillarEncoder, self).__init__()
        self.name = name

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                64, 128, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                128, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
        }
        self.backbone_strides = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
        }

    def forward(self, sp_tensor):
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()

        return x_conv4


@BACKBONES.register_module
class SpMiddlePillarEncoder18(nn.Module):
    def __init__(self,
                 in_planes=32, name="SpMiddlePillarEncoder18", **kwargs):
        super(SpMiddlePillarEncoder18, self).__init__()
        self.name = name

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                64, 128, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                128, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
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

    def forward(self, sp_tensor):
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


@BACKBONES.register_module
class SpMiddlePillarEncoder2x18(nn.Module):
    def __init__(self,
                 in_planes=32, name="SpMiddlePillarEncoder2x18", **kwargs):
        super(SpMiddlePillarEncoder2x18, self).__init__()
        self.name = name

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv2 = spconv.SparseSequential(
            Sparse2DBasicBlockV(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                64, 128, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                128, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def forward(self, sp_tensor):
        x_conv2 = self.conv2(sp_tensor)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder4x18(nn.Module):
    def __init__(self,
                 in_planes=32, name="SpMiddlePillarEncoder4x18", **kwargs):
        super(SpMiddlePillarEncoder4x18, self).__init__()
        self.name = name

        dense_block = post_act_block_dense
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv3 = spconv.SparseSequential(
            Sparse2DBasicBlockV(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                128, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def forward(self, sp_tensor):
        x_conv3 = self.conv3(sp_tensor)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder8x18(nn.Module):
    def __init__(self,
                 in_planes=32, name="SpMiddlePillarEncoder8x18", **kwargs):
        super(SpMiddlePillarEncoder8x18, self).__init__()
        self.name = name

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv4 = spconv.SparseSequential(
            Sparse2DBasicBlockV(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'x_conv4': 256,
            'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv4': 8,
            'x_conv5': 16,
        }

    def forward(self, sp_tensor):
        x_conv4 = self.conv4(sp_tensor)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )