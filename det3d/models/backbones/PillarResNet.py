from torch import nn
from ..registry import BACKBONES
from ..utils import build_norm_layer
from spconv.pytorch import SparseSequential, SparseConv2d, SparseReLU
from .base import Sparse2DBasicBlock, Sparse2DBasicBlockV, post_act_block_dense


@BACKBONES.register_module
class PillarResNet18S(nn.Module):
    def __init__(self, in_channels=32, **kwargs):
        super().__init__()

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
        }

    def forward(self, sp_tensor):
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }
        return backbone_features
    
    
@BACKBONES.register_module
class PillarResNet18(nn.Module):
    def __init__(self, in_channels=32, **kwargs):
        super().__init__()

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
            'conv5': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
            'conv5': 16,
        }

    def forward(self, sp_tensor):
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        
        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
            'conv5': x_conv5,
        }
        return backbone_features
    

@BACKBONES.register_module
class PillarResNet34S(nn.Module):
    def __init__(self, in_channels=32, **kwargs):
        super().__init__()

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
        }

    def forward(self, sp_tensor):
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }
        return backbone_features
    
    
@BACKBONES.register_module
class PillarResNet34(nn.Module):
    def __init__(self, in_channels=32, **kwargs):
        super().__init__()

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
            'conv5': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
            'conv5': 16,
        }

    def forward(self, sp_tensor):
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
     
        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
            'conv5': x_conv5,
        }
        return backbone_features