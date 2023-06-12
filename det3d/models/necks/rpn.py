import numpy as np
import torch

from torch import nn
from spconv.pytorch import SparseConvTensor

from det3d.models.utils import Sequential
from det3d.torchie.cnn import xavier_init

from ..registry import NECKS
from ..utils import build_norm_layer


@NECKS.register_module
class RPN(nn.Module):
    def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides,
                 us_num_filters, in_channels, norm_cfg=None, logger=None, **kwargs):
        super().__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        blocks = []
        deblocks = []
        in_filters = [in_channels, *self._num_filters[:-1]]

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU())

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self._norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return tuple(x)


@NECKS.register_module
class RPNV1(nn.Module):
    def __init__(self, layer_nums, num_filters, in_channels,
                 norm_cfg=None, logger=None, **kwargs):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.norm_cfg = norm_cfg

        self.block_5, num_out_filters = self._build_layer(
            in_channels[0], in_channels[0], layer_nums[0], stride=1)
        
        self.deblock_5 = Sequential(
            nn.ConvTranspose2d(num_out_filters, in_channels[1], 2, stride=2, bias=False),
            build_norm_layer(self.norm_cfg, in_channels[1])[1],
            nn.ReLU()
        )

        # self.deblock_4 = Sequential(
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(in_channels[1], us_num_filters[1], 3, stride=1, bias=False),
        #     build_norm_layer(self.norm_cfg, us_num_filters[1])[1],
        #     nn.ReLU()
        # )
        
        self.block_4, _ = self._build_layer(
            in_channels[1] * 2, num_filters, layer_nums[1], stride=1)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        return 1

    def _build_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self.norm_cfg, planes)[1],
            nn.ReLU())

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self.norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features, **kwargs):
        x_conv4 = pillar_features['conv4']
        x_conv5 = pillar_features['conv5']
        if isinstance(x_conv4, SparseConvTensor):
            x_conv4 = x_conv4.dense()
        if isinstance(x_conv5, SparseConvTensor):
            x_conv5 = x_conv5.dense()

        ups = [x_conv4]
        x = self.block_5(x_conv5)
        ups.append(self.deblock_5(x))
        x = torch.cat(ups, dim=1)
        x = self.block_4(x)
        
        return tuple([x])


@NECKS.register_module
class RPNV2(nn.Module):
    def __init__(self, layer_nums, in_channels, num_filters, norm_cfg=None, logger=None):
        super().__init__()
    
        if norm_cfg is None:
            norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.norm_cfg = norm_cfg

        self.block_4 = self._build_layer(
            in_channels[0], in_channels[0], layer_nums[0], stride=1)

        self.deblock_4 = Sequential(
            nn.ConvTranspose2d(in_channels[0], in_channels[1], 2, stride=2, bias=False),
            build_norm_layer(self.norm_cfg, in_channels[1])[1],
            nn.ReLU(),
        )
        
        self.block_3 = self._build_layer(
            in_channels[1] * 2, num_filters, layer_nums[1], stride=1)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        return 1

    def _build_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self.norm_cfg, planes)[1],
            nn.ReLU())

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self.norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block
 
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features, **kwargs):
        x_conv3 = pillar_features['conv3']
        x_conv4 = pillar_features['conv4']
        if isinstance(x_conv3, SparseConvTensor):
            x_conv3 = x_conv3.dense()
        if isinstance(x_conv4, SparseConvTensor):
            x_conv4 = x_conv4.dense()

        ups = [x_conv3]
        x = self.block_4(x_conv4)
        ups.append(self.deblock_4(x))
        x = torch.cat(ups, dim=1)
        x = self.block_3(x)
        
        return tuple([x])
    

@NECKS.register_module
class RPNG(nn.Module):
    def __init__(self, layer_nums, in_channels, num_filters, norm_cfg=None, logger=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.block_5 = self._build_layer(
            in_channels[0], in_channels[0], layer_nums[0], stride=1)
        
        self.top_down_54 = Sequential(
            nn.ConvTranspose2d(in_channels[0], in_channels[1], 2, stride=2, bias=False),
            build_norm_layer(self.norm_cfg, in_channels[1])[1],
            nn.ReLU()
        )
        
        self.block_4 = self._build_layer(in_channels[1] * 2, num_filters[0], layer_nums[0])
        
        self.top_down_43 = Sequential(
            nn.ConvTranspose2d(num_filters[0], in_channels[2], 2, stride=2, bias=False),
            build_norm_layer(self.norm_cfg, in_channels[2])[1],
            nn.ReLU()
        )
        
        self.block_3 = self._build_layer(in_channels[2] * 2, num_filters[1], layer_nums[1])

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        return 1

    def _build_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self.norm_cfg, planes)[1],
            nn.ReLU())

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self.norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features, **kwargs):
        if isinstance(pillar_features, dict):
            x_conv3 = pillar_features['conv3']
            x_conv4 = pillar_features['conv4']
            x_conv5 = pillar_features['conv5']
            
            if isinstance(x_conv3, SparseConvTensor):
                x_conv3 = x_conv3.dense()
            if isinstance(x_conv4, SparseConvTensor):
                x_conv4 = x_conv4.dense()
            if isinstance(x_conv5, SparseConvTensor):
                x_conv5 = x_conv5.dense()
                
        # head stride 8
        ups = [x_conv4]
        x5 = self.block_5(x_conv5)
        ups.append(self.top_down_54(x5))
        x4 = torch.cat(ups, dim=1)
        x4 = self.block_4(x4)
        # head stride 4
        ups = [x_conv3]
        ups.append(self.top_down_43(x4))
        x3 = torch.cat(ups, dim=1)
        x3 = self.block_3(x3)

        return tuple([x4, x3])
    
    
@NECKS.register_module
class RPNGV2(nn.Module):
    def __init__(self, layer_nums, in_channels, num_filters, norm_cfg=None, logger=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.block_5 = self._build_layer(
            in_channels[0], in_channels[0], layer_nums[0], stride=1)
        
        self.top_down_54 = Sequential(
            nn.ConvTranspose2d(in_channels[0], num_filters[0] // 2, 2, stride=2, bias=False),
            build_norm_layer(self.norm_cfg, num_filters[0] // 2)[1],
            nn.ReLU()
        )
        
        self.reduce_4 = Sequential(
            nn.Conv2d(in_channels[1], num_filters[0] // 2, 3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, num_filters[0] // 2)[1],
            nn.ReLU()
        )
                
        self.block_4 = self._build_layer(num_filters[0], num_filters[0], layer_nums[0])

        self.top_down_43 = Sequential(
            nn.ConvTranspose2d(num_filters[0], num_filters[1] // 2, 2, stride=2, bias=False),
            build_norm_layer(self.norm_cfg, num_filters[1] // 2)[1],
            nn.ReLU()
        )
        
        self.reduce_3 = Sequential(
            nn.Conv2d(in_channels[2], num_filters[1] // 2, 3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, num_filters[1] // 2)[1],
            nn.ReLU()
        )
        
        self.block_3 = self._build_layer(num_filters[1], num_filters[1], layer_nums[1])

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        return 1

    def _build_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self.norm_cfg, planes)[1],
            nn.ReLU())

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self.norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features, **kwargs):
        if isinstance(pillar_features, dict):
            x_conv3 = pillar_features['conv3']
            x_conv4 = pillar_features['conv4']
            x_conv5 = pillar_features['conv5']
            
            if isinstance(x_conv3, SparseConvTensor):
                x_conv3 = x_conv3.dense()
            if isinstance(x_conv4, SparseConvTensor):
                x_conv4 = x_conv4.dense()
            if isinstance(x_conv5, SparseConvTensor):
                x_conv5 = x_conv5.dense()
        
        # head stride 8
        ups = [self.reduce_4(x_conv4)]
        x5 = self.block_5(x_conv5)
        ups.append(self.top_down_54(x5))
        x4 = torch.cat(ups, dim=1)
        x4 = self.block_4(x4)
        # head stride 4
        ups = [self.reduce_3(x_conv3)]
        ups.append(self.top_down_43(x4))
        x3 = torch.cat(ups, dim=1)
        x3 = self.block_3(x3)

        return tuple([x4, x3])