import torch
import numpy as np
from torch import nn

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SubMConv3d, SparseConv2d, SparseMaxPool2d, SparseInverseConv2d
except:
    import spconv
    from spconv import ops
    from spconv import SubMConv3d, SparseConv2d, SparseMaxPool2d, SparseInverseConv2d

from timm.models.layers import DropPath
from ..utils import build_norm_layer


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


def conv3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   dilation=1, conv_type='subm', norm_cfg=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, dilation=dilation,
                                 padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_cfg=None):
    m = spconv.SparseSequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

def conv2D3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding to keep the same input and output"""
    assert stride >= 1
    padding = dilation
    if stride == 1:
        return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
    else:
        return spconv.SparseConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )

def conv2D1x1(in_planes, out_planes, bias=False):
    """1x1 convolution"""
    return spconv.SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=bias,
            # indice_key=indice_key,
        )


class Sparse2DBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlock, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DBasicBlockV(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DBasicBlockV, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv0 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv1 = spconv.SparseSequential(
            conv2D3x3(planes, planes, stride, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = replace_feature(x, self.relu(x.features))
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DAttBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=7,
        stride=1,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DAttBlock, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(inplanes, planes, 3, stride, padding=1, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU()
        )

        self.conv2 = spconv.SubMConv2d(2, 1, kernel_size, stride, padding=kernel_size // 2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        feat = x.features

        max_pool, _ = torch.max(feat, dim=1, keepdim=True)
        avg_pool = torch.mean(feat, dim=1, keepdim=True)
        feat = torch.cat((max_pool, avg_pool), dim=1)
        x = replace_feature(x, feat)
        x = self.conv2(x)
        x = replace_feature(x, torch.sigmoid(x.features))

        return x


class Sparse2DBottleneckV(spconv.SparseModule):
    expansion = 4
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 norm_cfg=None,
                 indice_key=None,
                 ):
        super(Sparse2DBottleneckV, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv0 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes * self.expansion, 1, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes*self.expansion)[1]
        )
        self.conv1 = spconv.SparseSequential(
            conv2D1x1(planes * self.expansion, planes, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv3 = spconv.SparseSequential(
            conv2D1x1(planes, planes * self.expansion, bias=bias),
            build_norm_layer(norm_cfg, planes * self.expansion)[1]
        )
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        x = self.conv0(x)
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv3(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DBottleneck(spconv.SparseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 stride=1,
                 norm_cfg=None,
                 indice_key=None,
                 ):
        super(Sparse2DBottleneck, self).__init__()
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None
        planes = inplanes // self.expansion

        self.conv1 = spconv.SparseSequential(
            conv2D1x1(inplanes, planes, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv3 = spconv.SparseSequential(
            conv2D1x1(planes, inplanes, bias=bias),
            build_norm_layer(norm_cfg, inplanes)[1]
        )
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv3(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class Dense2DBottleneck(spconv.SparseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 stride=1,
                 norm_cfg=None,
                 ):
        super(Dense2DBottleneck, self).__init__()
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        bias = norm_cfg is not None
        planes = inplanes // self.expansion

        self.conv1 = spconv.SparseSequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride, padding=0, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.conv2 = spconv.SparseSequential(
            nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.conv3 = spconv.SparseSequential(
            nn.Conv2d(planes, inplanes, 1, stride=stride, padding=0, bias=bias),
            build_norm_layer(norm_cfg, planes)[1]
        )
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity
        out = self.relu(out)

        return out


class DropPathV(spconv.SparseModule):
    def __init__(self, drop_prob=0):
        super(DropPathV, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, indices, batch):
        if self.drop_prob == 0. or not self.training:
            return x
        with torch.no_grad():
            keep_prob = 1 - self.drop_prob
            random = keep_prob + torch.rand(batch, dtype=x.dtype, device=x.device)
            random.floor_()

            random_batch = torch.zeros_like(x)
            for i in range(batch):
                random_batch[indices[:, 0] == i] = random[i]

        output = x.div(keep_prob) * random_batch
        return output


class Sparse2DInverseBasicBlock(spconv.SparseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        drop_path=0.,
        layer_scale_init_value=None,
        indice_key=None
    ):
        super(Sparse2DInverseBasicBlock, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="LN", eps=1e-6)

        self.conv = spconv.SubMConv2d(inplanes, planes, 7, 1, padding=3, indice_key=indice_key)
        self.norm = build_norm_layer(norm_cfg, planes)[1]
        self.pwconv1 = nn.Linear(planes, self.expansion * planes)
        self.pwconv2 = nn.Linear(self.expansion * planes, planes)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((planes)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path = DropPathV(drop_path) if drop_path > 0 else nn.Identity()
        self.act = nn.GELU()
        self.stride = stride

    def forward(self, x):
        identity = x.features

        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.pwconv1(out.features))
        out = replace_feature(out, self.act(out.features))
        out = replace_feature(out, self.pwconv2(out.features))

        if self.gamma is not None:
            out = replace_feature(out, self.gamma * out.features)
        # out = replace_feature(out, identity + self.drop_path(out.features))
        out = replace_feature(out, identity + self.drop_path(out.features, out.indices, out.batch_size))
        return out


class Dense2DInverseBasicBlock(spconv.SparseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        drop_path=0.,
        layer_scale_init_value=None,
    ):
        super(Dense2DInverseBasicBlock, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="LN", eps=1e-6)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=7, padding=3, groups=inplanes)
        self.norm = build_norm_layer(norm_cfg, planes)[1]
        self.pwconv1 = nn.Linear(planes, self.expansion * planes)
        self.pwconv2 = nn.Linear(self.expansion * planes, planes)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((planes)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # self.drop_path = DropPathV(drop_path) if drop_path > 0 else nn.Identity()
        self.act = nn.GELU()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = self.gamma * out
        out = out.permute(0, 3, 1, 2)
        out = identity + self.drop_path(out)
        return out

class Dense2DBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
    ):
        super(Dense2DBasicBlock, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out

class Dense2DBasicBlockV(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
    ):
        super(Dense2DBasicBlockV, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        self.conv0 = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)

        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out


class UpsampleLayer(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 ):
        super(UpsampleLayer, self).__init__()
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        if stride > 1:
            self.deblock = nn.Sequential(
                nn.ConvTranspose2d(
                    inplanes,
                    planes,
                    stride,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(
                    norm_cfg,
                    planes,
                )[1],
                nn.ReLU(),
            )
        else:
            stride = np.round(1 / stride).astype(np.int64)
            self.deblock = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    stride,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(
                    norm_cfg,
                    planes,
                )[1],
                nn.ReLU(),
            )

    def forward(self, x):
        ups = self.deblock(x)
        return ups


class Sparse2DMerge(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        planes,
        norm_cfg=None,
        indice_key=None,
    ):
        super(Sparse2DMerge, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # bias = norm_cfg is not None
        # self.conv0 = spconv.SparseSequential(
        #     conv2D3x3(planes, planes, stride=1, indice_key=indice_key, bias=bias),
        #     build_norm_layer(norm_cfg, planes)[1],
        # )

    def forward(self, x, pool):
        identity = x.features

        # x.features = pool
        # out = self.conv0(x)

        x = replace_feature(x, x.features + pool)
        return x


