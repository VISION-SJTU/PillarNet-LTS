from torch import nn

from spconv.pytorch import SubMConv2d, SubMConv3d, SparseConv2d, SparseModule, \
    SparseSequential, SparseInverseConv2d, SparseReLU

from ..utils import build_norm_layer


def replace_feature(out, new_features):
    return out.replace_feature(new_features)


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

def conv2D3x3(in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True):
    """3x3 convolution with padding to keep the same input and output"""
    assert stride >= 1
    padding = dilation
    if stride == 1:
        return SubMConv2d(
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
        return SparseConv2d(
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
    return SubMConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=bias,
        )
    

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   dilation=1, conv_type='subm', norm_cfg=None):
    if conv_type == 'subm':
        conv = SubMConv2d(in_channels, out_channels, kernel_size, dilation=dilation,
                          padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = SparseConv2d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = SparseInverseConv2d(in_channels, out_channels, kernel_size, 
                                   indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    return SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        SparseReLU()
    )


def post_act_block_dense(in_channels, kernel_size, stride=1, padding=0, dilation=1, norm_cfg=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                  padding=padding, dilation=dilation, bias=False),
        build_norm_layer(norm_cfg, in_channels)[1],
        SparseReLU()
    )

    return m


class SparseBasicBlock(SparseModule):
    expansion = 1

    def __init__(self, planes, norm_cfg=None, indice_key=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        bias = norm_cfg is not None
        
        self.conv1 = SparseSequential(
            conv3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
            SparseReLU()
        )
        self.conv2 = SparseSequential(
            conv3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DBasicBlockV(SparseModule):
    expansion = 1

    def __init__(self, planes, norm_cfg=None, indice_key=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        bias = norm_cfg is not None
        self.conv0 = SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.conv1 = SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
            SparseReLU()
        )
        self.conv2 = SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class Sparse2DBasicBlock(SparseModule):
    expansion = 1

    def __init__(self, planes, norm_cfg=None, indice_key=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        bias = norm_cfg is not None
        self.conv1 = SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
            SparseReLU()
        )
        self.conv2 = SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out
    
    
class Dense2DBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out

