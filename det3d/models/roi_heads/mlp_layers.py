import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from ..utils import build_norm_layer


class Projector(nn.Module):
    def __init__(self, in_channels, num_patches, out_channels):
        super(Projector, self).__init__()

        self.out_proj = nn.Sequential(
            build_norm_layer(dict(type='LN'), in_channels * num_patches)[1],
            nn.Linear(in_channels * num_patches, out_channels),
        )

    def forward(self, x):
        # x: (B, N, C)
        batch = x.shape[0]
        x = x.view(batch, -1)
        x = self.out_proj(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, num_patches, expansion_factor=4, expansion_factor_token=0.5):
        super(MLPMixer, self).__init__()

        # token mixing
        inner_dim = int(num_patches * expansion_factor)
        self.token_mixer = nn.Sequential(
            build_norm_layer(dict(type='LN'), in_channels)[1],
            nn.Conv1d(num_patches, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(inner_dim, num_patches, kernel_size=1),
        )

        # channel_mixing
        inner_dim = int(in_channels * expansion_factor_token)
        self.channel_mixer = nn.Sequential(
            build_norm_layer(dict(type='LN'), in_channels)[1],
            nn.Linear(in_channels, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, in_channels),
        )

        # self.out_proj = nn.Sequential(
        #     build_norm_layer(dict(type='LN'), in_channels * num_patches)[1],
        #     nn.Linear(in_channels * num_patches, out_channels),
        # )

    def forward(self, x):
        # x: (B, N, C)
        x = self.token_mixer(x) + x

        # x: (B, N, C)
        x = self.channel_mixer(x) + x
        # batch = x.shape[0]
        # x = x.view(batch, -1)
        # x = self.out_proj(x)
        return x


############ ResMLP blocks ###############
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return self.alpha * x + self.beta


class ResMLPLayer(nn.Module):
    def __init__(self, in_channels, num_patches, expansion_factor=2, layer_scale_init=1e-4):
        super().__init__()

        # token mixer
        self.token_aff = Affine(in_channels)
        self.token_scale = nn.Parameter(layer_scale_init * torch.ones((in_channels), requires_grad=True))

        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patches, num_patches),
            Rearrange('b d n -> b n d'),
        )

        # channel mixer
        self.channel_aff = Affine(in_channels)
        self.channel_scale = nn.Parameter(layer_scale_init * torch.ones((in_channels), requires_grad=True))

        self.channel_mixer = nn.Sequential(
            nn.Linear(in_channels, in_channels * expansion_factor),
            nn.GELU(),
            nn.Linear(in_channels * expansion_factor, in_channels)
        )

        # post
        self.post_aff = Affine(in_channels)
        # self.out_proj = nn.Linear(in_channels * num_patches, out_channels)

    def forward(self, x):
        x = self.token_aff(x)
        x = x + self.token_scale * self.token_mixer(x)

        x = self.channel_aff(x)
        x = x + self.channel_scale * self.channel_mixer(x)

        x = self.post_aff(x)
        # batch = x.shape[0]
        # x = x.view(batch, -1)
        # x = self.out_proj(x)

        return x

