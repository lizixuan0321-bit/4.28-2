import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        import torch
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor
import numpy as np

"""
Original TokenLearner Module modified from:
https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py
"""

def depthwise_separable_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    """3x3 standard conv replaced by depthwise spatial conv + 1x1 pointwise conv."""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
    )

class LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim,  1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class MSFN(nn.Module):

    def __init__(self, in_dim, num_token, use_sum_pooling=False):
        super(MSFN, self).__init__()
        self.in_dim = in_dim
        self.num_token = num_token
        self.use_sum_pooling = use_sum_pooling

        self.selected_func = nn.Sequential(
            LayerNorm(self.in_dim),
            depthwise_separable_conv2d(self.in_dim, self.num_token),
            nn.GELU(),
            depthwise_separable_conv2d(self.num_token, self.num_token),
            nn.GELU(),
            depthwise_separable_conv2d(self.num_token, self.num_token),
            nn.GELU(),
            depthwise_separable_conv2d(self.num_token, self.num_token),
            Rearrange('b n h w -> b n (h w)'),
            nn.Sigmoid()
        )

        self.feat_func = Rearrange('b c h w -> b (h w) c')

    def forward(self, inputs):
        # select func: layernorm -> 3 * (conv + gelu) -> conv -> reshape -> sigmoid
        # output shape: (batch_size, num_token, H * W, 1)

        selected = self.selected_func(inputs)
        selected = selected[:, :, :, None]

        # feat func: reshape
        # output shape: (batch_size, 1, H * W, channels)
        feat = self.feat_func(inputs)
        feat = feat[:, None, :, :]

        if self.use_sum_pooling:
            inputs = torch.sum(feat * selected, dim=2)
        else:
            inputs = torch.mean(feat * selected, dim=2)
        inputs = inputs.permute(0, 2, 1)

        return inputs


