# SmoothNet inlined from mmotion.models.post_processing.smooth.smoothnet.
# Self-contained: no mmotion or pytorch3d dependency.

from typing import Optional

import numpy as np
import torch
from mmengine.runner import load_checkpoint
from torch import Tensor, nn

from prism.registry import MODELS
from prism.utils.geometry.rotation_convert import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _aa_to_rotmat(x):
    """axis-angle (..., 3) -> rotation matrix (..., 3, 3)"""
    return axis_angle_to_matrix(x)


def _rotmat_to_rot6d(x):
    """rotation matrix (..., 3, 3) -> 6d (..., 6)"""
    return matrix_to_rotation_6d(x)


def _rot6d_to_rotmat(x):
    """6d (..., 6) -> rotation matrix (..., 3, 3)"""
    return rotation_6d_to_matrix(x)


def _rotmat_to_aa(x):
    """rotation matrix (..., 3, 3) -> axis-angle (..., 3)"""
    return matrix_to_axis_angle(x)


class SmoothNetResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        return x + identity


@MODELS.register_module()
class SmoothNet(nn.Module):
    def __init__(self, window_size, output_size, hidden_size=512,
                 res_hidden_size=512, num_blocks=5, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        assert output_size <= window_size

        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(SmoothNetResBlock(hidden_size, res_hidden_size, dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        N, C, T = x.shape
        assert T >= self.window_size
        x = x.unfold(2, self.window_size, 1)
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)

        num_windows = T - self.window_size + 1
        out = x.new_zeros(N, C, T)
        count = x.new_zeros(T)
        for t in range(num_windows):
            out[..., t:t + self.output_size] += x[:, :, t]
            count[t:t + self.output_size] += 1.0
        return out.div(count)


@MODELS.register_module(name=['SmoothNetFilter', 'smoothnet'])
class SmoothNetFilter:
    def __init__(self, window_size, output_size, checkpoint=None,
                 hidden_size=512, res_hidden_size=512, num_blocks=5,
                 device='cpu'):
        super().__init__()
        self.window_size = window_size
        self.device = device
        self.smoothnet = SmoothNet(window_size, output_size, hidden_size,
                                   res_hidden_size, num_blocks)
        self.smoothnet.to(device)
        if checkpoint:
            load_checkpoint(self.smoothnet, checkpoint, map_location=self.device)
        self.smoothnet.eval()
        for p in self.smoothnet.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        x_type = 'tensor' if isinstance(x, torch.Tensor) else 'array'
        assert x.ndim == 3
        T, K, C = x.shape
        assert C in (3, 6, 9)

        if T < self.window_size:
            return x

        with torch.no_grad():
            if x_type == 'array':
                dtype = x.dtype
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if C == 9:
                input_type = 'matrix'
                x = _rotmat_to_rot6d(x.reshape(-1, 3, 3)).reshape(T, K, -1)
            elif C == 3:
                input_type = 'axis_angles'
                x = _rotmat_to_rot6d(_aa_to_rotmat(x.reshape(-1, 3))).reshape(T, K, -1)
            else:
                input_type = 'rotation_6d'
            x = x.view(1, T, -1).permute(0, 2, 1)
            smoothed = self.smoothnet(x)

        smoothed = smoothed.permute(0, 2, 1).view(T, K, -1)

        if input_type == 'matrix':
            smoothed = _rot6d_to_rotmat(smoothed.reshape(-1, 6)).reshape(T, K, C)
        elif input_type == 'axis_angles':
            smoothed = _rotmat_to_aa(_rot6d_to_rotmat(smoothed.reshape(-1, 6))).reshape(T, K, C)

        if x_type == 'array':
            smoothed = smoothed.cpu().numpy().astype(dtype)

        return smoothed
