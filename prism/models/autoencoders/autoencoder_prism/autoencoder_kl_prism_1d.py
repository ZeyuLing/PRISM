# -*- coding: utf-8 -*-
"""
1D KL-VAE for SMPL motion sequences.

Unlike the 2D variant (AutoencoderKLMotionWan2DTK) which treats each joint as a
spatial token with conv2d over (T, K), this 1D variant flattens all joints into the
channel dimension and uses conv1d over the temporal dimension T only.

Input:  [B, T, K*C] where K=num_joints, C=6 (rotation_6d)
Latent: [B, z_dim, T_down]  (no joint dimension in latent space)
Output: [B, T, K*C]
"""
from __future__ import annotations
from typing import Optional, Tuple, Union, List

import logging
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.loaders import FromOriginalModelMixin

from prism.registry import MODELS, HF_MODELS
from .wan_causalconv import WanCausalConv1d
from .wan_encdec import WanEncoder1D, WanDecoder1D
from ..gaussian_distribution import DiagonalGaussianDistributionNd

logger = logging.getLogger(__name__)


@HF_MODELS.register_module(force=True)
@MODELS.register_module(force=True)
class AutoencoderKLPrism1D(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    1D Causal KL-VAE for SMPL/SMPL-X motion sequences.

    Unlike AutoencoderKLMotionWan2DTK which processes [B, T, K, C] with 2D conv
    over (T, K), this model processes [B, T, D] with 1D conv over T only,
    where D = K * C (joints flattened into channels).

    This serves as an ablation baseline to compare:
      - 2D VAE (joint-aware): each joint is a separate spatial token
      - 1D VAE (joint-agnostic): all joints packed into channel dim
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: Optional[int] = None,
        z_dim: int = 16,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: List[int] = [],
        temporal_downsample: Tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
        is_residual: bool = False,
        in_channels: int = 138,   # K*C = 23*6 = 138 (all joints flattened)
        out_channels: int = 138,
        scale_factor_temporal: Optional[int] = 4,
        latents_mean=[],
        latents_std=[],
        use_static: bool = False,
        use_rollout_trans: bool = True,
        mid_attention: str = "none",
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        # ---------------- Encoder / Decoder (1D causal) ----------------
        self.encoder = WanEncoder1D(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,  # mean + logvar
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            temporal_downsample=temporal_downsample,
            dropout=dropout,
            is_residual=is_residual,
            mid_attention=mid_attention,
        )
        self.quant_conv = WanCausalConv1d(z_dim * 2, z_dim * 2, kernel_size=1, padding=0)
        self.post_quant_conv = WanCausalConv1d(z_dim, z_dim, kernel_size=1, padding=0)

        self.decoder = WanDecoder1D(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            temporal_upsample=list(self.temporal_upsample),
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
            mid_attention=mid_attention,
        )

        # Cache management
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, WanCausalConv1d) for m in self.decoder.modules()),
            "encoder": sum(isinstance(m, WanCausalConv1d) for m in self.encoder.modules()),
        }

    def clear_cache(self):
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        """x: [B, D, T]"""
        _, _, T = x.shape
        self.clear_cache()

        # WAN-style chunking: first 1 frame, then stride-4
        iters = 1 + max(0, (T - 1) // 4)
        out = None
        for i in range(iters):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1): 1 + 4 * i],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    @apply_forward_hook
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] → latent [B, 2*z_dim, T_down] (mean & logvar packed).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input as [B, T, D], got {tuple(x.shape)}")
        x_bdt = x.permute(0, 2, 1).contiguous()  # [B, D, T]
        return self._encode(x_bdt)

    def _decode(self, z: torch.Tensor):
        """z: [B, z_dim, T_down]"""
        _, _, num_frames = z.shape
        self.clear_cache()

        x = self.post_quant_conv(z)
        for i in range(num_frames):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)

        self.clear_cache()
        return out

    @apply_forward_hook
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, z_dim, T_down] → sample [B, T, D]
        """
        if z.dim() != 3:
            raise ValueError(f"Expected latent as [B, z_dim, T_down], got {tuple(z.shape)}")
        decoded = self._decode(z)  # [B, D, T]
        return decoded.permute(0, 2, 1).contiguous()  # [B, T, D]

    def forward(
        self,
        sample: torch.Tensor,  # [B, T, D]
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        if sample.dim() != 3:
            raise ValueError(f"Expected sample as [B, T, D], got {tuple(sample.shape)}")
        z = self.encode(sample)
        dist = DiagonalGaussianDistributionNd(z)
        z = dist.sample(generator=generator) if sample_posterior else dist.mode()
        dec = self.decode(z)
        return dec
