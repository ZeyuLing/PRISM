# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import (
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.transformers.transformer_wan import (
    WanTransformer3DModel,
    WanAttention,
    WanAttnProcessor,
    FeedForward,
)
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.configuration_utils import register_to_config
from diffusers.models.normalization import FP32LayerNorm

from mmotion.registry import HF_MODELS

from .motion_rope import MotionWanRotaryPosEmbed

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def create_custom_forward(module):
    def custom_forward(*args, **kwargs):
        return module(*args, **kwargs)

    return custom_forward

class WanTransformerNotextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        return hidden_states


@HF_MODELS.register_module(force=True)
class PrismTransformerNotextMotionModel(WanTransformer3DModel):
    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 1),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 32,
        out_channels: int = 16,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
    ) -> None:
        super(WanTransformer3DModel, self).__init__()
        assert patch_size[-1] == 1, "we don't support joints patchify"
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = MotionWanRotaryPosEmbed(
            attention_head_dim, patch_size, rope_max_seq_len
        )
        self.patch_embedding = nn.Conv2d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerNotextBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, num_joints = hidden_states.shape
        p_t, p_j = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_num_joints = num_joints // p_j

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        # b n c
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj = self.condition_embedder(
            timestep,
            timestep_seq_len=ts_seq_len,
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (
                self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (
                self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_num_joints,
            p_t,
            p_j,
            -1,
        )
        # b t j pt pj c -> b c t pt j pj
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        return output


if __name__ == "__main__":
    from mmengine.device import get_device

    device = get_device()
    dtype = torch.bfloat16
    model = PrismTransformerNotextMotionModel(
        patch_size=(1, 1),
        attention_head_dim=128,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=32,
        num_attention_heads=12,
        num_layers=30,
        out_channels=16,
        rope_max_seq_len=1024,
    ).to(device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        hidden_states = torch.randn(2, 32, 17, 24).to(device=device, dtype=dtype)
        timestep = torch.tensor([0]).to(device=device, dtype=dtype)
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
        )
        print(output.shape)
