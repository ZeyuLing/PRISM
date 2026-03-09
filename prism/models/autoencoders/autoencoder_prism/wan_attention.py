from typing import Optional

from torch import nn
import torch
from torch.nn import functional as F

from .wan_norm import WanRMSNorm


def _elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Positive feature map for linear attention (ELU(x)+1)."""
    return F.elu(x, alpha=1.0) + 1.0


class WanChannelLinearAttention(nn.Module):
    r"""
    Linear attention over the **channel dimension C** of [B, C, T].
    Cost O(T * C * d) with d=proj_dim << C, so suitable for large C (e.g. 512).
    No temporal dependency → supports arbitrary length encode/decode.
    Uses positive feature map phi = ELU(x)+1 so that sim(Q,K,V) = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ (phi(K)^T @ 1)).
    """

    def __init__(self, dim: int, proj_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.to_q = nn.Linear(1, proj_dim)
        self.to_k = nn.Linear(1, proj_dim)
        self.to_v = nn.Linear(1, 1)
        self.proj_out = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, C, T]. Each channel at each t attends to all channels via linear attention."""
        B, C, T = x.shape
        identity = x
        # Treat each channel as one token: [B, C, T] -> [B, T, C, 1]
        x_btc = x.permute(0, 2, 1).unsqueeze(-1)
        q = self.to_q(x_btc)
        k = self.to_k(x_btc)
        v = self.to_v(x_btc)
        phi_q = _elu_plus_one(q)
        phi_k = _elu_plus_one(k)
        # S = phi(K)^T @ V: [B,T,C,d]^T @ [B,T,C,1] -> [B,T,d,1]; einsum subscripts must be letters
        S = torch.einsum("btcd,btce->btde", phi_k, v)
        Z = torch.einsum("btcd,btce->btde", phi_k, torch.ones_like(v))
        Z = Z.clamp(min=1e-6)
        out = torch.einsum("btcd,btde->btce", phi_q, S) / torch.einsum("btcd,btde->btce", phi_q, Z)
        out = self.dropout(out)
        out = out.squeeze(-1).permute(0, 2, 1).contiguous()
        out = self.proj_out(out)
        return identity + out


class WanJointTokenAttention(nn.Module):
    r"""
    Joint-aware channel attention: assume C = K * (C//K) with K = num_joints.
    Each joint's (C//K) channels are aggregated to one token, then K tokens do full self-attention;
    output is broadcast back to C. Cost O(T * (C^2/K + K^2*d)) — efficient when K is moderate.
    """

    def __init__(self, dim: int, num_joints: int, token_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        assert dim % num_joints == 0
        self.dim = dim
        self.num_joints = num_joints
        self.channel_per_joint = cpj = dim // num_joints
        self.token_dim = token_dim
        self.to_tokens = nn.Linear(cpj, token_dim)
        self.norm = WanRMSNorm(token_dim, channel_dim=-1)
        self.to_qkv = nn.Linear(token_dim, token_dim * 3)
        self.proj = nn.Linear(token_dim, token_dim)
        self.from_tokens = nn.Linear(token_dim, cpj)
        self.dropout = nn.Dropout(dropout)
        self._scale = token_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]. Split C into K groups, aggregate to K tokens, attend, broadcast back."""
        B, C, T = x.shape
        identity = x
        K, cpj = self.num_joints, self.channel_per_joint
        x = x.view(B, K, cpj, T).permute(0, 3, 1, 2).contiguous()
        tokens = self.to_tokens(x)
        tokens = self.norm(tokens)
        qkv = self.to_qkv(tokens).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self._scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = self.proj(out)
        out = self.from_tokens(out)
        out = out.permute(0, 2, 3, 1).contiguous().view(B, C, T)
        return identity + out


class WanTemporalAttention(nn.Module):
    r"""
    Causal self-attention over the **time dimension T** of shape [B, C, T].
    Semantically: each frame attends to (previous) frames. No hard channel-split;
    the sequence is the temporal axis.

    Optional window_size: each position only attends to the last `window_size` positions,
    so cost is O(T * window_size) instead of O(T^2). window_size=None => full causal.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.norm = WanRMSNorm(dim, channel_dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T] -> [B, C, T]
        """
        B, C, T = x.shape
        identity = x
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.window_size is not None:
            j_idx = torch.arange(T, device=x.device, dtype=torch.long)
            i_idx = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(1)
            valid = (j_idx <= i_idx) & (j_idx >= i_idx - self.window_size + 1)
            mask = torch.where(valid, 0.0, float("-inf")).to(attn.dtype)
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=attn.dtype),
                diagonal=1,
            )
            attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = out.permute(0, 2, 1).contiguous()
        return out + identity


class WanKWiseAttention(nn.Module):
    r"""
    Self-attention over the **joint (K) dimension**.
    Input x: [B, C, T, K] — for each (b, t) there are K joints, each with C channels.
    Attention is applied over the K dimension: each of the K joints attends to all K joints
    (joint-wise self-attention). Not over channels.

    Args:
        dim (int): Channel dimension C per joint (input shape ... K has C channels per position).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = WanRMSNorm(dim)
        self.to_qkv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, K = x.size()

        x = x.permute(0, 2, 1, 3).reshape(batch_size * time, channels, K)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, K)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, K] -> [b, c, t, K]
        x = x.view(batch_size, time, channels, K)
        x = x.permute(0, 2, 1, 3)

        return x + identity


if __name__ == "__main__":
    attention = WanKWiseAttention(dim=32)
    x = torch.randn(2, 32, 17, 22)
    print(attention(x).shape)
