"""
Rotary Position Embedding (RoPE) module for MotionWan Transformer.

This module implements 2D rotary position embeddings specifically designed for
motion data, which has two spatial dimensions: temporal frames and body joints.
The RoPE is factorized into separate embeddings for each dimension, following
the approach used in video/image transformers.

Reference:
    - RoFormer: Enhanced Transformer with Rotary Position Embedding
      (https://arxiv.org/abs/2104.09864)
    - Rotary Position Embedding for Vision Transformer
      (https://arxiv.org/abs/2403.13298)
"""

from typing import Tuple
from torch import nn
import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed


class MotionWanRotaryPosEmbed(nn.Module):
    """
    2D Rotary Position Embedding for motion sequences.

    This module generates rotary position embeddings for motion data with two
    spatial dimensions: temporal (frames) and spatial (joints). The embedding
    is factorized into two 1D embeddings that are later combined, allowing the
    model to capture both temporal dynamics and spatial joint relationships.

    The attention head dimension is split between the two axes:
        - First half (t_dim): Encodes temporal/frame position
        - Second half (j_dim): Encodes spatial/joint position

    Architecture:
        1. Pre-compute 1D RoPE for max_seq_len positions for both dimensions
        2. During forward pass, slice and reshape based on actual input size
        3. Expand and combine to create 2D positional encoding

    Args:
        attention_head_dim (int): Dimension of each attention head. This will be
            split between temporal and spatial dimensions (temporal gets the
            larger half if odd).
        patch_size (Tuple[int, int]): Patch size as (patch_frames, patch_joints).
            Used to compute the number of patches along each dimension.
        max_seq_len (int): Maximum sequence length for pre-computing RoPE.
            Should be >= max(num_frames // patch_frames, num_joints // patch_joints).
        theta (float): Base frequency for rotary embeddings. Higher values lead to
            slower frequency decay. Defaults to 10000.0.

    Attributes:
        freqs_cos (torch.Tensor): Pre-computed cosine frequencies.
            Shape: (max_seq_len, attention_head_dim).
        freqs_sin (torch.Tensor): Pre-computed sine frequencies.
            Shape: (max_seq_len, attention_head_dim).

    Example:
        >>> rope = MotionWanRotaryPosEmbed(
        ...     attention_head_dim=64,
        ...     patch_size=(1, 1),
        ...     max_seq_len=256,
        ... )
        >>> # Input: (batch, channels, frames, joints)
        >>> hidden_states = torch.randn(2, 128, 64, 22)
        >>> freqs_cos, freqs_sin = rope(hidden_states)
        >>> # Output shapes: (1, 64*22, 1, 64)
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        # Split attention head dimension between temporal and joint axes
        # j_dim gets half, t_dim gets the rest (handles odd dimensions)
        j_dim = attention_head_dim // 2
        t_dim = attention_head_dim - j_dim

        # Use float64 for frequency computation precision (float32 on MPS)
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        freqs_cos = []
        freqs_sin = []

        # Compute 1D RoPE for each dimension (temporal and joint)
        for dim in [t_dim, j_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,  # Return real-valued cos/sin instead of complex
                repeat_interleave_real=True,  # Interleave for real rotation
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        # Concatenate temporal and joint frequencies along the last dimension
        # Shape: (max_seq_len, attention_head_dim)
        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary position embeddings for the given input tensor.

        This method generates 2D positional encodings by:
        1. Computing the number of patches in each dimension
        2. Slicing pre-computed frequencies to match input dimensions
        3. Expanding and combining frequencies for 2D grid structure

        Args:
            hidden_states (torch.Tensor): Input tensor with shape
                (batch_size, num_channels, num_frames, num_joints).
                Note: The actual values are not used; only the shape is needed
                to determine the output dimensions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - freqs_cos: Cosine frequencies for rotary embedding.
                    Shape: (1, num_patches, 1, attention_head_dim)
                    where num_patches = (num_frames // p_t) * (num_joints // p_j)
                - freqs_sin: Sine frequencies for rotary embedding.
                    Shape: (1, num_patches, 1, attention_head_dim)

        Note:
            The output tensors are shaped for broadcasting with attention scores
            in the transformer layers. The batch and head dimensions are
            singleton to enable broadcasting.
        """
        # Extract input dimensions
        batch_size, num_channels, num_frames, num_joints = hidden_states.shape

        # Calculate number of patches per dimension
        p_t, p_j = self.patch_size  # patch size for (frames, joints)
        ppf = num_frames // p_t  # patches per frame dimension
        ppj = num_joints // p_j  # patches per joint dimension

        # Define split sizes to separate temporal and joint frequencies
        split_sizes = [
            self.attention_head_dim - (self.attention_head_dim // 2),  # t_dim
            self.attention_head_dim // 2,  # j_dim
        ]

        # Split concatenated frequencies back into temporal and joint components
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        # Slice and expand temporal frequencies: (ppf,) -> (ppf, ppj, t_dim)
        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, -1).expand(ppf, ppj, -1)
        freqs_cos_j = freqs_cos[1][:ppj].view(1, ppj, -1).expand(ppf, ppj, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, -1).expand(ppf, ppj, -1)
        freqs_sin_j = freqs_sin[1][:ppj].view(1, ppj, -1).expand(ppf, ppj, -1)

        # Concatenate temporal and joint frequencies and reshape for attention
        # Final shape: (1, ppf * ppj, 1, attention_head_dim)
        # - dim 0: batch (broadcast)
        # - dim 1: sequence (patches)
        # - dim 2: heads (broadcast)
        # - dim 3: head dimension
        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_j], dim=-1).reshape(
            1, ppf * ppj, 1, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_j], dim=-1).reshape(
            1, ppf * ppj, 1, -1
        )

        return freqs_cos, freqs_sin


if __name__ == "__main__":
    """
    Test script for MotionWanRotaryPosEmbed module.

    This script validates:
    1. Basic initialization and forward pass
    2. Output shapes match expected dimensions
    3. Different input configurations (varying frames, joints, patch sizes)
    4. Frequency values are within expected ranges
    5. Consistency of outputs for same inputs
    """
    print("=" * 60)
    print("Testing MotionWanRotaryPosEmbed module")
    print("=" * 60)

    # ==================== Test Configuration ====================
    batch_size = 2
    num_channels = 128
    num_frames = 64
    num_joints = 22
    attention_head_dim = 64
    patch_size = (1, 1)
    max_seq_len = 256

    # ==================== Test 1: Basic Initialization ====================
    print("\n[Test 1] Basic Initialization")
    print("-" * 50)

    rope = MotionWanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        theta=10000.0,
    )

    print(f"attention_head_dim: {rope.attention_head_dim}")
    print(f"patch_size: {rope.patch_size}")
    print(f"max_seq_len: {rope.max_seq_len}")
    print(f"freqs_cos buffer shape: {rope.freqs_cos.shape}")
    print(f"freqs_sin buffer shape: {rope.freqs_sin.shape}")

    # Validate buffer shapes
    assert rope.freqs_cos.shape == (
        max_seq_len,
        attention_head_dim,
    ), "freqs_cos shape mismatch!"
    assert rope.freqs_sin.shape == (
        max_seq_len,
        attention_head_dim,
    ), "freqs_sin shape mismatch!"
    print("✓ Initialization assertions passed!")

    # ==================== Test 2: Basic Forward Pass ====================
    print("\n[Test 2] Basic Forward Pass")
    print("-" * 50)

    hidden_states = torch.randn(batch_size, num_channels, num_frames, num_joints)
    freqs_cos, freqs_sin = rope(hidden_states)

    p_t, p_j = patch_size
    expected_seq_len = (num_frames // p_t) * (num_joints // p_j)

    print(f"Input hidden_states shape: {hidden_states.shape}")
    print(f"Output freqs_cos shape: {freqs_cos.shape}")
    print(f"Output freqs_sin shape: {freqs_sin.shape}")
    print(f"Expected sequence length: {expected_seq_len}")

    # Validate output shapes
    expected_shape = (1, expected_seq_len, 1, attention_head_dim)
    assert (
        freqs_cos.shape == expected_shape
    ), f"freqs_cos shape mismatch! Got {freqs_cos.shape}, expected {expected_shape}"
    assert (
        freqs_sin.shape == expected_shape
    ), f"freqs_sin shape mismatch! Got {freqs_sin.shape}, expected {expected_shape}"
    print("✓ Forward pass shape assertions passed!")

    # ==================== Test 3: Different Patch Sizes ====================
    print("\n[Test 3] Different Patch Sizes")
    print("-" * 50)

    test_configs = [
        ((1, 1), 64, 22),  # No patching
        ((2, 1), 64, 22),  # Temporal patching only
        ((1, 2), 64, 22),  # Joint patching only
        ((4, 2), 64, 22),  # Both dimensions patched
    ]

    for p_size, n_frames, n_joints in test_configs:
        rope_test = MotionWanRotaryPosEmbed(
            attention_head_dim=attention_head_dim,
            patch_size=p_size,
            max_seq_len=max_seq_len,
        )
        test_input = torch.randn(batch_size, num_channels, n_frames, n_joints)
        cos_out, sin_out = rope_test(test_input)

        expected_patches = (n_frames // p_size[0]) * (n_joints // p_size[1])
        print(
            f"  patch_size={p_size}, frames={n_frames}, joints={n_joints} "
            f"-> seq_len={expected_patches}, output_shape={cos_out.shape}"
        )

        assert cos_out.shape[1] == expected_patches, "Sequence length mismatch!"

    print("✓ Different patch size tests passed!")

    # ==================== Test 4: Frequency Value Validation ====================
    print("\n[Test 4] Frequency Value Validation")
    print("-" * 50)

    # Check that cos/sin values are in valid range [-1, 1]
    assert (
        freqs_cos.min() >= -1.0 and freqs_cos.max() <= 1.0
    ), "freqs_cos values out of range!"
    assert (
        freqs_sin.min() >= -1.0 and freqs_sin.max() <= 1.0
    ), "freqs_sin values out of range!"

    # Check that cos^2 + sin^2 ≈ 1 (Pythagorean identity)
    identity_check = freqs_cos**2 + freqs_sin**2
    identity_error = (identity_check - 1.0).abs().max().item()
    print(f"  freqs_cos range: [{freqs_cos.min():.4f}, {freqs_cos.max():.4f}]")
    print(f"  freqs_sin range: [{freqs_sin.min():.4f}, {freqs_sin.max():.4f}]")
    print(f"  cos²+sin² identity error (max): {identity_error:.2e}")

    assert identity_error < 1e-5, "Pythagorean identity not satisfied!"
    print("✓ Frequency value validation passed!")

    # ==================== Test 5: Consistency Check ====================
    print("\n[Test 5] Consistency Check (Same Input -> Same Output)")
    print("-" * 50)

    # Run forward twice with same input
    cos1, sin1 = rope(hidden_states)
    cos2, sin2 = rope(hidden_states)

    assert torch.allclose(cos1, cos2), "Inconsistent freqs_cos output!"
    assert torch.allclose(sin1, sin2), "Inconsistent freqs_sin output!"
    print("  Repeated forward passes produce identical outputs")
    print("✓ Consistency check passed!")

    # ==================== Test 6: Different Input Sizes ====================
    print("\n[Test 6] Different Input Sizes")
    print("-" * 50)

    size_configs = [
        (32, 22),  # Shorter sequence
        (128, 22),  # Longer sequence
        (64, 44),  # More joints
        (100, 30),  # Non-standard dimensions
    ]

    for n_frames, n_joints in size_configs:
        test_input = torch.randn(batch_size, num_channels, n_frames, n_joints)
        cos_out, sin_out = rope(test_input)

        expected_patches = (n_frames // patch_size[0]) * (n_joints // patch_size[1])
        print(
            f"  frames={n_frames}, joints={n_joints} -> "
            f"patches={expected_patches}, shape={cos_out.shape}"
        )

        assert cos_out.shape == (
            1,
            expected_patches,
            1,
            attention_head_dim,
        ), "Shape mismatch!"

    print("✓ Different input size tests passed!")

    # ==================== Test 7: Parameter Statistics ====================
    print("\n[Test 7] Module Statistics")
    print("-" * 50)

    # Count parameters (RoPE typically has no learnable parameters)
    total_params = sum(p.numel() for p in rope.parameters())
    buffer_elements = rope.freqs_cos.numel() + rope.freqs_sin.numel()

    print(f"  Learnable parameters: {total_params}")
    print(f"  Buffer elements (freqs_cos + freqs_sin): {buffer_elements:,}")
    print(f"  Memory for buffers: {buffer_elements * 4 / 1024:.2f} KB (float32)")

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
