import os
import sys
import logging

from diffusers import DiffusionPipeline
from einops import rearrange
import numpy as np
import torch
from transformers import PreTrainedTokenizer, UMT5EncoderModel
from prism.models.autoencoders import AutoencoderKLPrism2DTK
from prism.models.autoencoders.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)
from prism.models.motion_processor.smpl_processor import SMPLPoseProcessor
from prism.models.transformers.motion_prism import PrismTransformerMotionModel
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from prism.registry import HF_MODELS

from prism.utils.geometry.rotation_convert import rotation_6d_to_axis_angle

from diffusers.utils.torch_utils import randn_tensor


class PrismARPipeline(DiffusionPipeline):
    """Autoregressive Text-to-Motion Pipeline.

    This pipeline generates long motion sequences by autoregressively generating
    multiple segments. Each segment uses the last frame of the previous segment
    as the first frame condition for the next segment.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLPrism2DTK,
        scheduler: FlowMatchEulerDiscreteScheduler,
        smpl_processor: SMPLPoseProcessor,
        transformer: PrismTransformerMotionModel,
        expand_timesteps: bool = True,
        is_causal: bool = False,
        dtype=torch.bfloat16,
    ):
        device = next(transformer.parameters()).device
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.register_to_config(expand_timesteps=expand_timesteps, is_causal=is_causal)

        self.smpl_processor: SMPLPoseProcessor = smpl_processor

        self.latents_mean = torch.tensor(
            vae.config.latents_mean, dtype=dtype, device=device
        ).view(1, self.vae.config.z_dim, 1, 1)

        self.latents_std = torch.tensor(
            vae.config.latents_std, dtype=dtype, device=device
        ).view(1, self.vae.config.z_dim, 1, 1)

        self.vae_scale_factor_temporal = vae.config.scale_factor_temporal

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        num_frames: int = 81,
        num_joints: int = 23,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        first_frame_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare latents for denoising with optional first frame conditioning.

        Args:
            batch_size: Number of samples in the batch.
            num_channels_latents: Number of latent channels.
            num_frames: Number of motion frames.
            num_joints: Number of joints.
            dtype: Data type for tensors.
            device: Device to place tensors on.
            first_frame_latents: Optional encoded first frame latents [B, C, 1, J].

        Returns:
            latents: Random noise tensor [B, C, T_latent, J].
            condition: Condition tensor with first frame encoded [B, C, T_latent, J].
            first_frame_mask: Mask indicating which positions to denoise [B, C, T_latent, J].
                0 for condition positions (first frame), 1 for positions to denoise.
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            num_joints,
        )

        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)

        # Create condition tensor and mask
        condition = torch.zeros_like(latents)
        first_frame_mask = torch.ones_like(latents)

        if first_frame_latents is not None:
            # first_frame_latents: [B, C, 1, J] or [1, C, 1, J]
            # Expand batch dimension if needed
            if first_frame_latents.shape[0] == 1 and batch_size > 1:
                first_frame_latents = first_frame_latents.expand(batch_size, -1, -1, -1)
            # Set the first frame condition
            condition[:, :, :1, :] = first_frame_latents
            # Mask: 0 for first frame (keep condition), 1 for rest (to denoise)
            first_frame_mask[:, :, :1, :] = 0.0

        return latents, condition, first_frame_mask

    def load_condition_pose(self, motion_path: str) -> torch.Tensor:
        """Load and process condition pose from npz file.

        Args:
            motion_path: Path to the npz file containing motion data.

        Returns:
            Processed motion tensor of shape [1, 1, J, C] ready for VAE encoding.
            Where C=6 (6D rotation representation), J=num_joints.
            VAE expects [B, T, K, C] format.
        """
        device = self.vae.device
        dtype = self.vae.dtype

        smplx_dict = self.smpl_processor.load_smplx_dict_from_npz(motion_path)
        # [T, D] where D = J * 6
        motion = (
            self.smpl_processor.smplx_dict_to_motion_vector(smplx_dict)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        motion = self.smpl_processor.normalize(motion)

        # [B, T, D] -> [B, T, J, 6]
        motion = rearrange(motion, "b t (j d) -> b t j d", d=6)

        # Only use the first frame for condition
        if motion.shape[1] != 1:
            logging.info(
                f"Warning: Original motion has {motion.shape[1]} frames, only use the first frame for condition pose"
            )
            motion = motion[:, :1]  # [B, 1, J, 6]

        # Return in VAE expected format: [B, T, J, C]
        return motion.to(device=device, dtype=dtype)

    def extract_last_frame_motion(self, motion_vec: torch.Tensor) -> torch.Tensor:
        """Extract the last frame from decoded motion for autoregressive conditioning.

        Args:
            motion_vec: Decoded motion tensor of shape [B, T, J, C] from VAE.

        Returns:
            Last frame motion tensor of shape [B, 1, J, C] ready for VAE encoding.
        """
        # motion_vec is [B, T, J, C], extract last frame
        last_frame = motion_vec[:, -1:, :, :]  # [B, 1, J, C]
        return last_frame

    @torch.no_grad()
    def encode_motion(
        self,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        """Encode motion to VAE latent space.

        Args:
            motion: Motion tensor of shape [B, T, J, C] where C=6 (6D rotation).
                This is the format expected by VAE.encode().

        Returns:
            Latent tensor of shape [B, Z_dim, T_latent, J].
        """
        # Encode by SMPL VAE: [B, T, J, C] -> [B, Z_dim*2, T_latent, J]
        # VAE internally permutes to [B, C, T, J] before encoding
        z = self.vae.encode(motion)

        # Sample from the latent distribution (use mode for deterministic encoding)
        lat = DiagonalGaussianDistributionNd(z)
        z = lat.mode()

        # Normalize latents
        z = (z - self.latents_mean) / self.latents_std

        return z  # [B, Z_dim, T_latent, J]

    @torch.no_grad()
    def generate_single_segment(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        first_frame_motion: Optional[torch.Tensor] = None,
        num_frames: int = 129,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        max_sequence_length: int = 512,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Generate a single motion segment.

        Args:
            prompt: Text prompt for this segment.
            negative_prompt: Negative prompt for classifier-free guidance.
            first_frame_motion: First frame condition tensor [B, 1, J, C] or None.
            num_frames: Number of frames to generate.
            num_joints: Number of joints.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            max_sequence_length: Maximum sequence length for text encoding.
            attention_kwargs: Additional kwargs for attention.

        Returns:
            Decoded motion tensor of shape [B, T, J, C].
        """
        device = next(self.transformer.parameters()).device
        do_cfg = guidance_scale > 1.0
        batch_size = 1

        # Encode first frame if provided
        first_frame_latents = None
        if first_frame_motion is not None:
            first_frame_latents = self.encode_motion(first_frame_motion)

        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_motion_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents, condition, first_frame_mask = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            num_joints=num_joints,
            num_frames=num_frames,
            dtype=transformer_dtype,
            device=device,
            first_frame_latents=first_frame_latents,
        )

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        for i, t in enumerate(timesteps):
            current_model = self.transformer
            current_guidance_scale = guidance_scale

            if self.config.expand_timesteps:
                latent_model_input = (
                    (1 - first_frame_mask) * condition + first_frame_mask * latents
                ).to(transformer_dtype)
                temp_ts = (first_frame_mask[0][0] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                is_causal=self.config.is_causal,
            )

            if do_cfg:
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    is_causal=self.config.is_causal,
                )
                noise_pred = noise_uncond + current_guidance_scale * (
                    noise_pred - noise_uncond
                )

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Force-restore condition frame latents after each step
            # so they remain noise-free throughout the entire denoising process.
            if first_frame_latents is not None:
                latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        # Merge condition back for final output (redundant but safe)
        if self.config.expand_timesteps and first_frame_latents is not None:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        # Decode to motion
        motion_vec = self.decode_motion(latents)

        return motion_vec

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        first_frame_motion_path: Optional[str] = None,
        num_frames_per_segment: Union[int, List[int]] = 129,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        use_static: bool = False,
        use_smooth: bool = False,
        normalize: bool = True,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
        max_sequence_length: int = 512,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        overlap_frames: int = 1,
    ) -> Dict:
        """Generate long motion autoregressively from multiple prompts.

        Args:
            prompts: List of text prompts, one for each segment.
                If a single string is provided, it will be wrapped in a list.
            negative_prompt: Negative prompt for classifier-free guidance.
            first_frame_motion_path: Optional path to npz file for first segment's
                first frame condition.
            num_frames_per_segment: Number of frames per segment (int for all segments, or list of int per segment).
            num_joints: Number of joints in the output motion.
            num_inference_steps: Number of denoising steps per segment.
            guidance_scale: Classifier-free guidance scale.
            use_static: Whether to use static joint refinement.
            use_smooth: Whether to apply smoothing to output motion.
            normalize: Whether to normalize facing direction and ground plane.
            mocap_framerate: Frame rate of the output motion.
            gender: Gender for SMPL model ('neutral', 'male', 'female').
            max_sequence_length: Maximum sequence length for text encoding.
            attention_kwargs: Additional kwargs for attention.
            overlap_frames: Number of overlapping frames between segments (default 1).
                The last frame of previous segment becomes the first frame of next segment.

        Returns:
            smplx_dict: Dictionary containing SMPL-X parameters for the full motion.
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]

        num_segments = len(prompts)
        logging.info(f"Generating {num_segments} motion segments autoregressively...")

        # Per-segment frame counts (round each to valid VAE length)
        scale = self.vae_scale_factor_temporal

        def _round_frames(n: int) -> int:
            if (n - 1) % scale != 0:
                return (n // scale) * scale + 1
            return max(1, n)

        if isinstance(num_frames_per_segment, list):
            if len(num_frames_per_segment) != num_segments:
                logging.info(
                    f"num_frames_per_segment list length {len(num_frames_per_segment)} != num_segments {num_segments}; using first value for all."
                )
                single = _round_frames(num_frames_per_segment[0] if num_frames_per_segment else 129)
                num_frames_per_segment_list = [single] * num_segments
            else:
                num_frames_per_segment_list = [_round_frames(n) for n in num_frames_per_segment]
        else:
            single = _round_frames(num_frames_per_segment)
            num_frames_per_segment_list = [single] * num_segments

        # Load first frame condition if provided
        first_frame_motion = None
        if first_frame_motion_path is not None:
            first_frame_motion = self.load_condition_pose(first_frame_motion_path)
            # Only use first frame
            first_frame_motion = first_frame_motion[:, :1]

        # Store all motion segments
        all_motion_segments = []

        # Generate each segment
        with self.progress_bar(total=num_segments) as progress_bar:
            for seg_idx, prompt in enumerate(prompts):
                logging.info(
                    f"Generating segment {seg_idx + 1}/{num_segments}: {prompt[:50]}..."
                )

                # Generate single segment
                num_frames_this = num_frames_per_segment_list[seg_idx]
                motion_vec = self.generate_single_segment(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    first_frame_motion=first_frame_motion,
                    num_frames=num_frames_this,
                    num_joints=num_joints,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    max_sequence_length=max_sequence_length,
                    attention_kwargs=attention_kwargs,
                )

                # Store segment (excluding first frame if not first segment to avoid duplication)
                if seg_idx == 0:
                    all_motion_segments.append(motion_vec)
                else:
                    # Skip the first frame to avoid duplication with previous segment's last frame
                    all_motion_segments.append(motion_vec[:, overlap_frames:])

                # Extract last frame as condition for next segment
                first_frame_motion = self.extract_last_frame_motion(motion_vec)

                progress_bar.update()

        # Concatenate all segments along time dimension
        # motion_vec shape: [B, T, J, C]
        full_motion = torch.cat(all_motion_segments, dim=1)
        logging.info(f"Total motion frames: {full_motion.shape[1]}")

        # Post-process to SMPL-X format
        smplx_dict = self.post_process_motion(
            full_motion,
            use_static=use_static,
            use_smooth=use_smooth,
            normalize=normalize,
            mocap_framerate=mocap_framerate,
            gender=gender,
        )

        return smplx_dict

    def decode_motion(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to motion.

        Args:
            latents: Latent tensor of shape [B, C, T_latent, J].

        Returns:
            Motion tensor of shape [B, T, J, C].
        """
        latents = latents * self.latents_std.to(latents.device) + self.latents_mean.to(latents.device)
        motion = self.vae.decode(latents)
        return motion

    def post_process_motion(
        self,
        x_dec: torch.Tensor,
        use_static: bool = False,
        use_smooth: bool = False,
        normalize: bool = True,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
    ) -> Dict:
        """Post-process decoded motion to SMPL-X format.

        Args:
            x_dec: Decoded motion tensor of shape [B, T, J, C].
            use_static: Whether to use post-hoc static joint refinement.
            use_smooth: Whether to apply smoothing.
            normalize: Whether to normalize facing direction and ground plane.
            mocap_framerate: Frame rate of the motion.
            gender: Gender for SMPL model.

        Returns:
            Dictionary containing SMPL-X parameters.
        """
        x_dec = rearrange(x_dec, "b t j d -> b t (j d)")
        x_dec = self.smpl_processor.denormalize(x_dec)
        transl_abs_rel = x_dec[..., :6]
        transl = self.smpl_processor.inv_convert_transl(transl_abs_rel)
        pred_poses = x_dec[..., 6:]

        pred_poses = rearrange(pred_poses, "b t (j d)-> (b t) j d", d=6)

        pred_poses = rotation_6d_to_axis_angle(pred_poses)
        pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

        if use_static:
            pred_poses = self.smpl_processor.post_hoc_static_refine(
                transl, pred_poses, rot_type="axis_angle"
            )

        pred_smplx_dict = self.smpl_processor.transl_pose_to_smplx_dict(
            transl.squeeze(0),
            pred_poses.squeeze(0),
            mocap_framerate=mocap_framerate,
            gender=gender,
            rot_type="axis_angle",
        )

        if use_smooth:
            pred_smplx_dict = self.smpl_processor.smooth_smplx_dict(pred_smplx_dict)

        if normalize:
            pred_smplx_dict = self.smpl_processor.normalize_smplx_dict(pred_smplx_dict)

        return pred_smplx_dict

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_motion_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Encodes the prompt into text encoder hidden states."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = 1

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_motion_per_prompt=num_motion_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        negative_prompt_embeds = None

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_motion_per_prompt=num_motion_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_motion_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_motion_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_motion_per_prompt, seq_len, -1
        )

        return prompt_embeds


def main(
    prompts: str = "A person walks forward slowly;A person jumps up;A person turns around;A person walks forward quickly",
    negative_prompt: str = "",
    first_frame_motion_path: str = None,
    use_static: bool = False,
    use_smooth: bool = False,
    mocap_framerate: float = 30.0,
    gender: str = "neutral",
    num_frames_per_segment: int = 129,
    num_joints: int = 23,
    guidance_scale: float = 5.0,
    expand_timesteps: bool = True,
    overlap_frames: int = 1,
    max_sequence_length: int = 256,
    trainer_cfg: str = "configs/motionwan/motionwan_1b_tp2m_hq_t5xxl_256text_aug_1frame.py",
    trainer_ckpt: str = "work_dirs/motionwan_1b_tp2m_hq_t5xxl_256text_aug_1frame/iter_16000.pth",
    output_path: str = "outputs/motionwan_ar_t2m/",
):
    """Main entry for Autoregressive Text-to-Motion generation.

    Args:
        prompts: Multiple prompts separated by semicolon (;).
            Example: "A person walks forward;A person turns left;A person sits down"
        negative_prompt: Negative prompt for classifier-free guidance.
        first_frame_motion_path: Optional path to npz file for first frame condition.
        use_static: Whether to use static joint refinement.
        use_smooth: Whether to apply smoothing to output motion.
        mocap_framerate: Frame rate of the output motion.
        gender: Gender for SMPL model ('neutral', 'male', 'female').
        num_frames_per_segment: Number of frames per segment.
        num_joints: Number of joints in the output motion.
        guidance_scale: Classifier-free guidance scale.
        expand_timesteps: Whether to use per-token timesteps.
        overlap_frames: Number of overlapping frames between segments.
        trainer_cfg: Path to trainer config file.
        trainer_ckpt: Path to trainer checkpoint.
        output_path: Base output directory.
    """
    from mmengine import Config
    from mmengine.runner import load_checkpoint
    from prism.registry import MODELS

    # Parse prompts (separated by semicolon)
    prompt_list = [p.strip() for p in prompts.split(";") if p.strip()]
    print(f"Number of segments to generate: {len(prompt_list)}")
    for i, p in enumerate(prompt_list):
        print(f"  Segment {i + 1}: {p}")

    # Build output path
    output_path = os.path.join(
        output_path,
        os.path.basename(trainer_ckpt).split(".")[0],
        f"ar_{len(prompt_list)}segments",
    )
    os.makedirs(output_path, exist_ok=True)

    # Save prompts to file
    with open(os.path.join(output_path, "prompts.txt"), "w") as f:
        for i, p in enumerate(prompt_list):
            f.write(f"Segment {i + 1}: {p}\n")

    # Load trainer and checkpoint
    trainer_cfg = Config.fromfile(trainer_cfg)["model"]
    trainer: PrismTrainer = MODELS.build(trainer_cfg)
    load_checkpoint(trainer, trainer_ckpt, strict=True, map_location="cpu")

    # Build pipeline
    pipe = PrismARPipeline(
        tokenizer=trainer.tokenizer,
        text_encoder=trainer.text_encoder,
        vae=trainer.vae,
        transformer=trainer.transformer,
        scheduler=HF_MODELS.build(
            dict(
                type="FlowMatchEulerDiscreteScheduler",
                num_train_timesteps=1000,
                shift=5.0,
                use_dynamic_shifting=False,
                base_shift=0.5,
                max_shift=1.15,
            ),
        ),
        smpl_processor=trainer.smpl_pose_processor,
        expand_timesteps=expand_timesteps,
    )

    # Generate motion autoregressively
    smplx_dict = pipe(
        prompts=prompt_list,
        negative_prompt=negative_prompt,
        first_frame_motion_path=first_frame_motion_path,
        use_static=use_static,
        use_smooth=use_smooth,
        mocap_framerate=mocap_framerate,
        gender=gender,
        num_frames_per_segment=num_frames_per_segment,
        num_joints=num_joints,
        guidance_scale=guidance_scale,
        overlap_frames=overlap_frames,
        max_sequence_length=max_sequence_length,
    )

    # Save output
    np.savez(
        os.path.join(output_path, "smplx_dict.npz"),
        **smplx_dict,
    )
    print(f"Output path: {output_path}")
    print(f"Total frames generated: {smplx_dict['transl'].shape[0]}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
