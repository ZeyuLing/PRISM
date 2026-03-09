import os
import sys

sys.path.append(os.curdir)

from diffusers import DiffusionPipeline
from einops import rearrange
import numpy as np
import torch
from transformers import PreTrainedTokenizer, UMT5EncoderModel
from mmotion.models.autoencoders import AutoencoderKLPrism2DTK
from mmotion.models.motion_processor.smpl_processor import SMPLPoseProcessor
from mmotion.models.transformers.motion_prism import PrismTransformerMotionModel
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from typing import Any, Dict, List, Optional, Union
from mmengine import print_log
from mmotion.registry import HF_MODELS
from mmotion.trainers.trainer_prism.trainer_tp2m_prism import PrismTrainer

from mmotion.utils.geometry.rotation_convert import rotation_6d_to_axis_angle

from diffusers.utils.torch_utils import randn_tensor

 
class PrismPipeline(DiffusionPipeline):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLPrism2DTK,
        scheduler: FlowMatchEulerDiscreteScheduler,
        smpl_processor: SMPLPoseProcessor,
        transformer: PrismTransformerMotionModel,
        expand_timesteps: bool = False,  # True for our implementation
    ):
        device = next(transformer.parameters()).device
        dtype = torch.bfloat16
        super().__init__()

        self.register_modules(
            vae=vae.to(device, dtype),
            text_encoder=text_encoder.to(device, dtype),
            tokenizer=tokenizer,
            transformer=transformer.to(device, dtype),
            scheduler=scheduler,
        )

        self.register_to_config(expand_timesteps=expand_timesteps)

        self.smpl_processor = smpl_processor.to(device, dtype)

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
    ) -> torch.Tensor:

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            num_joints,
        )

        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_frames: int = 361,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        use_static: bool = False,
        use_smooth: bool = False,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
        max_sequence_length: int = 512,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = next(self.transformer.parameters()).device
        do_cfg = guidance_scale > 1.0
        if num_frames % self.vae_scale_factor_temporal != 1:
            print_log(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        else:
            assert isinstance(prompt, list)
            batch_size = len(prompt)

        # 3. Encode input prompt
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

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        # [B, C, T, J]
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            num_joints=num_joints,
            num_frames=num_frames,
            dtype=transformer_dtype,
            device=device,
        )
        # b c t j
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                current_model = self.transformer
                current_guidance_scale = guidance_scale

                latent_model_input = latents.to(transformer_dtype)

                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * num_latent_joints
                    temp_ts = (mask[0][0] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                )

                if do_cfg:
                    noise_uncond = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                    )
                    noise_pred = noise_uncond + current_guidance_scale * (
                        noise_pred - noise_uncond
                    )
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        motion_vec = self.decode_motion(latents)

        smplx_dict = self.post_process_motion(
            motion_vec,
            use_static=use_static,
            use_smooth=use_smooth,
            mocap_framerate=mocap_framerate,
            gender=gender,
        )

        return smplx_dict

    def decode_motion(self, latents):
        latents = latents * self.latents_std.to(latents.device) + self.latents_mean.to(latents.device)
        motion = self.vae.decode(latents)
        return motion

    def post_process_motion(
        self,
        x_dec,
        use_static: bool = False,
        use_smooth: bool = False,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
    ) -> Dict:
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
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_motion_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

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

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_motion_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_motion_per_prompt, seq_len, -1
        )

        return prompt_embeds


def main(
    prompt: str = "A person is running.",
    negative_prompt: str = "",
    use_static: bool = True,
    use_smooth: bool = True,
    mocap_framerate: float = 30.0,
    gender: str = "neutral",
    num_frames: int = 81,
    num_joints=23,  # 22 rotation + 1 translation
    guidance_scale: int = 5,
    trainer_cfg: str = "configs/motionwan/motionwan_5b_tp2m.py",
    trainer_ckpt: str = "work_dirs/motionwan_5b_tp2m/iter_52000.pth",
    output_path: str = "outputs/motionwan_5b_tp2m",
):
    from mmengine import Config
    from mmengine.runner import load_checkpoint
    from mmotion.registry import MODELS

    output_path = os.path.join(
        output_path,
        os.path.basename(trainer_ckpt).split(".")[0],
        prompt.replace(" ", "_")[:50],
    )
    os.makedirs(output_path, exist_ok=True)

    trainer_cfg = Config.fromfile(trainer_cfg)["model"]
    trainer: PrismTrainer = MODELS.build(trainer_cfg)
    load_checkpoint(trainer, trainer_ckpt, strict=True, map_location="cpu")

    pipe = PrismPipeline(
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
        expand_timesteps=True,
    )
    smplx_dict = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_static=use_static,
        use_smooth=use_smooth,
        mocap_framerate=mocap_framerate,
        gender=gender,
        num_frames=num_frames,
        num_joints=num_joints,
        guidance_scale=guidance_scale,
    )

    np.savez(
        os.path.join(output_path, "smplx_dict.npz"),
        **smplx_dict,
    )
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
