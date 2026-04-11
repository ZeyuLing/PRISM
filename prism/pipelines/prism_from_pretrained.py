# Load PRISM pipeline from a HuggingFace-style checkpoint directory.
# Self-contained: no mmotion dependency.

import json
import os
from typing import Optional, Union

import torch

from prism.models.autoencoders import AutoencoderKLPrism2DTK
from prism.pipelines.prism_ar_t2m_pipeline import PrismARPipeline
from prism.registry import HF_MODELS, MODELS


def _load_state_dict_from_dir(
    component_dir: str, device: torch.device = torch.device("cpu")
):
    """Load state dict from a subdir (model.safetensors or pytorch_model.bin)."""
    path_safe = os.path.join(component_dir, "model.safetensors")
    path_bin = os.path.join(component_dir, "pytorch_model.bin")
    if os.path.isfile(path_safe):
        from safetensors.torch import load_file
        return load_file(path_safe, device=str(device))
    if os.path.isfile(path_bin):
        try:
            return torch.load(path_bin, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path_bin, map_location=device)
    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin in {component_dir}"
    )


def load_prism_pipeline_from_pretrained(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    *,
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    tokenizer_path: Optional[str] = None,
    text_encoder_path: Optional[str] = None,
    smpl_processor_cfg: Optional[dict] = None,
    vae_from_pretrained: Optional[str] = None,
):
    """
    Load PRISM AR pipeline from a HuggingFace-style checkpoint directory.

    Args:
        pretrained_model_name_or_path: Path to the prism/ directory (containing
            config.json, transformer/, text_encoder/, vae/).
        device: Device to load models onto (default: cuda:0 if available else cpu).
        torch_dtype: dtype for inference (default: float32).
        tokenizer_path: Path to tokenizer. If None, uses env PRISM_TOKENIZER_PATH.
        text_encoder_path: Path to text encoder. If None, uses checkpoint text_encoder/.
        smpl_processor_cfg: Full config dict for SMPLPoseProcessor. If None, uses default.
        vae_from_pretrained: If checkpoint has no vae/ weights, load VAE from this path.

    Returns:
        PrismARPipeline ready for inference.
    """
    path = os.path.abspath(os.path.expanduser(str(pretrained_model_name_or_path)))
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Not a directory: {path}")

    with open(os.path.join(path, "config.json")) as f:
        top_cfg = json.load(f)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    # ── Transformer ──
    from prism.models.transformers.motion_prism import PrismTransformerMotionModel
    trans_cfg = top_cfg.get("transformer", {}).copy()
    if "model_type" in trans_cfg:
        trans_cfg.pop("model_type")
    if isinstance(trans_cfg.get("patch_size"), list):
        trans_cfg["patch_size"] = tuple(trans_cfg["patch_size"])
    transformer = PrismTransformerMotionModel(**trans_cfg)
    trans_dir = os.path.join(path, "transformer")
    if not os.path.isdir(trans_dir):
        raise FileNotFoundError(f"Transformer dir not found: {trans_dir}")
    trans_sd = _load_state_dict_from_dir(trans_dir, device)
    missing, unexpected = transformer.load_state_dict(trans_sd, strict=False)
    if missing:
        raise KeyError(f"Transformer missing keys: {list(missing)[:10]}")
    transformer = transformer.to(device=device, dtype=torch_dtype)

    # ── Text encoder ──
    te_dir = os.path.join(path, "text_encoder")
    if text_encoder_path and os.path.isdir(text_encoder_path):
        _te_path = text_encoder_path
    elif os.path.isdir(te_dir):
        _te_path = te_dir
    else:
        _te_path = os.environ.get(
            "PRISM_TEXT_ENCODER_PATH",
            os.path.join(path, "text_encoder"),
        )
    if not os.path.isdir(_te_path):
        raise FileNotFoundError(
            f"Text encoder not found: {_te_path}. "
            f"Set PRISM_TEXT_ENCODER_PATH env var or put text_encoder/ in checkpoint."
        )
    from transformers import UMT5EncoderModel
    text_encoder = UMT5EncoderModel.from_pretrained(
        _te_path, torch_dtype=torch_dtype, device_map=device
    )

    # ── Tokenizer ──
    tok_path = tokenizer_path or os.environ.get("PRISM_TOKENIZER_PATH", "")
    if not tok_path or not os.path.isdir(tok_path):
        tok_path = os.path.join(os.path.dirname(_te_path), "tokenizer")
    if not os.path.isdir(tok_path):
        tok_path = os.path.join(path, "tokenizer")
    if not os.path.isdir(tok_path):
        raise FileNotFoundError(
            f"Tokenizer not found. Set PRISM_TOKENIZER_PATH env var or put tokenizer/ in checkpoint."
        )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # ── VAE ──
    vae_dir = os.path.join(path, "vae")
    has_vae_weights = (
        os.path.isdir(vae_dir)
        and (
            os.path.isfile(os.path.join(vae_dir, "model.safetensors"))
            or os.path.isfile(os.path.join(vae_dir, "pytorch_model.bin"))
        )
    )
    if has_vae_weights:
        # diffusers from_pretrained expects diffusion_pytorch_model.safetensors
        # but our checkpoint may use model.safetensors. Load manually if needed.
        vae_config_path = os.path.join(vae_dir, "config.json")
        vae_safe = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        vae_bin = os.path.join(vae_dir, "diffusion_pytorch_model.bin")
        if os.path.isfile(vae_safe) or os.path.isfile(vae_bin):
            vae = AutoencoderKLPrism2DTK.from_pretrained(vae_dir)
        else:
            # Load from config + manual state_dict
            with open(vae_config_path) as f:
                vae_cfg = json.load(f)
            vae_cfg.pop("_class_name", None)
            vae_cfg.pop("_diffusers_version", None)
            vae_cfg.pop("model_type", None)
            vae = AutoencoderKLPrism2DTK(**vae_cfg)
            vae_sd = _load_state_dict_from_dir(vae_dir, device)
            vae.load_state_dict(vae_sd, strict=False)
    elif vae_from_pretrained and os.path.isdir(vae_from_pretrained):
        vae = AutoencoderKLPrism2DTK.from_pretrained(vae_from_pretrained)
    else:
        raise FileNotFoundError(
            "No vae/ in checkpoint and vae_from_pretrained not provided."
        )
    vae = vae.to(device=device, dtype=torch_dtype)

    # ── SMPL processor ──
    if smpl_processor_cfg is not None:
        smpl_processor = MODELS.build(smpl_processor_cfg)
    else:
        from prism.models.motion_processor.smpl_processor import SMPLPoseProcessor

        # All paths resolved relative to the checkpoint directory.
        # Users can override via env vars if they need custom locations.
        stats_file = os.environ.get("PRISM_STATS_FILE", "")
        if not stats_file or not os.path.isfile(stats_file):
            stats_file = os.path.join(path, "stats.json")
        if not os.path.isfile(stats_file):
            raise FileNotFoundError(
                f"Stats file not found at {os.path.join(path, 'stats.json')}. "
                f"Place stats.json in the checkpoint directory or set PRISM_STATS_FILE env var."
            )

        smpl_model_path = os.environ.get("PRISM_SMPL_MODEL_PATH", "")
        if not smpl_model_path or not os.path.isdir(smpl_model_path):
            smpl_model_path = os.path.join(path, "smpl_models", "smplx")
        if not os.path.isdir(smpl_model_path):
            raise FileNotFoundError(
                f"SMPL model not found at {os.path.join(path, 'smpl_models', 'smplx')}. "
                f"Place smpl_models/smplx/ in the checkpoint directory or set PRISM_SMPL_MODEL_PATH env var."
            )

        smoothnet_ckpt = os.environ.get("PRISM_SMOOTHNET_CKPT", "")
        if not smoothnet_ckpt or not os.path.isfile(smoothnet_ckpt):
            smoothnet_ckpt = os.path.join(path, "smoothnet", "smoothnet_windowsize64.pth.tar")

        smooth_model = dict(
            type="smoothnet",
            window_size=64,
            output_size=64,
            checkpoint=smoothnet_ckpt if os.path.isfile(smoothnet_ckpt) else None,
        )

        smpl_processor = SMPLPoseProcessor(
            do_normalize=True,
            stats_file=stats_file,
            rot_type="rotation_6d",
            transl_type="abs_rel",
            smpl_type="smpl_22",
            smpl_model=dict(
                type="SmplxLiteV437Coco17",
                model_path=smpl_model_path,
            ),
            smooth_model=smooth_model,
        )

    # ── Scheduler ──
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=5.0,
        use_dynamic_shifting=False,
        base_shift=0.5,
        max_shift=1.15,
    )

    pipe = PrismARPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        smpl_processor=smpl_processor,
        expand_timesteps=True,
        dtype=torch_dtype,
    )
    return pipe
