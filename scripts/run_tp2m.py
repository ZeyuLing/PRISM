#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pose-conditioned (TP2M): first frame NPZ or preset pose + text prompt → motion.
Run from the main repo root (versatilemotion). Uses HuggingFace-style checkpoint.

  # From NPZ file:
  python opensource/prism/scripts/run_tp2m.py \\
    --checkpoint opensource/prism/pretrained_models/prism_1.4b \\
    --first_frame_npz /path/to/first_frame.npz \\
    --prompt "The person stands up and walks." \\
    --output_dir outputs/tp2m

  # From preset pose (standing, tpose, squat, kneel, sit):
  python opensource/prism/scripts/run_tp2m.py \\
    --checkpoint opensource/prism/pretrained_models/prism_1.4b \\
    --first_frame_pose tpose \\
    --prompt "The person begins to walk forward slowly." \\
    --output_dir outputs/tp2m
"""

import argparse
import os
import sys
import tempfile

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Preset pose prompts (from generate_teaser_demos.py)
PRESET_POSE_PROMPTS = {
    "standing": ("A person stands still with arms at sides.", 33, 20),
    "tpose": (
        "A person stands still in a T-pose with arms extended to the sides.",
        33,
        20,
    ),
    "squat": (
        "A person does a deep squat and stays still.",
        65,
        50,
    ),
    "kneel": (
        "A person slowly kneels down on one knee and stays still.",
        65,
        50,
    ),
    "sit": (
        "A person sits down on the ground cross-legged and stays still.",
        65,
        50,
    ),
}


def _create_condition_pose_from_preset(pipe, pose_name: str) -> str:
    """Generate a condition pose from preset and save to temp npz."""
    if pose_name not in PRESET_POSE_PROMPTS:
        raise ValueError(
            f"Unknown pose '{pose_name}'. Available: {list(PRESET_POSE_PROMPTS.keys())}"
        )
    prompt, num_frames, frame_idx = PRESET_POSE_PROMPTS[pose_name]
    num_frames = (num_frames - 1) // 4 * 4 + 1  # VAE-compatible

    smplx_dict = pipe(
        prompts=[prompt],
        negative_prompt="",
        num_frames_per_segment=num_frames,
        num_joints=23,
        guidance_scale=5.0,
    )

    # Extract the specified frame
    cond_dict = {}
    for key, val in smplx_dict.items():
        if hasattr(val, "shape") and len(val.shape) >= 1 and val.shape[0] > frame_idx:
            cond_dict[key] = val[frame_idx : frame_idx + 1]
        else:
            cond_dict[key] = val

    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    import numpy as np

    np.savez(path, **cond_dict)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="PRISM TP2M: first frame (npz or preset) + prompt → motion"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="opensource/prism/pretrained_models/prism_1.4b",
        help="Path to HuggingFace-style prism checkpoint dir",
    )
    parser.add_argument(
        "--first_frame_npz",
        type=str,
        default=None,
        help="Path to .npz with first frame SMPL params (transl, body_pose, global_orient)",
    )
    parser.add_argument(
        "--first_frame_pose",
        type=str,
        default=None,
        choices=list(PRESET_POSE_PROMPTS.keys()),
        help="Preset pose: standing, tpose, squat, kneel, sit (generates first frame from model)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for the motion",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tp2m",
        help="Output directory",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=129,
        help="Number of frames",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.first_frame_npz and args.first_frame_pose:
        raise ValueError("Provide either --first_frame_npz or --first_frame_pose, not both.")
    if not args.first_frame_npz and not args.first_frame_pose:
        raise ValueError("Provide either --first_frame_npz or --first_frame_pose.")

    from mmotion.pipelines.prism_from_pretrained import (
        load_prism_pipeline_from_pretrained,
    )
    import numpy as np

    pipe = load_prism_pipeline_from_pretrained(
        args.checkpoint,
        device=args.device,
    )
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    first_frame_path = args.first_frame_npz
    temp_cond_path = None
    if args.first_frame_pose:
        print(f"[TP2M] Generating condition pose from preset: {args.first_frame_pose}")
        temp_cond_path = _create_condition_pose_from_preset(pipe, args.first_frame_pose)
        first_frame_path = temp_cond_path

    if not os.path.isfile(first_frame_path):
        raise FileNotFoundError(f"First frame NPZ not found: {first_frame_path}")

    try:
        smplx_dict = pipe(
            prompts=args.prompt,
            negative_prompt="",
            first_frame_motion_path=first_frame_path,
            num_frames_per_segment=args.num_frames,
            num_joints=23,
            guidance_scale=args.guidance_scale,
        )
        out_path = os.path.join(out_dir, "smplx_dict.npz")
        np.savez(out_path, **smplx_dict)
        with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
            f.write(args.prompt + "\n")
        cond_src = args.first_frame_pose or args.first_frame_npz
        with open(os.path.join(out_dir, "first_frame_source.txt"), "w") as f:
            f.write(str(cond_src) + "\n")
        print(f"Saved: {out_path} ({smplx_dict['transl'].shape[0]} frames)")
    finally:
        if temp_cond_path and os.path.isfile(temp_cond_path):
            os.remove(temp_cond_path)


if __name__ == "__main__":
    main()
