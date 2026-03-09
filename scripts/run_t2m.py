#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text-to-motion: single prompt → one motion clip.
Run from the main repo root (versatilemotion). Uses HuggingFace-style checkpoint.

  python opensource/prism/scripts/run_t2m.py \\
    --checkpoint opensource/prism/pretrained_models/prism_1.4b \\
    --prompt "A person walks forward and waves." \\
    --output_dir outputs/t2m
"""

import argparse
import os
import sys

# Add main repo root so we can import mmotion
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="PRISM T2M: single prompt → motion")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="opensource/prism/pretrained_models/prism_1.4b",
        help="Path to HuggingFace-style prism checkpoint dir",
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
        default="outputs/t2m",
        help="Output directory for smplx_dict.npz",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=129,
        help="Number of frames (default 129)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda:0 or cpu)",
    )
    args = parser.parse_args()

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

    smplx_dict = pipe(
        prompts=args.prompt,
        negative_prompt="",
        num_frames_per_segment=args.num_frames,
        num_joints=23,
        guidance_scale=args.guidance_scale,
    )
    out_path = os.path.join(out_dir, "smplx_dict.npz")
    np.savez(out_path, **smplx_dict)
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(args.prompt + "\n")
    print(f"Saved: {out_path} ({smplx_dict['transl'].shape[0]} frames)")


if __name__ == "__main__":
    main()
