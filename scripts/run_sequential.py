#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sequential multi-segment: list of prompts (and optional per-segment lengths).

  python scripts/run_sequential.py \
    --checkpoint pretrained_models/prism_1.4b \
    --prompts "A person waves." "A person walks." "A person bows." \
    --lengths 65 129 65 \
    --output_dir outputs/sequential
"""

import argparse
import os
import sys

_PRISM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PRISM_ROOT not in sys.path:
    sys.path.insert(0, _PRISM_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="PRISM sequential: multiple prompts → long motion"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_models/prism_1.4b",
        help="Path to HuggingFace-style prism checkpoint dir",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="One or more text prompts (one per segment)",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="*",
        default=None,
        help="Optional: one length per segment (frames). If omitted, 129 for all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sequential",
        help="Output directory",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts if p.strip()]
    if not prompts:
        raise ValueError("At least one prompt required")
    if args.lengths is not None and len(args.lengths) != len(prompts):
        raise ValueError(
            f"lengths count ({len(args.lengths)}) must match prompts count ({len(prompts)})"
        )
    num_frames_per_segment = args.lengths if args.lengths else 129

    from prism.pipelines.prism_from_pretrained import (
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
        prompts=prompts,
        negative_prompt="",
        num_frames_per_segment=num_frames_per_segment,
        num_joints=23,
        guidance_scale=args.guidance_scale,
    )
    out_path = os.path.join(out_dir, "smplx_dict.npz")
    np.savez(out_path, **smplx_dict)
    with open(os.path.join(out_dir, "prompts.txt"), "w") as f:
        for i, p in enumerate(prompts):
            f.write(f"Segment {i + 1}: {p}\n")
    print(f"Saved: {out_path} ({smplx_dict['transl'].shape[0]} frames)")


if __name__ == "__main__":
    main()
