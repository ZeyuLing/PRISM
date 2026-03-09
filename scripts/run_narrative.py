#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Narrative / free-form text: one long description.
Without rewriter: split by semicolon and use as segments.
With rewriter: optional LLM service to decompose into segments (you host the API).

  python opensource/prism/scripts/run_narrative.py \\
    --checkpoint opensource/prism/pretrained_models/prism_1.4b \\
    --text "A person walks in, waves, sits down, then stands and bows." \\
    --no_rewriter \\
    --output_dir outputs/narrative
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="PRISM narrative: long text → motion (optional rewriter)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="opensource/prism/pretrained_models/prism_1.4b",
        help="Path to HuggingFace-style prism checkpoint dir",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Long natural-language description of the motion",
    )
    parser.add_argument(
        "--no_rewriter",
        action="store_true",
        help="Do not call any rewriter; split --text by semicolon as segment prompts",
    )
    parser.add_argument(
        "--rewriter_url",
        type=str,
        default=None,
        help="If set (and not --no_rewriter), URL of rewriter API to decompose text into segments",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/narrative",
        help="Output directory",
    )
    parser.add_argument(
        "--num_frames_per_segment",
        type=int,
        default=129,
        help="Frames per segment when not using rewriter (or rewriter does not return lengths)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.no_rewriter or not args.rewriter_url:
        # Simple split by semicolon
        prompts = [p.strip() for p in args.text.split(";") if p.strip()]
        if not prompts:
            prompts = [args.text.strip() or "A person moves."]
        num_frames_per_segment = args.num_frames_per_segment
        lengths = None
    else:
        # Optional: call rewriter to get segments + lengths (not implemented here;
        # user can plug in their own client)
        raise NotImplementedError(
            "Rewriter API client not implemented in this script. Use --no_rewriter and split --text by semicolon."
        )

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
        prompts=prompts,
        negative_prompt="",
        num_frames_per_segment=num_frames_per_segment if lengths is None else lengths,
        num_joints=23,
        guidance_scale=args.guidance_scale,
    )
    out_path = os.path.join(out_dir, "smplx_dict.npz")
    np.savez(out_path, **smplx_dict)
    with open(os.path.join(out_dir, "prompts.txt"), "w") as f:
        f.write(f"Input text: {args.text}\n\n")
        for i, p in enumerate(prompts):
            f.write(f"Segment {i + 1}: {p}\n")
    print(f"Saved: {out_path} ({smplx_dict['transl'].shape[0]} frames)")


if __name__ == "__main__":
    main()
