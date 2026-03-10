# PRISM: Streaming Human Motion Generation with Per-Joint Latent Decomposition

<p align="center">
  <a href="https://arxiv.org/abs/2603.08590">
    <img src="https://img.shields.io/badge/Paper-ArXiv-B31B1B?style=for-the-badge&logo=arxiv" alt="Paper"/>
  </a>
  <a href="https://huggingface.co/ZeyuLing/PRISM-TP2M-1.4B">
    <img src="https://img.shields.io/badge/Model-HuggingFace-FFBF00?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  </a>
  <a href="https://www.youtube.com/watch?v=3PBFpYcwGIM">
    <img src="https://img.shields.io/badge/Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube"/>
  </a>
  <a href="https://github.com/ZeyuLing/PRISM">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>

<p align="center"><b>Zeyu Ling</b>, <b>Qing Shuai</b>, <b>Teng Zhang</b>, <b>Shiyang Li</b>, <b>Bo Han</b>, <b>Changqing Zou</b></p>

## Abstract

Text-to-motion generation has advanced rapidly, yet two challenges persist. First, existing motion autoencoders compress each frame into a single monolithic latent vector, entangling trajectory and per-joint rotations in an unstructured representation that downstream generators struggle to model faithfully. Second, text-to-motion, pose-conditioned generation, and long-horizon sequential synthesis typically require separate models or task-specific mechanisms, with autoregressive approaches suffering from severe error accumulation over extended rollouts.

We present PRISM, addressing each challenge with a dedicated contribution. **(1) A joint-factorized motion latent space**: each body joint occupies its own token, forming a structured 2D grid (time x joints) compressed by a causal VAE with forward-kinematics supervision. This simple change to the latent space, without modifying the generator, substantially improves generation quality, revealing that latent space design has been an underestimated bottleneck. **(2) Noise-free condition injection**: each latent token carries its own timestep embedding, allowing conditioning frames to be injected as clean tokens (timestep 0) while the remaining tokens are denoised. This unifies text-to-motion and pose-conditioned generation in a single model, and directly enables autoregressive segment chaining for streaming synthesis. Self-forcing training further suppresses drift in long rollouts. With these two components, we train a single motion generation foundation model that seamlessly handles text-to-motion, pose-conditioned generation, autoregressive sequential generation, and narrative motion composition, achieving state-of-the-art on HumanML3D, MotionHub, BABEL, and a 50-scenario user study.

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=3PBFpYcwGIM">
    <img src="https://img.youtube.com/vi/3PBFpYcwGIM/maxresdefault.jpg" alt="PRISM Demo Video" width="860"/>
  </a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=3PBFpYcwGIM">
    <img src="https://img.shields.io/badge/%E2%96%B6%20Play%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Play Demo"/>
  </a>
</p>

## Checkpoint Download

Pretrained weights are hosted on Hugging Face and should be downloaded to `pretrained_models/prism_1.4b`:

```bash
pip install huggingface_hub
huggingface-cli download ZeyuLing/PRISM-TP2M-1.4B --local-dir pretrained_models/prism_1.4b
```

Or from Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("ZeyuLing/PRISM-TP2M-1.4B", local_dir="pretrained_models/prism_1.4b")
```

## Dependencies

- Python >= 3.9
- PyTorch (CUDA recommended)
- transformers, diffusers, einops, mmengine
- SMPL/SMPL-X body model (for full mesh rendering)

This project is designed to run inside the [versatilemotion](https://github.com/ZeyuLing/versatilemotion) repository.

```bash
cd /path/to/versatilemotion
pip install -r requirements.txt
```

## Quick Inference

All scripts assume you are in the `versatilemotion` repository root and checkpoint path is:
`opensource/prism/pretrained_models/prism_1.4b`.

### Task Scripts Overview

| Script | Task | Input | Output |
|---|---|---|---|
| `run_t2m.py` | Text-to-Motion (single segment) | one text prompt | `smplx_dict.npz`, `prompt.txt` |
| `run_tp2m.py` | Pose-Conditioned Generation (TP2M) | first frame (`.npz` or preset pose) + text | `smplx_dict.npz`, `prompt.txt`, `first_frame_source.txt` |
| `run_sequential.py` | Sequential Multi-Segment | multiple prompts (+ optional per-segment lengths) | `smplx_dict.npz`, `prompts.txt` |
| `run_narrative.py` | Narrative / Free-form Text | long text (semicolon split when `--no_rewriter`) | `smplx_dict.npz`, `prompts.txt` |

### Text-to-Motion

Use this script when you want one prompt -> one motion clip.

```bash
python opensource/prism/scripts/run_t2m.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --prompt "A person walks forward and waves." \
  --output_dir outputs/t2m
```

Key arguments:
- `--prompt`: required text prompt.
- `--num_frames`: generated frame count (default `129`).
- `--guidance_scale`: classifier-free guidance scale (default `5.0`).
- `--device`: optional device (e.g., `cuda:0`).

### Pose-Conditioned Generation

Use this script when the first frame is given and you want controlled continuation.

```bash
python opensource/prism/scripts/run_tp2m.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --first_frame_pose tpose \
  --prompt "The person begins to walk forward slowly and naturally." \
  --output_dir outputs/tp2m
```

Input modes:
- `--first_frame_npz /path/to/cond.npz`: provide your own first-frame condition.
- `--first_frame_pose tpose`: use built-in preset pose.

Preset poses: `standing`, `tpose`, `squat`, `kneel`, `sit`.  
Exactly one of `--first_frame_npz` and `--first_frame_pose` must be provided.

### Sequential Multi-Segment

Use this script for long-form generation with explicit segment prompts.

```bash
python opensource/prism/scripts/run_sequential.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --prompts "A person waves." "A person walks." "A person bows." \
  --lengths 65 129 65 \
  --output_dir outputs/sequential
```

Key arguments:
- `--prompts`: required list of segment prompts.
- `--lengths`: optional per-segment frame counts; if omitted, all segments use `129`.
- `--guidance_scale`: classifier-free guidance scale.

### Narrative / Free-form Text

Use this script when you have a paragraph-like narrative instead of explicit segment list.

```bash
python opensource/prism/scripts/run_narrative.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --text "A person walks in, waves, sits down, then stands and bows." \
  --no_rewriter \
  --output_dir outputs/narrative
```

Notes:
- Current script supports `--no_rewriter` mode robustly: it splits `--text` by semicolon (`;`) into segments.
- If you do not use semicolons, the full text is treated as one segment.
- `--rewriter_url` path is reserved, but external rewriter client logic is not implemented in this open-source script.

## Citation

```bibtex
@article{ling2026prism,
  title={PRISM: Streaming Human Motion Generation with Per-Joint Latent Decomposition},
  author={Ling, Zeyu and Shuai, Qing and Zhang, Teng and Li, Shiyang and Han, Bo and Zou, Changqing},
  journal={arXiv preprint arXiv:2603.08590},
  year={2026},
  url={https://arxiv.org/abs/2603.08590}
}
```

## License

See the [versatilemotion](https://github.com/ZeyuLing/versatilemotion) repository for license terms.
