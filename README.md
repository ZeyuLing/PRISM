# PRISM: Streaming Human Motion Generation with Per-Joint Latent Decomposition

<p align="center">
  <a href="https://github.com/ZeyuLing/PRISM">
    <img src="https://img.shields.io/badge/Paper-ArXiv_(under_review)-B31B1B?style=for-the-badge&logo=arxiv" alt="Paper"/>
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

PRISM is a unified diffusion framework for 3D human motion generation, supporting **text-to-motion (T2M)**, **pose-conditioned generation (TP2M)**, and **long-horizon sequential generation** in a single model.

## Abstract

We present PRISM, a streaming human motion generation framework built on per-joint latent decomposition. PRISM factorizes motion representation into semantically meaningful joint-wise latent tokens, which enables stable autoregressive generation over long horizons while preserving local motion detail. With one unified model, PRISM supports text-driven motion synthesis, pose-conditioned continuation, and multi-segment narrative composition. Experiments show strong generation quality, temporal consistency, and robustness for long sequences in both quantitative benchmarks and qualitative demos.

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

### Text-to-Motion

```bash
python opensource/prism/scripts/run_t2m.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --prompt "A person walks forward and waves." \
  --output_dir outputs/t2m
```

### Pose-Conditioned Generation

```bash
python opensource/prism/scripts/run_tp2m.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --first_frame_pose tpose \
  --prompt "The person begins to walk forward slowly and naturally." \
  --output_dir outputs/tp2m
```

Preset poses: `standing`, `tpose`, `squat`, `kneel`, `sit`.

### Sequential Multi-Segment

```bash
python opensource/prism/scripts/run_sequential.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --prompts "A person waves." "A person walks." "A person bows." \
  --lengths 65 129 65 \
  --output_dir outputs/sequential
```

### Narrative / Free-form Text

```bash
python opensource/prism/scripts/run_narrative.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --text "A person walks in, waves, sits down, then stands and bows." \
  --no_rewriter \
  --output_dir outputs/narrative
```

## Citation

```bibtex
@article{ling2026prism,
  title={PRISM: Streaming Human Motion Generation with Per-Joint Latent Decomposition},
  author={Ling, Zeyu and Shuai, Qing and Zhang, Teng and Li, Shiyang and Han, Bo and Zou, Changqing},
  journal={arXiv preprint},
  year={2026}
}
```

## License

See the [versatilemotion](https://github.com/ZeyuLing/versatilemotion) repository for license terms.
