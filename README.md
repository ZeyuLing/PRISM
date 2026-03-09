# PRISM: Text-to-Motion and Sequential Motion Generation

PRISM is a diffusion-based 3D human motion generation model supporting **text-to-motion (T2M)**, **pose-conditioned generation (TP2M)**, **sequential multi-segment generation**, and **narrative/free-form text** (with optional LLM decomposition). All tasks use a single checkpoint in HuggingFace-style layout.

## Demo Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=3PBFpYcwGIM">
    <img src="https://img.youtube.com/vi/3PBFpYcwGIM/maxresdefault.jpg" alt="PRISM Demo" width="640"/>
  </a>
</p>
<p align="center">
  <a href="https://www.youtube.com/watch?v=3PBFpYcwGIM">▶ Watch on YouTube</a>
</p>

## Links

| | |
|---|---|
| **Paper** | [GitHub](https://github.com/ZeyuLing/PRISM) (arXiv under review) |
| **Model** | [Hugging Face](https://huggingface.co/ZeyuLing/PRISM-TP2M-1.4B) |

---

## Directory layout

```
prism/
├── README.md                 # This file
├── requirements.txt         # Python dependencies for inference
├── pretrained_models/       # Download checkpoint here (see below)
├── prism/                   # Model & pipeline code
├── scripts/                 # Inference scripts
│   ├── run_t2m.py           # Text-to-motion
│   ├── run_tp2m.py          # Pose-conditioned (TP2M)
│   ├── run_sequential.py    # Multi-segment
│   └── run_narrative.py     # Free-form text (optional LLM rewrite)
└── ...
```

---

## 1. Download checkpoint

Pretrained weights are hosted on Hugging Face. Download to `pretrained_models/prism_1.4b`:

```bash
pip install huggingface_hub
huggingface-cli download ZeyuLing/PRISM-TP2M-1.4B --local-dir pretrained_models/prism_1.4b
```

Or from Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("ZeyuLing/PRISM-TP2M-1.4B", local_dir="pretrained_models/prism_1.4b")
```

---

## 2. Dependencies

- Python ≥3.9
- PyTorch (CUDA recommended)
- transformers, diffusers, einops, mmengine
- SMPL/SMPL-X body model (for full mesh rendering)

This repo is designed to run inside the [versatilemotion](https://github.com/ZeyuLing/versatilemotion) main repository. Install dependencies from the main repo, then run scripts from the main repo root:

```bash
cd /path/to/versatilemotion
pip install -r requirements.txt
```

---

## 3. Running inference

All scripts assume you are in the **versatilemotion repository root** and that the checkpoint is at `opensource/prism/pretrained_models/prism_1.4b`.

### Text-to-motion (T2M)

```bash
python opensource/prism/scripts/run_t2m.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --prompt "A person walks forward and waves." \
  --output_dir outputs/t2m
```

### Pose-conditioned (TP2M)

```bash
python opensource/prism/scripts/run_tp2m.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --first_frame_pose tpose \
  --prompt "The person begins to walk forward slowly and naturally." \
  --output_dir outputs/tp2m
```

Preset poses: `standing`, `tpose`, `squat`, `kneel`, `sit`.

### Sequential multi-segment

```bash
python opensource/prism/scripts/run_sequential.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --prompts "A person waves." "A person walks." "A person bows." \
  --lengths 65 129 65 \
  --output_dir outputs/sequential
```

### Narrative / free-form text

```bash
python opensource/prism/scripts/run_narrative.py \
  --checkpoint opensource/prism/pretrained_models/prism_1.4b \
  --text "A person walks in, waves, sits down, then stands and bows." \
  --no_rewriter \
  --output_dir outputs/narrative
```

---

## 4. Citation

```bibtex
@inproceedings{prism2026,
  title={PRISM: Streaming Human Motion Generation with Per-Joint Latent Decomposition},
  booktitle={ECCV},
  year={2026},
}
```

---

## 5. License

See the [versatilemotion](https://github.com/ZeyuLing/versatilemotion) repository for license terms.
