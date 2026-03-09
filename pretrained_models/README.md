# Pretrained Models

Pretrained weights are not included in this repository. Download from Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli download ZeyuLing/PRISM-TP2M-1.4B --local-dir pretrained_models/prism_1.4b
```

Or from Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("ZeyuLing/PRISM-TP2M-1.4B", local_dir="pretrained_models/prism_1.4b")
```

After download, the checkpoint will be at `pretrained_models/prism_1.4b/` with the HuggingFace-style layout (config.json, transformer/, text_encoder/, vae/).
