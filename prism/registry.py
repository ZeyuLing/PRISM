# Prism-local registry, mirrors the subset of mmotion.registry used by prism.
# Uses mmengine's Registry infra but is self-contained (no mmotion dependency).

from typing import Any, Dict, Optional, Union
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

__all__ = ["MODELS", "HF_MODELS"]

# ── MODELS: standard nn.Module registry ──────────────────────────────────
MODELS = Registry(
    "model",
    parent=MMENGINE_MODELS,
    locations=["prism.models"],
)


# ── HF_MODELS builder (diffusers / transformers classes) ─────────────────
def build_hf_model_from_cfg(
    cfg: Dict[str, Any],
    registry: Registry,
    default_args: Optional[Dict[str, Any]] = None,
):
    if not isinstance(cfg, dict):
        raise TypeError(f"`cfg` must be a dict, got {type(cfg)}")
    args = dict(cfg)
    if default_args:
        for k, v in default_args.items():
            args.setdefault(k, v)
    if "type" not in args:
        raise KeyError("`cfg` must contain the key 'type'")
    obj_type = args.pop("type")

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"'{obj_type}' is not registered in {registry.name}")
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"`type` must be a string or callable, got {type(obj_type)}")

    if "from_pretrained" in args:
        fp_args = args.pop("from_pretrained")
        assert isinstance(fp_args, dict)
        assert "pretrained_model_name_or_path" in fp_args
        fp_args = {**args, **fp_args}
        return obj_cls.from_pretrained(**fp_args)

    if "from_single_file" in args:
        fs_args = args.pop("from_single_file")
        assert isinstance(fs_args, dict)
        assert "pretrained_model_link_or_path" in fs_args
        fs_args = {**args, **fs_args}
        return obj_cls.from_single_file(**fs_args)

    if "from_config" in args:
        fc_args = args.pop("from_config")
        assert isinstance(fc_args, dict)
        assert "config" in fc_args
        fc_args = {**args, **fc_args}
        return obj_cls.from_config(fc_args)

    return obj_cls(**args)


HF_MODELS = Registry(
    "hf_model",
    build_hf_model_from_cfg,
    locations=["prism.models", "prism.pipelines"],
)
