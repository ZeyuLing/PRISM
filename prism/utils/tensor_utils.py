# Standalone tensor_to_array utility (no mmotion dependency).
from typing import Any, Mapping
import numpy as np


def tensor_to_array(
    x: Any,
    *,
    copy: bool = True,
    allow_bfloat16: bool = False,
    on_quantized: str = "dequantize",
    on_sparse: str = "to_dense",
    dtype=None,
) -> Any:
    """Convert (possibly nested) PyTorch tensors to NumPy arrays."""
    try:
        import torch
    except Exception:
        return (
            np.asarray(x, dtype=dtype)
            if not isinstance(x, (dict, list, tuple))
            else x
        )

    def _one(t: Any):
        if isinstance(t, np.ndarray):
            arr = t.astype(dtype, copy=copy) if dtype is not None else (t.copy() if copy else t)
            return np.ascontiguousarray(arr)

        if isinstance(t, torch.Tensor):
            if getattr(t, "is_meta", False):
                raise ValueError("Cannot convert a meta tensor to NumPy.")
            if t.layout != torch.strided:
                if on_sparse == "to_dense":
                    t = t.to_dense()
                else:
                    raise ValueError(f"Cannot convert non-strided tensor with layout={t.layout}.")
            if getattr(t, "is_quantized", False):
                if on_quantized == "dequantize":
                    t = t.dequantize()
                else:
                    raise ValueError("Quantized tensors must be dequantized before NumPy conversion.")
            t = t.detach()
            if t.device.type != "cpu":
                t = t.to("cpu")
            if t.dtype == torch.bfloat16:
                if not (allow_bfloat16 and hasattr(np, "bfloat16")):
                    t = t.to(torch.float32)
            if not t.is_contiguous():
                t = t.contiguous()
            arr = t.numpy()
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            if copy:
                arr = arr.copy()
            return np.ascontiguousarray(arr)

        if np.isscalar(t):
            return np.array(t, dtype=dtype) if dtype is not None else np.array(t)

        if isinstance(t, Mapping):
            return {
                k: tensor_to_array(v, copy=copy, allow_bfloat16=allow_bfloat16,
                                   on_quantized=on_quantized, on_sparse=on_sparse, dtype=dtype)
                for k, v in t.items()
            }

        if isinstance(t, (list, tuple)):
            out = [
                tensor_to_array(v, copy=copy, allow_bfloat16=allow_bfloat16,
                                on_quantized=on_quantized, on_sparse=on_sparse, dtype=dtype)
                for v in t
            ]
            return type(t)(out)

        return np.asarray(t, dtype=dtype) if dtype is not None else np.asarray(t)

    return _one(x)
