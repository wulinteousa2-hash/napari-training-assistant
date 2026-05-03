from __future__ import annotations

import numpy as np


def normalize_zero_one(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / max(mx - mn, eps)


def infer_spatial_ndim(x: np.ndarray) -> int:
    """
    Heuristic:
    - 2D grayscale: (Y, X)
    - 2D RGB: (Y, X, 3)
    - 3D grayscale: (Z, Y, X)
    """
    if x.ndim == 2:
        return 2
    if x.ndim == 3 and x.shape[-1] == 3:
        return 2
    if x.ndim == 3:
        return 3
    if x.ndim == 4 and x.shape[-1] == 3:
        return 3
    raise ValueError(f"Cannot infer 2D/3D spatial ndim from shape {x.shape}")


def rgb_to_gray_if_needed(x: np.ndarray) -> np.ndarray:
    if x.ndim >= 3 and x.shape[-1] == 3:
        return x.mean(axis=-1)
    return x


def ensure_mask_integer(mask: np.ndarray) -> np.ndarray:
    if np.issubdtype(mask.dtype, np.integer):
        return mask
    return mask.astype(np.int32)


def get_unique_labels(mask: np.ndarray, max_items: int = 50) -> list[int]:
    vals = np.unique(mask)
    return [int(v) for v in vals[:max_items]]