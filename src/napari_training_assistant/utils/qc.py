from __future__ import annotations

from typing import Any

import numpy as np

from napari_training_assistant.utils.array_utils import get_unique_labels


def summarize_array(name: str, arr: Any) -> str:
    a = np.asarray(arr)
    lines = [
        f"Name: {name}",
        f"Shape: {a.shape}",
        f"Dtype: {a.dtype}",
        f"Min: {np.nanmin(a)}",
        f"Max: {np.nanmax(a)}",
    ]
    return "\n".join(lines)


def summarize_mask(name: str, arr: Any) -> str:
    a = np.asarray(arr)
    labels = get_unique_labels(a)
    lines = [
        f"Mask name: {name}",
        f"Shape: {a.shape}",
        f"Dtype: {a.dtype}",
        f"Unique labels: {labels}",
        f"Background zero present: {0 in labels}",
    ]
    return "\n".join(lines)


def validate_image_mask_pair(image: Any, mask: Any) -> tuple[bool, str]:
    img = np.asarray(image)
    msk = np.asarray(mask)

    if img.ndim >= 3 and img.shape[-1] == 3:
        img_spatial = img.shape[:-1]
    else:
        img_spatial = img.shape

    ok = img_spatial == msk.shape
    msg = (
        f"Image spatial shape: {img_spatial}\n"
        f"Mask shape: {msk.shape}\n"
        f"Pair valid: {ok}"
    )
    return ok, msg
