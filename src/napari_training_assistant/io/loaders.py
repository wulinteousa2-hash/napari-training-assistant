from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import tifffile
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from PIL import Image


def load_tiff_any(path: str | Path) -> np.ndarray:
    path = str(path)
    return tifffile.imread(path)


def load_pillow_any(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def load_omezarr_any(path: str | Path) -> Any:
    """
    Load OME-Zarr using ome-zarr-py.
    Returns the first image pyramid level if found.
    This may be a dask-backed array depending on store/content.
    """
    loc = parse_url(str(path))
    if loc is None:
        raise ValueError(f"Could not parse OME-Zarr path: {path}")

    reader = Reader(loc)
    nodes = list(reader())
    if not nodes:
        raise ValueError(f"No readable nodes found in OME-Zarr: {path}")

    for node in nodes:
        data = node.data
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # first multiscale level
        if data is not None:
            return data

    raise ValueError(f"No image array found in OME-Zarr: {path}")


def load_image_any(path: str | Path) -> Any:
    p = str(path).lower()
    if p.endswith(".tif") or p.endswith(".tiff"):
        return load_tiff_any(path)
    if p.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return load_pillow_any(path)
    if p.endswith(".zarr"):
        return load_omezarr_any(path)
    raise ValueError(f"Unsupported image format: {path}")


def ensure_numpy(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    if isinstance(a, zarr.Array):
        return np.asarray(a)
    if isinstance(a, da.Array):
        return a.compute()
    return np.asarray(a)
