from __future__ import annotations

from typing import Iterator

import numpy as np


def _normalize_overlap_percent(overlap_percent: int) -> int:
    return max(0, min(90, int(overlap_percent)))


def _step_from_overlap(size: int, overlap_percent: int) -> int:
    overlap_percent = _normalize_overlap_percent(overlap_percent)
    step = int(round(size * (1.0 - overlap_percent / 100.0)))
    return max(step, 1)


def _valid_starts_1d(length: int, patch_size: int, overlap_percent: int) -> list[int]:
    """
    Return boundary-aware start indices so every extracted patch is full-size.

    Example:
      length=1302, patch=512, overlap=0
      -> [0, 512, 790]

    The last start is shifted inward to guarantee a full patch ending at the boundary.
    """
    length = int(length)
    patch_size = int(patch_size)

    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if length < patch_size:
        return []

    step = _step_from_overlap(patch_size, overlap_percent)
    last_valid_start = length - patch_size

    starts = list(range(0, last_valid_start + 1, step))
    if not starts:
        starts = [0]

    if starts[-1] != last_valid_start:
        starts.append(last_valid_start)

    # Remove duplicates while preserving order
    out = []
    seen = set()
    for s in starts:
        s = int(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def iter_patch_starts_2d(
    shape_yx: tuple[int, int],
    patch_xy: int,
    overlap_percent: int,
) -> Iterator[tuple[int, int]]:
    """
    Yield only starts that produce full-size 2D patches of shape (patch_xy, patch_xy).
    """
    H, W = map(int, shape_yx)
    patch_xy = int(patch_xy)

    ys = _valid_starts_1d(H, patch_xy, overlap_percent)
    xs = _valid_starts_1d(W, patch_xy, overlap_percent)

    for y0 in ys:
        for x0 in xs:
            yield y0, x0


def iter_patch_starts_3d(
    shape_zyx: tuple[int, int, int],
    patch_zyx: tuple[int, int, int],
    overlap_percent: int,
) -> Iterator[tuple[int, int, int]]:
    """
    Yield only starts that produce full-size 3D patches of shape (pz, py, px).
    """
    Z, H, W = map(int, shape_zyx)
    pz, py, px = map(int, patch_zyx)

    zs = _valid_starts_1d(Z, pz, overlap_percent)
    ys = _valid_starts_1d(H, py, overlap_percent)
    xs = _valid_starts_1d(W, px, overlap_percent)

    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                yield z0, y0, x0


def extract_patch_2d(
    img: np.ndarray,
    msk: np.ndarray,
    y0: int,
    x0: int,
    patch_xy: int,
):
    patch_xy = int(patch_xy)
    i_patch = img[y0:y0 + patch_xy, x0:x0 + patch_xy]
    m_patch = msk[y0:y0 + patch_xy, x0:x0 + patch_xy]
    return i_patch, m_patch


def extract_patch_3d(
    img: np.ndarray,
    msk: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    patch_zyx: tuple[int, int, int],
):
    pz, py, px = map(int, patch_zyx)
    i_patch = img[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
    m_patch = msk[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
    return i_patch, m_patch


def mask_has_signal(mask: np.ndarray) -> bool:
    return np.any(mask > 0)