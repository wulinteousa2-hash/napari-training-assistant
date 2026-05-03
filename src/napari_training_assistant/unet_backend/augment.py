from __future__ import annotations

import numpy as np


def random_flip_2d(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        img = np.flip(img, axis=-2)
        msk = np.flip(msk, axis=-2)
    if rng.random() < 0.5:
        img = np.flip(img, axis=-1)
        msk = np.flip(msk, axis=-1)
    return img.copy(), msk.copy()


def random_rot90_2d(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    k = int(rng.integers(0, 4))
    img = np.rot90(img, k=k, axes=(-2, -1))
    msk = np.rot90(msk, k=k, axes=(-2, -1))
    return img.copy(), msk.copy()


def random_intensity_jitter(img: np.ndarray, rng: np.random.Generator, scale_range=(0.9, 1.1), bias_range=(-0.05, 0.05)):
    s = rng.uniform(*scale_range)
    b = rng.uniform(*bias_range)
    out = img * s + b
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def augment_2d_conservative(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    img, msk = random_flip_2d(img, msk, rng)
    img, msk = random_rot90_2d(img, msk, rng)
    img = random_intensity_jitter(img, rng)
    return img, msk


def augment_3d_conservative(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    # Conservative: only Y/X flips plus mild intensity jitter.
    if rng.random() < 0.5:
        img = np.flip(img, axis=-2)
        msk = np.flip(msk, axis=-2)
    if rng.random() < 0.5:
        img = np.flip(img, axis=-1)
        msk = np.flip(msk, axis=-1)
    img = random_intensity_jitter(img, rng)
    return img.copy(), msk.copy()