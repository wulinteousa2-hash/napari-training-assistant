from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from napari_training_assistant.io.loaders import ensure_numpy, load_image_any
from napari_training_assistant.unet_backend.augment import augment_2d_conservative, augment_3d_conservative
from napari_training_assistant.unet_backend.patching import (
    extract_patch_2d,
    extract_patch_3d,
    iter_patch_starts_2d,
    iter_patch_starts_3d,
    mask_has_signal,
)
from napari_training_assistant.utils.array_utils import normalize_zero_one, rgb_to_gray_if_needed


@dataclass
class PatchSample:
    image_path: str
    mask_path: str
    index: tuple[int, ...]


class PatchDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        mode_2d_or_3d: str,
        task_type: str,
        patch_xy: int,
        patch_z: int | None,
        overlap_percent: int,
        include_empty_mask: bool,
        augment: bool = False,
        seed: int = 123,
        cache_arrays: bool = True,
    ):
        if len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must have the same length")

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.mode_2d_or_3d = mode_2d_or_3d
        self.task_type = task_type
        self.patch_xy = int(patch_xy)
        self.patch_z = None if patch_z is None else int(patch_z)
        self.overlap_percent = int(overlap_percent)
        self.include_empty_mask = bool(include_empty_mask)
        self.augment = bool(augment)
        self.cache_arrays = bool(cache_arrays)
        self.rng = np.random.default_rng(seed)

        # RAM caches
        self._image_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}

        self.samples: list[PatchSample] = []
        self._build_index()

    def _prepare_pair(self, img, msk):
        img = ensure_numpy(img)
        msk = ensure_numpy(msk)

        img = rgb_to_gray_if_needed(img)
        img = normalize_zero_one(img).astype(np.float32, copy=False)

        if self.task_type == "binary":
            # collapse any nonzero label to foreground
            msk = (msk > 0).astype(np.float32, copy=False)
        else:
            # keep categorical integer labels for multiclass
            msk = msk.astype(np.int64, copy=False)

        return img, msk

    def _load_prepared_image(self, path: str) -> np.ndarray:
        if self.cache_arrays and path in self._image_cache:
            return self._image_cache[path]

        img = load_image_any(path)
        img = ensure_numpy(img)
        img = rgb_to_gray_if_needed(img)
        img = normalize_zero_one(img).astype(np.float32, copy=False)

        if self.cache_arrays:
            self._image_cache[path] = img
        return img

    def _load_prepared_mask(self, path: str) -> np.ndarray:
        if self.cache_arrays and path in self._mask_cache:
            return self._mask_cache[path]

        msk = load_image_any(path)
        msk = ensure_numpy(msk)

        if self.task_type == "binary":
            msk = (msk > 0).astype(np.float32, copy=False)
        else:
            msk = msk.astype(np.int64, copy=False)

        if self.cache_arrays:
            self._mask_cache[path] = msk
        return msk

    def _load_prepared_pair(self, img_path: str, msk_path: str):
        img = self._load_prepared_image(img_path)
        msk = self._load_prepared_mask(msk_path)
        return img, msk

    def _build_index(self):
        kept = 0
        skipped_dim = 0
        skipped_shape = 0
        skipped_empty = 0

        unique_images = set()
        unique_masks = set()

        for img_p, msk_p in zip(self.image_paths, self.mask_paths):
            img, msk = self._load_prepared_pair(img_p, msk_p)

            unique_images.add(img_p)
            unique_masks.add(msk_p)

            if self.mode_2d_or_3d == "2d":
                if img.ndim != 2 or msk.ndim != 2:
                    skipped_dim += 1
                    continue

                for y0, x0 in iter_patch_starts_2d(
                    img.shape,
                    self.patch_xy,
                    self.overlap_percent,
                ):
                    i_patch, m_patch = extract_patch_2d(
                        img, msk, y0, x0, self.patch_xy
                    )

                    if i_patch.shape != (self.patch_xy, self.patch_xy):
                        skipped_shape += 1
                        continue
                    if m_patch.shape != (self.patch_xy, self.patch_xy):
                        skipped_shape += 1
                        continue

                    if self.include_empty_mask or mask_has_signal(m_patch):
                        self.samples.append(PatchSample(img_p, msk_p, (y0, x0)))
                        kept += 1
                    else:
                        skipped_empty += 1

            elif self.mode_2d_or_3d == "3d":
                if img.ndim != 3 or msk.ndim != 3:
                    skipped_dim += 1
                    continue

                patch_z = self.patch_z if self.patch_z is not None else 16

                for z0, y0, x0 in iter_patch_starts_3d(
                    img.shape,
                    (patch_z, self.patch_xy, self.patch_xy),
                    self.overlap_percent,
                ):
                    i_patch, m_patch = extract_patch_3d(
                        img,
                        msk,
                        z0,
                        y0,
                        x0,
                        (patch_z, self.patch_xy, self.patch_xy),
                    )

                    if i_patch.shape != (patch_z, self.patch_xy, self.patch_xy):
                        skipped_shape += 1
                        continue
                    if m_patch.shape != (patch_z, self.patch_xy, self.patch_xy):
                        skipped_shape += 1
                        continue

                    if self.include_empty_mask or mask_has_signal(m_patch):
                        self.samples.append(PatchSample(img_p, msk_p, (z0, y0, x0)))
                        kept += 1
                    else:
                        skipped_empty += 1

            else:
                raise ValueError("mode_2d_or_3d must be '2d' or '3d'")

        print(f"[PatchDataset] total samples kept: {kept}")
        print(f"[PatchDataset] skipped_dim: {skipped_dim}")
        print(f"[PatchDataset] skipped_shape: {skipped_shape}")
        print(f"[PatchDataset] skipped_empty: {skipped_empty}")
        print(f"[PatchDataset] unique images cached: {len(unique_images) if self.cache_arrays else 0}")
        print(f"[PatchDataset] unique masks cached: {len(unique_masks) if self.cache_arrays else 0}")
        print(f"[PatchDataset] cache enabled: {self.cache_arrays}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        img, msk = self._load_prepared_pair(sample.image_path, sample.mask_path)

        if self.mode_2d_or_3d == "2d":
            y0, x0 = sample.index
            img, msk = extract_patch_2d(img, msk, y0, x0, self.patch_xy)

            if img.shape != (self.patch_xy, self.patch_xy):
                raise ValueError(
                    f"2D image patch has wrong shape {img.shape}, "
                    f"expected {(self.patch_xy, self.patch_xy)}"
                )
            if msk.shape != (self.patch_xy, self.patch_xy):
                raise ValueError(
                    f"2D mask patch has wrong shape {msk.shape}, "
                    f"expected {(self.patch_xy, self.patch_xy)}"
                )

            if self.augment:
                img, msk = augment_2d_conservative(img, msk, self.rng)

            img = img[None, ...]  # C,Y,X

            if self.task_type == "binary":
                msk = msk[None, ...].astype(np.float32, copy=False)  # C,Y,X
            else:
                msk = msk.astype(np.int64, copy=False)  # Y,X

        else:
            z0, y0, x0 = sample.index
            patch_z = self.patch_z if self.patch_z is not None else 16

            img, msk = extract_patch_3d(
                img,
                msk,
                z0,
                y0,
                x0,
                (patch_z, self.patch_xy, self.patch_xy),
            )

            if img.shape != (patch_z, self.patch_xy, self.patch_xy):
                raise ValueError(
                    f"3D image patch has wrong shape {img.shape}, "
                    f"expected {(patch_z, self.patch_xy, self.patch_xy)}"
                )
            if msk.shape != (patch_z, self.patch_xy, self.patch_xy):
                raise ValueError(
                    f"3D mask patch has wrong shape {msk.shape}, "
                    f"expected {(patch_z, self.patch_xy, self.patch_xy)}"
                )

            if self.augment:
                img, msk = augment_3d_conservative(img, msk, self.rng)

            img = img[None, ...]  # C,Z,Y,X

            if self.task_type == "binary":
                msk = msk[None, ...].astype(np.float32, copy=False)  # C,Z,Y,X
            else:
                msk = msk.astype(np.int64, copy=False)  # Z,Y,X

        return torch.from_numpy(img), torch.from_numpy(msk)
