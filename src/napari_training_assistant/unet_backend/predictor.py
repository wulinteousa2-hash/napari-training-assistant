from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from napari_training_assistant.io.loaders import ensure_numpy, load_image_any
from napari_training_assistant.io.writers import load_json, save_tiff
from napari_training_assistant.unet_backend.models import UNet2D, UNet3D
from napari_training_assistant.utils.array_utils import normalize_zero_one, rgb_to_gray_if_needed


def load_run_metadata(run_dir: str | Path) -> tuple[dict, dict]:
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return cfg, summary


def load_model_from_run_folder(run_dir: str | Path, device: str | None = None):
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")

    mode = cfg["mode_2d_or_3d"]
    out_channels = int(cfg["out_channels"])

    if mode == "2d":
        model = UNet2D(in_channels=int(cfg.get("in_channels", 1)), out_channels=out_channels)
    else:
        model = UNet3D(in_channels=int(cfg.get("in_channels", 1)), out_channels=out_channels)

    ckpt = torch.load(run_dir / "best_model.pt", map_location=device or "cpu")
    model.load_state_dict(ckpt)
    model.eval()

    if device is not None:
        model = model.to(device)

    return model, cfg


@torch.no_grad()
def _predict_full_2d(model, image: np.ndarray, task_type: str, device: str):
    x = rgb_to_gray_if_needed(image)
    x = normalize_zero_one(x).astype(np.float32)
    x = torch.from_numpy(x[None, None, ...]).to(device)

    logits = model(x)
    if task_type == "binary":
        pred = (torch.sigmoid(logits) >= 0.5).float()[0, 0].cpu().numpy().astype(np.uint8)
        return pred
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


@torch.no_grad()
def _predict_tiled_2d(model, image: np.ndarray, task_type: str, device: str, patch_xy: int):
    x = rgb_to_gray_if_needed(image)
    x = normalize_zero_one(x).astype(np.float32)

    H, W = x.shape
    p = int(patch_xy)

    if task_type == "binary":
        accum = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)
    else:
        out_channels = model.head.out_channels
        accum = np.zeros((out_channels, H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - p + 1, 1), p))
    xs = list(range(0, max(W - p + 1, 1), p))
    if len(ys) == 0 or ys[-1] != max(H - p, 0):
        ys.append(max(H - p, 0))
    if len(xs) == 0 or xs[-1] != max(W - p, 0):
        xs.append(max(W - p, 0))

    seen = set()
    for y0 in ys:
        for x0 in xs:
            key = (y0, x0)
            if key in seen:
                continue
            seen.add(key)

            patch = x[y0:y0 + p, x0:x0 + p]
            t = torch.from_numpy(patch[None, None, ...]).to(device)
            logits = model(t)

            if task_type == "binary":
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
                accum[y0:y0 + p, x0:x0 + p] += probs
            else:
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                accum[:, y0:y0 + p, x0:x0 + p] += probs

            count[y0:y0 + p, x0:x0 + p] += 1.0

    if task_type == "binary":
        probs = accum / np.maximum(count, 1e-8)
        return (probs >= 0.5).astype(np.uint8)

    probs = accum / np.maximum(count[None, ...], 1e-8)
    return np.argmax(probs, axis=0).astype(np.uint8)


@torch.no_grad()
def _predict_full_3d(model, image: np.ndarray, task_type: str, device: str):
    x = normalize_zero_one(image.astype(np.float32))
    x = torch.from_numpy(x[None, None, ...]).to(device)

    logits = model(x)
    if task_type == "binary":
        pred = (torch.sigmoid(logits) >= 0.5).float()[0, 0].cpu().numpy().astype(np.uint8)
        return pred
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


@torch.no_grad()
def _predict_tiled_3d(model, image: np.ndarray, task_type: str, device: str, patch_z: int, patch_xy: int):
    x = normalize_zero_one(image.astype(np.float32))
    Z, Y, X = x.shape

    if task_type == "binary":
        accum = np.zeros((Z, Y, X), dtype=np.float32)
        count = np.zeros((Z, Y, X), dtype=np.float32)
    else:
        out_channels = model.head.out_channels
        accum = np.zeros((out_channels, Z, Y, X), dtype=np.float32)
        count = np.zeros((Z, Y, X), dtype=np.float32)

    step_z = int(patch_z)
    step_xy = int(patch_xy)

    zs = list(range(0, max(Z - patch_z + 1, 1), step_z))
    ys = list(range(0, max(Y - patch_xy + 1, 1), step_xy))
    xs = list(range(0, max(X - patch_xy + 1, 1), step_xy))

    if len(zs) == 0 or zs[-1] != max(Z - patch_z, 0):
        zs.append(max(Z - patch_z, 0))
    if len(ys) == 0 or ys[-1] != max(Y - patch_xy, 0):
        ys.append(max(Y - patch_xy, 0))
    if len(xs) == 0 or xs[-1] != max(X - patch_xy, 0):
        xs.append(max(X - patch_xy, 0))

    seen = set()
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                key = (z0, y0, x0)
                if key in seen:
                    continue
                seen.add(key)

                patch = x[z0:z0 + patch_z, y0:y0 + patch_xy, x0:x0 + patch_xy]
                t = torch.from_numpy(patch[None, None, ...]).to(device)
                logits = model(t)

                if task_type == "binary":
                    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
                    accum[z0:z0 + patch_z, y0:y0 + patch_xy, x0:x0 + patch_xy] += probs
                else:
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    accum[:, z0:z0 + patch_z, y0:y0 + patch_xy, x0:x0 + patch_xy] += probs

                count[z0:z0 + patch_z, y0:y0 + patch_xy, x0:x0 + patch_xy] += 1.0

    if task_type == "binary":
        probs = accum / np.maximum(count, 1e-8)
        return (probs >= 0.5).astype(np.uint8)

    probs = accum / np.maximum(count[None, ...], 1e-8)
    return np.argmax(probs, axis=0).astype(np.uint8)


def _auto_strategy(mode: str, image: np.ndarray) -> str:
    if mode == "2d":
        return "full"
    # conservative for 3D
    voxels = int(np.prod(image.shape))
    return "full" if voxels <= 32_000_000 else "tiled"


def _predict_with_model(model, cfg: dict, image: np.ndarray, device: str, strategy: str = "auto"):
    mode = cfg["mode_2d_or_3d"]
    task_type = cfg["task_type"]
    image = ensure_numpy(image)
    image = rgb_to_gray_if_needed(image)

    if mode == "2d":
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image for 2D model, got shape {image.shape}")

        used_strategy = _auto_strategy(mode, image) if strategy == "auto" else strategy
        if used_strategy == "tiled":
            pred = _predict_tiled_2d(
                model, image, task_type, device,
                patch_xy=int(cfg["patch_xy"]),
            )
        else:
            pred = _predict_full_2d(model, image, task_type, device)
        return pred, used_strategy

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image for 3D model, got shape {image.shape}")

    used_strategy = _auto_strategy(mode, image) if strategy == "auto" else strategy
    if used_strategy == "full":
        pred = _predict_full_3d(model, image, task_type, device)
    else:
        pred = _predict_tiled_3d(
            model, image, task_type, device,
            patch_z=int(cfg["patch_z"]),
            patch_xy=int(cfg["patch_xy"]),
        )
    return pred, used_strategy


def predict_single_from_run_folder(run_dir: str | Path, image_path: str | Path, device: str, strategy: str = "auto"):
    model, cfg = load_model_from_run_folder(run_dir, device=device)
    image = ensure_numpy(load_image_any(image_path))
    pred, used_strategy = _predict_with_model(model, cfg, image, device=device, strategy=strategy)
    return pred, cfg, used_strategy


def predict_folder_from_run_folder(
    run_dir: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    device: str,
    strategy: str = "auto",
    overwrite: bool = False,
    progress_cb=None,
):
    run_dir = Path(run_dir)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model_from_run_folder(run_dir, device=device)

    tif_files = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ])

    report = []
    total = len(tif_files)

    for i, path in enumerate(tif_files, start=1):
        out_path = output_dir / f"{path.stem}_pred.tif"

        if out_path.exists() and not overwrite:
            status = "skipped_exists"
            report.append({
                "input": str(path),
                "output": str(out_path),
                "status": status,
                "strategy": "",
            })
            if progress_cb is not None:
                progress_cb(i, total, str(path), status)
            continue

        try:
            image = ensure_numpy(load_image_any(path))
            pred, used_strategy = _predict_with_model(model, cfg, image, device=device, strategy=strategy)
            save_tiff(out_path, pred)
            status = "ok"
        except Exception as e:
            used_strategy = ""
            status = f"error: {e}"

        report.append({
            "input": str(path),
            "output": str(out_path),
            "status": status,
            "strategy": used_strategy,
        })

        if progress_cb is not None:
            progress_cb(i, total, str(path), status)

    return report, cfg
