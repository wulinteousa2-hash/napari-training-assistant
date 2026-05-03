from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from napari_training_assistant.io.writers import save_csv_rows, save_json
from napari_training_assistant.unet_backend.config import RunConfig
from napari_training_assistant.unet_backend.datasets import PatchDataset
from napari_training_assistant.unet_backend.trainer import TrainConfig, train_model


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _project_path(project, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project.root / path


def _mode_from_architecture(architecture: dict[str, Any]) -> str:
    mode = str(architecture.get("spatial_dims", "2d")).lower()
    if mode not in {"2d", "3d"}:
        raise ValueError(f"Unsupported U-Net spatial_dims: {mode!r}")
    return mode


def _task_from_architecture(architecture: dict[str, Any]) -> str:
    task_type = str(architecture.get("output_mode", "binary")).lower()
    if task_type not in {"binary", "multiclass"}:
        raise ValueError(f"Unsupported U-Net output_mode: {task_type!r}")
    return task_type


def _checkpoint_by_id(project, checkpoint_id: str) -> dict[str, Any] | None:
    if not checkpoint_id:
        return None
    for checkpoint in project.checkpoint_entries():
        if checkpoint.get("checkpoint_id") == checkpoint_id:
            return checkpoint
    return None


def _starting_checkpoint(project, architecture: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    starting = project.load_starting_weights_config()
    mode = starting.get("mode", "latest_project_checkpoint")
    if mode == "scratch":
        return None, None
    if mode == "latest_project_checkpoint":
        candidate = project.latest_checkpoint()
    elif mode == "selected_project_checkpoint":
        candidate = _checkpoint_by_id(project, starting.get("checkpoint_id", ""))
        if candidate is None:
            raise ValueError("Selected checkpoint was not found in this project.")
    else:
        return None, None

    if candidate is None:
        return None, None

    compatible, reason = project.architecture_compatibility(
        architecture,
        candidate.get("architecture", {}),
    )
    if not compatible:
        raise ValueError(f"Cannot continue from incompatible checkpoint: {reason}")

    checkpoint_path = _project_path(project, candidate.get("checkpoint_path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Starting checkpoint file does not exist: {checkpoint_path}")
    return candidate, checkpoint_path


def build_run_config_from_project(project) -> RunConfig:
    project_config = project.load_config()
    architecture = project.active_task_config().get("architecture", project.load_architecture_config())
    mode = _mode_from_architecture(architecture)
    task_type = _task_from_architecture(architecture)
    run_number = len(project.checkpoint_entries()) + 1
    run_dir = project.active_task_root() / "history" / "unet_runs" / f"unet_run_{run_number:03d}_{_utc_slug()}"

    return RunConfig(
        mode_2d_or_3d=mode,
        task_type=task_type,
        model_name="unet2d" if mode == "2d" else "unet3d",
        in_channels=int(architecture.get("input_channels", 1)),
        out_channels=int(architecture.get("output_channels", 1)),
        patch_xy=int(project_config.get("patch_size", 256)),
        patch_z=None,
        overlap_percent=int(project_config.get("overlap_percent", 0)),
        include_empty_mask=bool(project_config.get("include_empty_mask", False)),
        batch_size=int(project_config.get("batch_size", 4)),
        epochs=int(project_config.get("epochs", 10)),
        learning_rate=float(project_config.get("learning_rate", 0.0001)),
        val_mode="split",
        val_split=float(project_config.get("validation_split", 0.2)),
        k_folds=5,
        use_gpu=torch.cuda.is_available(),
        image_dir=str(project.active_task_root() / "dataset" / "images"),
        mask_dir=str(project.active_task_root() / "dataset" / "masks"),
        output_dir=str(run_dir),
    )


def _selected_pairs(project, selected_pair_ids: list[str] | None) -> list[dict[str, Any]]:
    pairs = project.dataset_pairs()
    if selected_pair_ids is None:
        return pairs
    selected = set(selected_pair_ids)
    return [pair for pair in pairs if pair.get("pair_id") in selected]


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    header = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_dice",
        "val_dice",
        "train_iou",
        "val_iou",
        "train_f1",
        "val_f1",
    ]
    rows = [[row.get(name, "") for name in header] for row in history]
    save_csv_rows(path, header, rows)


def _best_row(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    return max(history, key=lambda row: float(row.get("val_dice", 0.0)))


def run_unet_training_for_project(
    project,
    *,
    selected_pair_ids: list[str] | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    architecture = project.active_task_config().get("architecture", project.load_architecture_config())
    run_config = build_run_config_from_project(project)
    pairs = _selected_pairs(project, selected_pair_ids)
    if not pairs:
        raise ValueError("No dataset pairs were selected for U-Net training.")

    starting_checkpoint, starting_checkpoint_path = _starting_checkpoint(project, architecture)
    run_dir = Path(run_config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [str(_project_path(project, pair["image_path"])) for pair in pairs]
    mask_paths = [str(_project_path(project, pair["mask_path"])) for pair in pairs]
    dataset = PatchDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        mode_2d_or_3d=run_config.mode_2d_or_3d,
        task_type=run_config.task_type,
        patch_xy=run_config.patch_xy,
        patch_z=run_config.patch_z,
        overlap_percent=run_config.overlap_percent,
        include_empty_mask=run_config.include_empty_mask,
        augment=any(project.load_config().get("augmentation_settings", {}).values()),
    )

    train_config = TrainConfig(
        mode_2d_or_3d=run_config.mode_2d_or_3d,
        task_type=run_config.task_type,
        in_channels=run_config.in_channels,
        out_channels=run_config.out_channels,
        batch_size=run_config.batch_size,
        epochs=run_config.epochs,
        lr=run_config.learning_rate,
        val_split=run_config.val_split,
        device="cuda" if run_config.use_gpu else "cpu",
        initial_checkpoint_path=str(starting_checkpoint_path or ""),
    )
    model, history = train_model(dataset, train_config, progress_cb=progress_cb)

    checkpoint_path = run_dir / "best_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    best = _best_row(history)
    summary = {
        "run_dir": str(run_dir),
        "number_of_images": len(pairs),
        "number_of_patches": len(dataset),
        "best_epoch": best.get("epoch", ""),
        "best_val_dice": best.get("val_dice", ""),
        "best_val_iou": best.get("val_iou", ""),
        "final_epoch": history[-1] if history else {},
        "starting_checkpoint_id": starting_checkpoint.get("checkpoint_id", "") if starting_checkpoint else "",
    }
    config_dict = run_config.to_dict()
    config_dict["train_config"] = asdict(train_config)
    config_dict["dataset_pair_ids"] = [pair["pair_id"] for pair in pairs]
    save_json(run_dir / "config.json", config_dict)
    save_json(run_dir / "summary.json", summary)
    _write_history_csv(run_dir / "history.csv", history)

    training_mode = "continue" if starting_checkpoint else "scratch"
    metadata = project.register_checkpoint(
        source_checkpoint_path=checkpoint_path,
        parent_checkpoint_id=summary["starting_checkpoint_id"],
        training_mode=training_mode,
        dataset_pair_ids=config_dict["dataset_pair_ids"],
        number_of_patches=len(dataset),
        train_validation_split=run_config.val_split,
        loss_metrics={
            "loss": float(best.get("train_loss", 0.0)) if best else 0.0,
            "val_loss": float(best.get("val_loss", 0.0)) if best else 0.0,
        },
        dice_iou_metrics={
            "dice": float(best.get("val_dice", 0.0)) if best else 0.0,
            "iou": float(best.get("val_iou", 0.0)) if best else 0.0,
        },
        benchmark_summary=(
            f"U-Net {run_config.mode_2d_or_3d}; {len(pairs)} images; "
            f"{len(dataset)} patches; best val dice {float(best.get('val_dice', 0.0)) if best else 0.0:.4f}"
        ),
        architecture=architecture,
        starting_weights=project.load_starting_weights_config(),
    )
    summary["checkpoint_id"] = metadata["checkpoint_id"]
    summary["checkpoint_path"] = metadata["checkpoint_path"]
    summary["dataset_pair_ids"] = config_dict["dataset_pair_ids"]
    save_json(run_dir / "summary.json", summary)
    return summary
