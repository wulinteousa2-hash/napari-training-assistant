from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from napari_training_assistant.unet_backend.losses import BinaryBCEDiceLoss, MulticlassDiceCELoss
from napari_training_assistant.unet_backend.metrics import (
    binary_metrics_from_logits,
    multiclass_mean_metrics_from_logits,
)
from napari_training_assistant.unet_backend.models import UNet2D, UNet3D


@dataclass
class TrainConfig:
    mode_2d_or_3d: str = "2d"
    task_type: str = "binary"
    in_channels: int = 1
    out_channels: int = 1
    batch_size: int = 2
    epochs: int = 5
    lr: float = 1e-3
    val_split: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    initial_checkpoint_path: str = ""


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.mode_2d_or_3d == "2d":
        return UNet2D(cfg.in_channels, cfg.out_channels)
    if cfg.mode_2d_or_3d == "3d":
        return UNet3D(cfg.in_channels, cfg.out_channels)
    raise ValueError("mode_2d_or_3d must be '2d' or '3d'")


def build_loss(cfg: TrainConfig):
    if cfg.task_type == "binary":
        return BinaryBCEDiceLoss()
    return MulticlassDiceCELoss(num_classes=cfg.out_channels)


def _run_epoch(model, loader, loss_fn, optimizer, cfg, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_metrics = []

    for x, y in loader:
        x = x.to(cfg.device)

        if cfg.task_type == "binary":
            y = y.to(cfg.device).float()
        else:
            # y may come in with shape (B, Z, Y, X) or (B, Y, X)
            y = y.to(cfg.device).long()

        with torch.set_grad_enabled(train):
            logits = model(x)

            if cfg.task_type == "binary":
                loss = loss_fn(logits, y)
                metrics = binary_metrics_from_logits(logits.detach(), y.detach())
            else:
                if y.ndim == logits.ndim:
                    y = y.squeeze(1)
                loss = loss_fn(logits, y)
                metrics = multiclass_mean_metrics_from_logits(logits.detach(), y.detach(), cfg.out_channels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        all_metrics.append(metrics)

    n = max(len(loader), 1)
    mean_loss = total_loss / n
    mean_metrics = {
        "dice": sum(m["dice"] for m in all_metrics) / max(len(all_metrics), 1),
        "iou": sum(m["iou"] for m in all_metrics) / max(len(all_metrics), 1),
        "f1": sum(m["f1"] for m in all_metrics) / max(len(all_metrics), 1),
    }
    return mean_loss, mean_metrics


def train_model(dataset, cfg: TrainConfig, progress_cb=None):
    n_total = len(dataset)
    if n_total <= 0:
        raise ValueError(
            "PatchDataset contains 0 samples. "
            "Check image/mask shapes, patch size, overlap, and empty-mask filtering."
        )

    n_val = max(1, int(round(n_total * cfg.val_split)))
    n_train = max(1, n_total - n_val)

    if n_train + n_val > n_total:
        n_val = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg).to(cfg.device)
    if cfg.initial_checkpoint_path:
        checkpoint_path = Path(cfg.initial_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Starting checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=cfg.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    best_val_dice = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        train_loss, train_metrics = _run_epoch(model, train_loader, loss_fn, optimizer, cfg, train=True)
        val_loss, val_metrics = _run_epoch(model, val_loader, loss_fn, optimizer, cfg, train=False)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_metrics["dice"],
            "val_dice": val_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if progress_cb is not None:
            progress_cb(epoch + 1, cfg.epochs, row)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
