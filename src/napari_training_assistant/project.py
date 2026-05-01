"""Persistent training project storage."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import tifffile

from napari_training_assistant import __version__
from napari_training_assistant.unet import describe_basic_unet


PROJECT_CONFIG = "project_config.json"
DATASET_MANIFEST = Path("dataset") / "manifest.json"
CHECKPOINTS_JSON = Path("checkpoints") / "checkpoints.json"
TRAINING_RUNS_JSON = Path("history") / "training_runs.json"
BENCHMARK_HISTORY_CSV = Path("history") / "benchmark_history.csv"
ARCHITECTURE_CONFIG = Path("architecture") / "architecture_config.json"
STARTING_WEIGHTS_CONFIG = Path("models") / "starting_weights_config.json"
MODEL_REGISTRY = Path("models") / "model_registry.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_augmentation_settings() -> dict[str, Any]:
    return {
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotation_degrees": [0, 90, 180, 270],
        "intensity_jitter": False,
    }


def default_architecture_config() -> dict[str, Any]:
    return {
        "backend": "basic_unet",
        "spatial_dims": "2d",
        "preset": "standard_unet",
        "depth": 4,
        "base_channels": 32,
        "normalization": "batch",
        "upsampling": "transpose",
        "input_channels": 1,
        "output_mode": "binary",
        "num_classes": 2,
        "output_channels": 1,
        "class_labels": {"0": "background", "1": "foreground"},
        "activation": "sigmoid",
        "loss": "bce_dice",
        "threshold": 0.5,
    }


def default_starting_weights_config() -> dict[str, Any]:
    return {
        "mode": "latest_project_checkpoint",
        "checkpoint_id": "",
        "imported_model_id": "",
        "imported_model_path": "",
        "compatibility_status": "valid",
        "compatibility_message": "No external starting weights selected.",
    }


def default_mask_preparation_config() -> dict[str, Any]:
    return {
        "mode": "merge_nonzero_to_foreground",
        "target_class_name": "foreground",
        "manual_label_map": {},
        "strict_multiclass_validation": True,
    }


def default_project_config(project_path: Path) -> dict[str, Any]:
    now = utc_now()
    return {
        "project_name": project_path.name,
        "created_at": now,
        "last_opened_at": now,
        "plugin_version": __version__,
        "default_training_mode": "Continue from latest checkpoint",
        "default_dataset_source": "All accepted masks",
        "patch_size": 256,
        "batch_size": 4,
        "epochs": 10,
        "learning_rate": 0.0001,
        "validation_split": 0.2,
        "binary_or_multiclass": "binary",
        "input_channels": 1,
        "normalization_mode": "percentile",
        "augmentation_settings": default_augmentation_settings(),
        "default_architecture_config_path": str(ARCHITECTURE_CONFIG),
        "default_starting_weights_config_path": str(STARTING_WEIGHTS_CONFIG),
        "default_model_backend": "basic_unet",
        "default_starting_model_mode": "latest_project_checkpoint",
        "mask_preparation": default_mask_preparation_config(),
        "latest_checkpoint_id": "",
        "latest_checkpoint_path": "",
        "latest_dataset_manifest_path": str(DATASET_MANIFEST),
        "last_selected_image_layer": "",
        "last_selected_mask_layer": "",
        "notes": "",
        "auto_configuration_decisions": [],
    }


@dataclass
class ProjectState:
    status: str
    missing_paths: list[str] = field(default_factory=list)
    incomplete_reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.status == "valid":
            return "valid"
        if self.status == "missing":
            return "missing"
        return "incomplete"


class TrainingProject:
    """Read/write API for a training project folder."""

    required_dirs = (
        Path("dataset") / "images",
        Path("dataset") / "masks",
        Path("checkpoints"),
        Path("predictions"),
        Path("history"),
        Path("logs"),
        Path("architecture"),
        Path("models") / "imported",
    )

    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def create_or_open(cls, root: str | Path) -> "TrainingProject":
        project = cls(Path(root).expanduser().resolve())
        project.ensure_structure()
        config = project.load_config() if project.config_path.exists() else {}
        if not config:
            config = default_project_config(project.root)
        else:
            config = {**default_project_config(project.root), **config}
            config["last_opened_at"] = utc_now()
            config["plugin_version"] = __version__
        project.save_config(config)
        project.ensure_manifest()
        project.ensure_checkpoint_index()
        project.ensure_training_history()
        project.ensure_benchmark_history()
        project.ensure_architecture_config()
        project.ensure_starting_weights_config()
        project.ensure_model_registry()
        return project

    @property
    def config_path(self) -> Path:
        return self.root / PROJECT_CONFIG

    @property
    def manifest_path(self) -> Path:
        return self.root / DATASET_MANIFEST

    @property
    def checkpoints_path(self) -> Path:
        return self.root / CHECKPOINTS_JSON

    @property
    def training_runs_path(self) -> Path:
        return self.root / TRAINING_RUNS_JSON

    @property
    def benchmark_history_path(self) -> Path:
        return self.root / BENCHMARK_HISTORY_CSV

    @property
    def architecture_config_path(self) -> Path:
        return self.root / ARCHITECTURE_CONFIG

    @property
    def starting_weights_config_path(self) -> Path:
        return self.root / STARTING_WEIGHTS_CONFIG

    @property
    def model_registry_path(self) -> Path:
        return self.root / MODEL_REGISTRY

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.root / "checkpoints" / "latest.pt"

    def ensure_structure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for relative_dir in self.required_dirs:
            (self.root / relative_dir).mkdir(parents=True, exist_ok=True)

    def ensure_manifest(self) -> None:
        if not self.manifest_path.exists():
            self.write_json(
                self.manifest_path,
                {"version": 1, "created_at": utc_now(), "updated_at": utc_now(), "pairs": []},
            )

    def ensure_checkpoint_index(self) -> None:
        if not self.checkpoints_path.exists():
            self.write_json(
                self.checkpoints_path,
                {"version": 1, "created_at": utc_now(), "updated_at": utc_now(), "checkpoints": []},
            )

    def ensure_training_history(self) -> None:
        if not self.training_runs_path.exists():
            self.write_json(
                self.training_runs_path,
                {"version": 1, "created_at": utc_now(), "updated_at": utc_now(), "runs": []},
            )

    def ensure_benchmark_history(self) -> None:
        if not self.benchmark_history_path.exists():
            with self.benchmark_history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "checkpoint_id",
                        "loss",
                        "val_loss",
                        "dice",
                        "iou",
                        "summary",
                    ],
                )
                writer.writeheader()

    def ensure_architecture_config(self) -> None:
        if not self.architecture_config_path.exists():
            self.write_json(self.architecture_config_path, default_architecture_config())

    def ensure_starting_weights_config(self) -> None:
        if not self.starting_weights_config_path.exists():
            self.write_json(self.starting_weights_config_path, default_starting_weights_config())

    def ensure_model_registry(self) -> None:
        if not self.model_registry_path.exists():
            self.write_json(
                self.model_registry_path,
                {"version": 1, "created_at": utc_now(), "updated_at": utc_now(), "models": []},
            )

    def validate(self) -> ProjectState:
        if not self.root.exists():
            return ProjectState("missing", [str(self.root)])
        missing = []
        for relative in (
            *self.required_dirs,
            Path(PROJECT_CONFIG),
            DATASET_MANIFEST,
            CHECKPOINTS_JSON,
            ARCHITECTURE_CONFIG,
            STARTING_WEIGHTS_CONFIG,
            MODEL_REGISTRY,
        ):
            if not (self.root / relative).exists():
                missing.append(str(relative))
        if missing:
            return ProjectState("incomplete", incomplete_reasons=[f"Missing {path}" for path in missing])
        return ProjectState("valid")

    def load_config(self) -> dict[str, Any]:
        return self.read_json(self.config_path)

    def save_config(self, config: dict[str, Any]) -> None:
        config["latest_dataset_manifest_path"] = str(DATASET_MANIFEST)
        config["default_architecture_config_path"] = str(ARCHITECTURE_CONFIG)
        config["default_starting_weights_config_path"] = str(STARTING_WEIGHTS_CONFIG)
        self.write_json(self.config_path, config)

    def update_settings(self, **settings: Any) -> dict[str, Any]:
        config = self.load_config()
        config.update(settings)
        self.save_config(config)
        return config

    def load_architecture_config(self) -> dict[str, Any]:
        self.ensure_architecture_config()
        return self.read_json(self.architecture_config_path)

    def save_architecture_config(self, architecture: dict[str, Any]) -> dict[str, Any]:
        architecture = {**default_architecture_config(), **architecture}
        architecture["output_channels"] = self.output_channels_for_architecture(architecture)
        architecture["activation"] = "sigmoid" if architecture["output_mode"] == "binary" else "softmax"
        architecture["loss"] = "bce_dice" if architecture["output_mode"] == "binary" else "cross_entropy_dice"
        self.write_json(self.architecture_config_path, architecture)
        config = self.load_config()
        config["default_model_backend"] = architecture["backend"]
        config["binary_or_multiclass"] = architecture["output_mode"]
        config["input_channels"] = architecture["input_channels"]
        self.save_config(config)
        return architecture

    def load_starting_weights_config(self) -> dict[str, Any]:
        self.ensure_starting_weights_config()
        return self.read_json(self.starting_weights_config_path)

    def save_starting_weights_config(self, starting_weights: dict[str, Any]) -> dict[str, Any]:
        starting_weights = {**default_starting_weights_config(), **starting_weights}
        self.write_json(self.starting_weights_config_path, starting_weights)
        config = self.load_config()
        config["default_starting_model_mode"] = starting_weights["mode"]
        self.save_config(config)
        return starting_weights

    def load_model_registry(self) -> dict[str, Any]:
        self.ensure_model_registry()
        return self.read_json(self.model_registry_path)

    def import_pretrained_model(
        self,
        source_path: str | Path,
        *,
        model_name: str = "",
        architecture: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve()
        model_id = f"imported_model_{len(self.load_model_registry().get('models', [])) + 1:03d}_{uuid4().hex[:8]}"
        relative_path = Path("models") / "imported" / f"{model_id}{source.suffix or '.pt'}"
        shutil.copy2(source, self.root / relative_path)
        entry = {
            "model_id": model_id,
            "model_name": model_name or source.stem,
            "source_filename": source.name,
            "model_path": str(relative_path),
            "architecture": architecture or self.load_architecture_config(),
            "imported_at": utc_now(),
            "format": source.suffix.lower().lstrip(".") or "unknown",
        }
        registry = self.load_model_registry()
        registry.setdefault("models", []).append(entry)
        registry["updated_at"] = utc_now()
        self.write_json(self.model_registry_path, registry)
        return entry

    def load_manifest(self) -> dict[str, Any]:
        self.ensure_manifest()
        return self.read_json(self.manifest_path)

    def dataset_pairs(self) -> list[dict[str, Any]]:
        return list(self.load_manifest().get("pairs", []))

    def dataset_count(self) -> int:
        return len(self.dataset_pairs())

    def add_pair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        image_layer_name: str = "",
        mask_layer_name: str = "",
        mask_preparation: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pair_id = f"pair_{len(self.dataset_pairs()) + 1:04d}_{uuid4().hex[:8]}"
        image_path = Path("dataset") / "images" / f"{pair_id}.tif"
        mask_path = Path("dataset") / "masks" / f"{pair_id}.tif"
        prepared_mask, preparation_metadata = self.prepare_mask(mask, mask_preparation)
        tifffile.imwrite(self.root / image_path, np.asarray(image))
        tifffile.imwrite(self.root / mask_path, prepared_mask)

        pair = {
            "pair_id": pair_id,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_layer_name": image_layer_name,
            "mask_layer_name": mask_layer_name,
            "shape": list(np.asarray(image).shape),
            "mask_shape": list(prepared_mask.shape),
            "created_at": utc_now(),
            "used_in_checkpoints": [],
            "mask_preparation": preparation_metadata,
            "metadata": metadata or {},
        }
        manifest = self.load_manifest()
        manifest["pairs"].append(pair)
        manifest["updated_at"] = utc_now()
        self.write_json(self.manifest_path, manifest)

        config = self.load_config()
        config["last_selected_image_layer"] = image_layer_name
        config["last_selected_mask_layer"] = mask_layer_name
        self.save_config(config)
        return pair

    def prepare_mask(
        self,
        mask: np.ndarray,
        mask_preparation: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = {
            **default_mask_preparation_config(),
            **self.load_config().get("mask_preparation", {}),
            **(mask_preparation or {}),
        }
        source = np.asarray(mask)
        source_labels = [int(value) for value in np.unique(source)]
        mode = settings["mode"]
        if mode == "merge_nonzero_to_foreground":
            prepared = (source > 0).astype(np.uint8)
            saved_labels = [int(value) for value in np.unique(prepared)]
            transform = "nonzero_to_foreground"
            mask_mode = "binary"
        elif mode == "keep_labels_as_multiclass":
            prepared = source.astype(np.int64, copy=False)
            saved_labels = [int(value) for value in np.unique(prepared)]
            transform = "keep_labels"
            mask_mode = "multiclass"
            architecture = self.load_architecture_config()
            expected = set(range(int(architecture.get("num_classes", 2))))
            unexpected = sorted(set(saved_labels) - expected)
            if unexpected and settings.get("strict_multiclass_validation", True):
                raise ValueError(
                    "Mask contains labels "
                    f"{saved_labels}, but current multiclass configuration expects labels "
                    f"0..{max(expected) if expected else 0}."
                )
        else:
            label_map = {int(k): int(v) for k, v in settings.get("manual_label_map", {}).items()}
            prepared = np.zeros_like(source, dtype=np.int64)
            for source_value, target_value in label_map.items():
                prepared[source == source_value] = target_value
            saved_labels = [int(value) for value in np.unique(prepared)]
            transform = "manual_label_map"
            mask_mode = "manual"
        metadata = {
            "mask_mode": mask_mode,
            "target_class_name": settings.get("target_class_name", "foreground"),
            "label_transform": transform,
            "source_labels": source_labels,
            "saved_labels": saved_labels,
        }
        if mode == "manual_label_map":
            metadata["manual_label_map"] = settings.get("manual_label_map", {})
        return prepared, metadata

    def load_checkpoints(self) -> dict[str, Any]:
        self.ensure_checkpoint_index()
        return self.read_json(self.checkpoints_path)

    def checkpoint_entries(self) -> list[dict[str, Any]]:
        return list(self.load_checkpoints().get("checkpoints", []))

    def latest_checkpoint(self) -> dict[str, Any] | None:
        latest_id = self.load_config().get("latest_checkpoint_id", "")
        for checkpoint in self.checkpoint_entries():
            if checkpoint.get("checkpoint_id") == latest_id:
                return checkpoint
        entries = self.checkpoint_entries()
        return entries[-1] if entries else None

    def selected_dataset_pairs(
        self,
        dataset_source: str,
        *,
        selected_pair_ids: Iterable[str] | None = None,
        parent_checkpoint_id: str = "",
    ) -> list[dict[str, Any]]:
        pairs = self.dataset_pairs()
        if dataset_source == "Selected masks only":
            selected = set(selected_pair_ids or [])
            return [pair for pair in pairs if pair["pair_id"] in selected]
        if dataset_source == "New masks since last checkpoint":
            checkpoint_id = parent_checkpoint_id or self.load_config().get("latest_checkpoint_id", "")
            return [pair for pair in pairs if checkpoint_id not in pair.get("used_in_checkpoints", [])]
        return pairs

    def register_checkpoint(
        self,
        *,
        checkpoint_bytes: bytes | None = None,
        source_checkpoint_path: str | Path | None = None,
        parent_checkpoint_id: str = "",
        training_mode: str = "continue",
        dataset_pair_ids: list[str],
        number_of_patches: int = 0,
        train_validation_split: float = 0.2,
        loss_metrics: dict[str, float] | None = None,
        dice_iou_metrics: dict[str, float] | None = None,
        benchmark_summary: str = "",
        architecture: dict[str, Any] | None = None,
        starting_weights: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        index = self.load_checkpoints()
        checkpoint_number = len(index.get("checkpoints", [])) + 1
        checkpoint_id = f"unet_run_{checkpoint_number:03d}"
        relative_path = Path("checkpoints") / f"{checkpoint_id}.pt"
        destination = self.root / relative_path
        if source_checkpoint_path:
            shutil.copy2(source_checkpoint_path, destination)
        else:
            destination.write_bytes(checkpoint_bytes if checkpoint_bytes is not None else b"")

        number_of_images = len(dataset_pair_ids)
        metadata = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "training_mode": training_mode,
            "dataset_pair_ids_used": dataset_pair_ids,
            "number_of_images": number_of_images,
            "number_of_patches": number_of_patches,
            "train_validation_split": train_validation_split,
            "loss_metrics": loss_metrics or {},
            "dice_iou_metrics": dice_iou_metrics or {},
            "timestamp": utc_now(),
            "checkpoint_path": str(relative_path),
            "benchmark_summary": benchmark_summary,
            "architecture": architecture or self.load_architecture_config(),
            "starting_weights": starting_weights or self.load_starting_weights_config(),
        }
        metadata["architecture_summary"] = describe_basic_unet(metadata["architecture"])
        index["checkpoints"].append(metadata)
        index["updated_at"] = utc_now()
        self.write_json(self.checkpoints_path, index)

        shutil.copy2(destination, self.latest_checkpoint_path)
        config = self.load_config()
        config["latest_checkpoint_id"] = checkpoint_id
        config["latest_checkpoint_path"] = str(Path("checkpoints") / "latest.pt")
        self.save_config(config)
        self.mark_pairs_used(dataset_pair_ids, checkpoint_id)
        self.append_training_run(metadata)
        self.append_benchmark(metadata)
        return metadata

    def mark_pairs_used(self, pair_ids: Iterable[str], checkpoint_id: str) -> None:
        selected = set(pair_ids)
        manifest = self.load_manifest()
        for pair in manifest.get("pairs", []):
            if pair.get("pair_id") in selected:
                used = pair.setdefault("used_in_checkpoints", [])
                if checkpoint_id not in used:
                    used.append(checkpoint_id)
        manifest["updated_at"] = utc_now()
        self.write_json(self.manifest_path, manifest)

    def append_training_run(self, checkpoint_metadata: dict[str, Any]) -> None:
        history = self.read_json(self.training_runs_path)
        history.setdefault("runs", []).append(checkpoint_metadata)
        history["updated_at"] = utc_now()
        self.write_json(self.training_runs_path, history)

    def training_runs(self) -> list[dict[str, Any]]:
        self.ensure_training_history()
        return list(self.read_json(self.training_runs_path).get("runs", []))

    def append_benchmark(self, checkpoint_metadata: dict[str, Any]) -> None:
        loss = checkpoint_metadata.get("loss_metrics", {})
        dice_iou = checkpoint_metadata.get("dice_iou_metrics", {})
        with self.benchmark_history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["timestamp", "checkpoint_id", "loss", "val_loss", "dice", "iou", "summary"],
            )
            writer.writerow(
                {
                    "timestamp": checkpoint_metadata.get("timestamp", ""),
                    "checkpoint_id": checkpoint_metadata.get("checkpoint_id", ""),
                    "loss": loss.get("loss", ""),
                    "val_loss": loss.get("val_loss", ""),
                    "dice": dice_iou.get("dice", ""),
                    "iou": dice_iou.get("iou", ""),
                    "summary": checkpoint_metadata.get("benchmark_summary", ""),
                }
            )

    def latest_benchmark_summary(self) -> str:
        if not self.benchmark_history_path.exists():
            return ""
        with self.benchmark_history_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1].get("summary", "") if rows else ""

    def save_prediction(self, prediction: np.ndarray, *, name: str | None = None) -> Path:
        existing = sorted((self.root / "predictions").glob("prediction_*.tif"))
        filename = name or f"prediction_{len(existing) + 1:03d}.tif"
        relative_path = Path("predictions") / filename
        tifffile.imwrite(self.root / relative_path, np.asarray(prediction))
        return relative_path

    def prediction_outputs(self) -> list[Path]:
        return sorted((self.root / "predictions").glob("*.tif"))

    @staticmethod
    def output_channels_for_architecture(architecture: dict[str, Any]) -> int:
        if architecture.get("output_mode") == "binary":
            return 1
        return int(architecture.get("num_classes", 2))

    @staticmethod
    def architecture_compatibility(
        selected: dict[str, Any],
        candidate: dict[str, Any],
    ) -> tuple[bool, str]:
        keys = (
            "backend",
            "spatial_dims",
            "preset",
            "depth",
            "base_channels",
            "normalization",
            "upsampling",
            "input_channels",
            "output_mode",
            "num_classes",
            "output_channels",
        )
        for key in keys:
            if selected.get(key) != candidate.get(key):
                return False, f"{key} mismatch: selected={selected.get(key)} candidate={candidate.get(key)}"
        return True, "Architecture is compatible."

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
