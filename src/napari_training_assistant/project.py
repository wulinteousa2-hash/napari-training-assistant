"""Persistent training project storage."""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import tifffile

from napari_training_assistant import __version__
from napari_training_assistant.io.loaders import load_image_any
from napari_training_assistant.unet import describe_basic_unet


PROJECT_CONFIG = "project_config.json"
DATASET_MANIFEST = Path("dataset") / "manifest.json"
CHECKPOINTS_JSON = Path("checkpoints") / "checkpoints.json"
TRAINING_RUNS_JSON = Path("history") / "training_runs.json"
BENCHMARK_HISTORY_CSV = Path("history") / "benchmark_history.csv"
ARCHITECTURE_CONFIG = Path("architecture") / "architecture_config.json"
STARTING_WEIGHTS_CONFIG = Path("models") / "starting_weights_config.json"
MODEL_REGISTRY = Path("models") / "model_registry.json"
SAM3_CONFIG = Path("sam3") / "sam3_config.json"
TASKS_JSON = Path("tasks") / "tasks.json"


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
        "auto_expand_multiclass_labels": True,
    }


def default_sam3_config() -> dict[str, Any]:
    return {
        "default_mode": "2d_box",
        "sam3_2d_model_dir": "",
        "sam3_3d_model_dir": "",
        "device": "cuda",
        "confidence_threshold": 0.35,
        "compile_model": False,
        "propagation_direction": "forward",
        "sam31_runtime_mode": "performance",
        "sam31_windows_compatibility_mode": False,
        "sam31_no_write_benchmark": False,
        "sam31_debug_diagnostics": False,
        "last_image_layer": "",
        "points_layer_name": "SAM3 points",
        "live_points_layer_name": "SAM3 live points",
        "boxes_layer_name": "SAM3 boxes",
        "exemplar_layer_name": "SAM3 exemplar boxes",
        "multiplex_prompt_layer_name": "SAM3 3D prompts",
        "preview_labels_layer_name": "SAM3 preview labels",
        "propagated_labels_layer_name": "SAM3 propagated labels",
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
        "default_sam3_config_path": str(SAM3_CONFIG),
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
        Path("sam3"),
        Path("tasks"),
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
        project.ensure_sam3_config()
        project.ensure_tasks()
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
    def sam3_config_path(self) -> Path:
        return self.root / SAM3_CONFIG

    @property
    def tasks_registry_path(self) -> Path:
        return self.root / TASKS_JSON

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

    def ensure_sam3_config(self) -> None:
        if not self.sam3_config_path.exists():
            self.write_json(self.sam3_config_path, default_sam3_config())

    @staticmethod
    def _task_id_from_name(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
        return slug or f"task_{uuid4().hex[:8]}"

    @staticmethod
    def _normalise_class_labels(class_labels: dict[Any, str] | list[str]) -> dict[str, str]:
        if isinstance(class_labels, dict):
            return {str(int(key)): str(value) for key, value in sorted(class_labels.items(), key=lambda item: int(item[0]))}
        return {str(index): str(value) for index, value in enumerate(class_labels)}

    def _task_relative_root(self, task_id: str) -> Path:
        return Path("tasks") / task_id

    def _task_root(self, task_id: str) -> Path:
        return self.root / self._task_relative_root(task_id)

    def _ensure_task_structure(self, task_id: str) -> None:
        task_root = self._task_root(task_id)
        for relative in (
            Path("dataset") / "images",
            Path("dataset") / "masks",
            Path("checkpoints"),
            Path("predictions"),
            Path("history"),
        ):
            (task_root / relative).mkdir(parents=True, exist_ok=True)
        manifest_path = task_root / "dataset" / "manifest.json"
        if not manifest_path.exists():
            self.write_json(
                manifest_path,
                {"version": 1, "created_at": utc_now(), "updated_at": utc_now(), "pairs": []},
            )
        checkpoints_path = task_root / "checkpoints" / "checkpoints.json"
        if not checkpoints_path.exists():
            self.write_json(
                checkpoints_path,
                {"version": 1, "created_at": utc_now(), "updated_at": utc_now(), "checkpoints": []},
            )
        training_runs_path = task_root / "history" / "training_runs.json"
        if not training_runs_path.exists():
            self.write_json(
                training_runs_path,
                {"version": 1, "created_at": utc_now(), "updated_at": utc_now(), "runs": []},
            )
        benchmark_path = task_root / "history" / "benchmark_history.csv"
        if not benchmark_path.exists():
            with benchmark_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "checkpoint_id", "loss", "val_loss", "dice", "iou", "summary"],
                )
                writer.writeheader()

    def _write_task_config(
        self,
        *,
        task_id: str,
        display_name: str,
        output_mode: str,
        class_labels: dict[str, str],
        starting_mode: str = "fresh_empty_model",
        architecture: dict[str, Any] | None = None,
        created_at: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        now = utc_now()
        config = {
            "task_id": task_id,
            "display_name": display_name,
            "output_mode": output_mode,
            "class_labels": class_labels,
            "created_at": created_at or now,
            "updated_at": now,
            "latest_checkpoint_id": "",
            "latest_checkpoint_path": "",
            "training_mode_default": "continue_latest",
            "starting_mode": starting_mode,
            "architecture": architecture or self.load_architecture_config(),
            "notes": notes,
        }
        task_root = self._task_root(task_id)
        task_root.mkdir(parents=True, exist_ok=True)
        self.write_json(task_root / "task_config.json", config)
        return config

    def ensure_tasks(self) -> None:
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        if not self.tasks_registry_path.exists():
            now = utc_now()
            task_id = "default_binary"
            class_labels = {"0": "background", "1": "foreground"}
            self._ensure_task_structure(task_id)
            self._write_task_config(
                task_id=task_id,
                display_name="Default binary",
                output_mode="binary",
                class_labels=class_labels,
                starting_mode="fresh_empty_model",
                created_at=now,
            )
            self.write_json(
                self.tasks_registry_path,
                {
                    "version": 1,
                    "active_task_id": task_id,
                    "tasks": [
                        {
                            "task_id": task_id,
                            "display_name": "Default binary",
                            "output_mode": "binary",
                            "class_labels": class_labels,
                            "created_at": now,
                            "updated_at": now,
                            "task_path": str(self._task_relative_root(task_id)),
                            "archived": False,
                        }
                    ],
                    "created_at": now,
                    "updated_at": now,
                },
            )
            return
        registry = self.load_task_registry()
        if not registry.get("tasks"):
            self.tasks_registry_path.unlink()
            self.ensure_tasks()
            return
        for entry in registry.get("tasks", []):
            self._ensure_task_structure(entry["task_id"])

    def load_task_registry(self) -> dict[str, Any]:
        if not self.tasks_registry_path.exists():
            self.ensure_tasks()
        return self.read_json(self.tasks_registry_path)

    def save_task_registry(self, registry: dict[str, Any]) -> None:
        registry["updated_at"] = utc_now()
        self.write_json(self.tasks_registry_path, registry)

    def task_entries(self, include_archived: bool = False) -> list[dict[str, Any]]:
        tasks = list(self.load_task_registry().get("tasks", []))
        if include_archived:
            return tasks
        return [task for task in tasks if not task.get("archived", False)]

    def active_task_id(self) -> str:
        registry = self.load_task_registry()
        active = registry.get("active_task_id", "")
        if active:
            return active
        entries = self.task_entries()
        if not entries:
            self.ensure_tasks()
            entries = self.task_entries()
        active = entries[0]["task_id"]
        self.set_active_task(active)
        return active

    def set_active_task(self, task_id: str) -> None:
        registry = self.load_task_registry()
        if task_id not in {task["task_id"] for task in registry.get("tasks", []) if not task.get("archived", False)}:
            raise ValueError(f"Unknown model task: {task_id}")
        registry["active_task_id"] = task_id
        self.save_task_registry(registry)

    def active_task_root(self) -> Path:
        task_id = self.active_task_id()
        self._ensure_task_structure(task_id)
        return self._task_root(task_id)

    def active_task_config(self) -> dict[str, Any]:
        path = self.active_task_root() / "task_config.json"
        if not path.exists():
            entry = next(task for task in self.task_entries(include_archived=True) if task["task_id"] == self.active_task_id())
            self._write_task_config(
                task_id=entry["task_id"],
                display_name=entry["display_name"],
                output_mode=entry["output_mode"],
                class_labels=entry["class_labels"],
            )
        return self.read_json(path)

    def save_active_task_config(self, config: dict[str, Any]) -> None:
        config["updated_at"] = utc_now()
        self.write_json(self.active_task_root() / "task_config.json", config)

    def create_task(
        self,
        name: str,
        output_mode: str,
        class_labels: dict[Any, str] | list[str],
        copy_from_task_id: str | None = None,
    ) -> dict[str, Any]:
        output_mode = output_mode.strip().lower()
        if output_mode not in {"binary", "multiclass"}:
            raise ValueError("output_mode must be binary or multiclass")
        registry = self.load_task_registry()
        base_task_id = self._task_id_from_name(name)
        task_id = base_task_id
        existing_ids = {task["task_id"] for task in registry.get("tasks", [])}
        suffix = 2
        while task_id in existing_ids:
            task_id = f"{base_task_id}_{suffix}"
            suffix += 1
        class_labels = self._normalise_class_labels(class_labels)
        if output_mode == "binary" and set(class_labels.keys()) != {"0", "1"}:
            raise ValueError("Binary tasks must define labels 0 and 1.")
        architecture = self.load_architecture_config()
        if copy_from_task_id:
            source_config = self.read_json(self._task_root(copy_from_task_id) / "task_config.json")
            architecture = source_config.get("architecture", architecture)
        architecture = {**architecture, "output_mode": output_mode, "class_labels": class_labels}
        architecture["num_classes"] = max(int(key) for key in class_labels) + 1
        architecture["output_channels"] = 1 if output_mode == "binary" else architecture["num_classes"]
        now = utc_now()
        self._ensure_task_structure(task_id)
        self._write_task_config(
            task_id=task_id,
            display_name=name,
            output_mode=output_mode,
            class_labels=class_labels,
            starting_mode="fresh_empty_model",
            architecture=architecture,
            created_at=now,
        )
        entry = {
            "task_id": task_id,
            "display_name": name,
            "output_mode": output_mode,
            "class_labels": class_labels,
            "created_at": now,
            "updated_at": now,
            "task_path": str(self._task_relative_root(task_id)),
            "archived": False,
        }
        registry.setdefault("tasks", []).append(entry)
        registry["active_task_id"] = task_id
        self.save_task_registry(registry)
        return entry

    def duplicate_task(self, source_task_id: str, new_name: str) -> dict[str, Any]:
        source_config = self.read_json(self._task_root(source_task_id) / "task_config.json")
        return self.create_task(
            new_name,
            source_config.get("output_mode", "binary"),
            source_config.get("class_labels", {"0": "background", "1": "foreground"}),
            copy_from_task_id=source_task_id,
        )

    def rename_task(self, task_id: str, new_name: str) -> None:
        registry = self.load_task_registry()
        for task in registry.get("tasks", []):
            if task["task_id"] == task_id:
                task["display_name"] = new_name
                task["updated_at"] = utc_now()
                break
        else:
            raise ValueError(f"Unknown model task: {task_id}")
        self.save_task_registry(registry)
        config_path = self._task_root(task_id) / "task_config.json"
        config = self.read_json(config_path)
        config["display_name"] = new_name
        config["updated_at"] = utc_now()
        self.write_json(config_path, config)

    def active_task_dataset_manifest_path(self) -> Path:
        return self.active_task_root() / "dataset" / "manifest.json"

    def active_task_checkpoint_index_path(self) -> Path:
        return self.active_task_root() / "checkpoints" / "checkpoints.json"

    def active_task_predictions_dir(self) -> Path:
        return self.active_task_root() / "predictions"

    def active_task_dataset_count(self) -> int:
        return len(self.dataset_pairs())

    def active_task_latest_checkpoint(self) -> dict[str, Any] | None:
        return self.latest_checkpoint()

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
            SAM3_CONFIG,
            TASKS_JSON,
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
        config["default_sam3_config_path"] = str(SAM3_CONFIG)
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
        if self.tasks_registry_path.exists():
            try:
                task_config = self.active_task_config()
                task_config["output_mode"] = architecture["output_mode"]
                task_config["class_labels"] = architecture["class_labels"]
                task_config["architecture"] = architecture
                self.save_active_task_config(task_config)
                registry = self.load_task_registry()
                for task in registry.get("tasks", []):
                    if task["task_id"] == task_config["task_id"]:
                        task["output_mode"] = architecture["output_mode"]
                        task["class_labels"] = architecture["class_labels"]
                        task["updated_at"] = utc_now()
                        break
                self.save_task_registry(registry)
            except Exception:
                pass
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

    def load_sam3_config(self) -> dict[str, Any]:
        self.ensure_sam3_config()
        return {**default_sam3_config(), **self.read_json(self.sam3_config_path)}

    def save_sam3_config(self, sam3_config: dict[str, Any]) -> dict[str, Any]:
        sam3_config = {**default_sam3_config(), **sam3_config}
        self.write_json(self.sam3_config_path, sam3_config)
        return sam3_config

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
        self.ensure_tasks()
        return self.read_json(self.active_task_dataset_manifest_path())

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
        source: str = "manual_label",
        mask_preparation: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pair_id = f"pair_{len(self.dataset_pairs()) + 1:04d}_{uuid4().hex[:8]}"
        task_relative = self._task_relative_root(self.active_task_id())
        image_path = task_relative / "dataset" / "images" / f"{pair_id}.tif"
        mask_path = task_relative / "dataset" / "masks" / f"{pair_id}.tif"
        prepared_mask, preparation_metadata = self.prepare_mask(mask, mask_preparation)
        self.write_tiff_array(self.root / image_path, np.asarray(image))
        self.write_tiff_array(self.root / mask_path, prepared_mask, labels=True)

        pair = {
            "pair_id": pair_id,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_layer_name": image_layer_name,
            "mask_layer_name": mask_layer_name,
            "shape": list(np.asarray(image).shape),
            "mask_shape": list(prepared_mask.shape),
            "created_at": utc_now(),
            "source": source,
            "used_in_checkpoints": [],
            "mask_preparation": preparation_metadata,
            "metadata": metadata or {},
        }
        manifest = self.load_manifest()
        manifest["pairs"].append(pair)
        manifest["updated_at"] = utc_now()
        self.write_json(self.active_task_dataset_manifest_path(), manifest)

        config = self.load_config()
        config["last_selected_image_layer"] = image_layer_name
        config["last_selected_mask_layer"] = mask_layer_name
        self.save_config(config)
        return pair

    def import_existing_pair(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_source = Path(image_path).expanduser().resolve()
        mask_source = Path(mask_path).expanduser().resolve()
        if not image_source.exists() or not mask_source.exists():
            raise FileNotFoundError("Image and mask paths must exist.")
        pair_id = f"pair_{len(self.dataset_pairs()) + 1:04d}_{uuid4().hex[:8]}"
        task_relative = self._task_relative_root(self.active_task_id())
        image_dest = task_relative / "dataset" / "images" / f"{pair_id}{image_source.suffix or '.tif'}"
        mask_dest = task_relative / "dataset" / "masks" / f"{pair_id}{mask_source.suffix or '.tif'}"
        shutil.copy2(image_source, self.root / image_dest)
        shutil.copy2(mask_source, self.root / mask_dest)
        try:
            image = load_image_any(image_source)
            mask = load_image_any(mask_source)
        except Exception:
            image = np.asarray([])
            mask = np.asarray([])
        pair = {
            "pair_id": pair_id,
            "image_path": str(image_dest),
            "mask_path": str(mask_dest),
            "image_layer_name": image_source.name,
            "mask_layer_name": mask_source.name,
            "shape": list(np.asarray(image).shape),
            "mask_shape": list(np.asarray(mask).shape),
            "created_at": utc_now(),
            "source": "imported_pair",
            "used_in_checkpoints": [],
            "mask_preparation": {},
            "metadata": {
                "source_image_path": str(image_source),
                "source_mask_path": str(mask_source),
                **(metadata or {}),
            },
        }
        manifest = self.load_manifest()
        manifest["pairs"].append(pair)
        manifest["updated_at"] = utc_now()
        self.write_json(self.active_task_dataset_manifest_path(), manifest)
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
            prepared = source.astype(self.label_dtype_for(source_labels), copy=False)
            saved_labels = [int(value) for value in np.unique(prepared)]
            transform = "keep_labels"
            mask_mode = "multiclass"
            architecture = self.active_task_config().get("architecture", self.load_architecture_config())
            expected = set(range(int(architecture.get("num_classes", 2))))
            unexpected = sorted(set(saved_labels) - expected)
            if unexpected and settings.get("auto_expand_multiclass_labels", True):
                self._expand_active_task_for_multiclass_labels(saved_labels)
            elif unexpected and settings.get("strict_multiclass_validation", True):
                raise ValueError(
                    "Mask contains labels "
                    f"{saved_labels}, but current multiclass configuration expects labels "
                    f"0..{max(expected) if expected else 0}."
                )
        else:
            label_map = {int(k): int(v) for k, v in settings.get("manual_label_map", {}).items()}
            target_labels = list(label_map.values()) or [0]
            prepared = np.zeros_like(source, dtype=self.label_dtype_for(target_labels))
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
            "source_shape": list(source.shape),
            "saved_shape": list(prepared.shape),
            "spatial_dims": int(source.ndim),
        }
        if mode == "manual_label_map":
            metadata["manual_label_map"] = settings.get("manual_label_map", {})
        return prepared, metadata

    def _expand_active_task_for_multiclass_labels(self, labels: list[int]) -> None:
        labels = sorted({int(label) for label in labels})
        if not labels:
            return
        task_config = self.active_task_config()
        existing = {
            str(int(key)): str(value)
            for key, value in task_config.get("class_labels", {}).items()
        }
        for label in labels:
            existing.setdefault(str(label), "background" if label == 0 else f"class_{label}")

        num_classes = max(labels) + 1
        architecture = {**self.load_architecture_config(), **task_config.get("architecture", {})}
        architecture["output_mode"] = "multiclass"
        architecture["num_classes"] = num_classes
        architecture["output_channels"] = num_classes
        architecture["class_labels"] = {
            str(index): existing.get(str(index), "background" if index == 0 else f"class_{index}")
            for index in range(num_classes)
        }
        architecture["activation"] = "softmax"
        architecture["loss"] = "cross_entropy_dice"

        task_config["output_mode"] = "multiclass"
        task_config["class_labels"] = architecture["class_labels"]
        task_config["architecture"] = architecture
        self.save_active_task_config(task_config)

        registry = self.load_task_registry()
        for task in registry.get("tasks", []):
            if task["task_id"] == task_config["task_id"]:
                task["output_mode"] = "multiclass"
                task["class_labels"] = architecture["class_labels"]
                task["updated_at"] = utc_now()
                break
        self.save_task_registry(registry)
        self.save_architecture_config(architecture)

    def load_checkpoints(self) -> dict[str, Any]:
        self.ensure_tasks()
        return self.read_json(self.active_task_checkpoint_index_path())

    def checkpoint_entries(self) -> list[dict[str, Any]]:
        return list(self.load_checkpoints().get("checkpoints", []))

    def latest_checkpoint(self) -> dict[str, Any] | None:
        latest_id = self.active_task_config().get("latest_checkpoint_id", "")
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
            checkpoint_id = parent_checkpoint_id or self.active_task_config().get("latest_checkpoint_id", "")
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
        relative_path = self._task_relative_root(self.active_task_id()) / "checkpoints" / f"{checkpoint_id}.pt"
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
        self.write_json(self.active_task_checkpoint_index_path(), index)

        task_latest_path = self.active_task_root() / "checkpoints" / "latest.pt"
        shutil.copy2(destination, task_latest_path)
        shutil.copy2(destination, self.latest_checkpoint_path)
        config = self.load_config()
        config["latest_checkpoint_id"] = checkpoint_id
        config["latest_checkpoint_path"] = str(relative_path.parent / "latest.pt")
        self.save_config(config)
        task_config = self.active_task_config()
        task_config["latest_checkpoint_id"] = checkpoint_id
        task_config["latest_checkpoint_path"] = str(relative_path.parent / "latest.pt")
        task_config["starting_mode"] = "continue_latest_checkpoint"
        self.save_active_task_config(task_config)
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
        self.write_json(self.active_task_dataset_manifest_path(), manifest)

    def append_training_run(self, checkpoint_metadata: dict[str, Any]) -> None:
        history = self.read_json(self.active_task_root() / "history" / "training_runs.json")
        history.setdefault("runs", []).append(checkpoint_metadata)
        history["updated_at"] = utc_now()
        self.write_json(self.active_task_root() / "history" / "training_runs.json", history)

    def training_runs(self) -> list[dict[str, Any]]:
        self.ensure_training_history()
        self.ensure_tasks()
        return list(self.read_json(self.active_task_root() / "history" / "training_runs.json").get("runs", []))

    def append_benchmark(self, checkpoint_metadata: dict[str, Any]) -> None:
        loss = checkpoint_metadata.get("loss_metrics", {})
        dice_iou = checkpoint_metadata.get("dice_iou_metrics", {})
        with (self.active_task_root() / "history" / "benchmark_history.csv").open("a", newline="", encoding="utf-8") as handle:
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
        benchmark_path = self.active_task_root() / "history" / "benchmark_history.csv"
        if not benchmark_path.exists():
            return ""
        with benchmark_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1].get("summary", "") if rows else ""

    def save_prediction(self, prediction: np.ndarray, *, name: str | None = None) -> Path:
        existing = sorted(self.active_task_predictions_dir().glob("prediction_*.tif"))
        filename = name or f"prediction_{len(existing) + 1:03d}.tif"
        relative_path = self._task_relative_root(self.active_task_id()) / "predictions" / filename
        tifffile.imwrite(self.root / relative_path, np.asarray(prediction))
        return relative_path

    def prediction_outputs(self) -> list[Path]:
        return sorted(self.active_task_predictions_dir().glob("*.tif"))

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

    @staticmethod
    def label_dtype_for(labels: Iterable[int]) -> np.dtype:
        labels = [int(label) for label in labels]
        max_label = max(labels) if labels else 0
        if max_label <= np.iinfo(np.uint8).max:
            return np.dtype(np.uint8)
        if max_label <= np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        return np.dtype(np.uint32)

    @staticmethod
    def write_tiff_array(path: Path, array: np.ndarray, *, labels: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.asarray(array)
        kwargs: dict[str, Any] = {}
        if labels or (data.ndim == 3 and data.shape[-1] not in (3, 4)):
            kwargs["photometric"] = "minisblack"
        tifffile.imwrite(path, data, **kwargs)
