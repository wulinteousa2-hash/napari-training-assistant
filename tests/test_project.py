from pathlib import Path

import numpy as np

from napari_training_assistant.project import TrainingProject
from napari_training_assistant.unet import describe_basic_unet


def test_create_project_writes_required_structure(tmp_path: Path):
    project = TrainingProject.create_or_open(tmp_path / "training_project")

    assert project.config_path.exists()
    assert project.manifest_path.exists()
    assert project.checkpoints_path.exists()
    assert project.training_runs_path.exists()
    assert project.benchmark_history_path.exists()
    assert project.architecture_config_path.exists()
    assert project.starting_weights_config_path.exists()
    assert project.model_registry_path.exists()
    assert project.validate().status == "valid"


def test_add_pair_persists_manifest_and_reload(tmp_path: Path):
    project = TrainingProject.create_or_open(tmp_path / "training_project")
    pair = project.add_pair(
        np.zeros((8, 8), dtype=np.uint8),
        np.ones((8, 8), dtype=np.uint8),
        image_layer_name="image",
        mask_layer_name="mask",
    )

    reopened = TrainingProject.create_or_open(project.root)
    assert reopened.dataset_count() == 1
    assert reopened.dataset_pairs()[0]["pair_id"] == pair["pair_id"]
    assert (project.root / pair["image_path"]).exists()
    assert (project.root / pair["mask_path"]).exists()


def test_register_checkpoint_appends_history_and_updates_latest(tmp_path: Path):
    project = TrainingProject.create_or_open(tmp_path / "training_project")
    pair = project.add_pair(np.zeros((4, 4)), np.ones((4, 4)))

    checkpoint = project.register_checkpoint(
        checkpoint_bytes=b"checkpoint",
        parent_checkpoint_id="",
        training_mode="scratch",
        dataset_pair_ids=[pair["pair_id"]],
        number_of_patches=3,
        train_validation_split=0.25,
        loss_metrics={"loss": 1.0, "val_loss": 1.2},
        dice_iou_metrics={"dice": 0.8, "iou": 0.7},
        benchmark_summary="dice 0.8",
    )

    config = project.load_config()
    assert config["latest_checkpoint_id"] == checkpoint["checkpoint_id"]
    assert (project.root / "checkpoints" / "latest.pt").exists()
    assert len(project.checkpoint_entries()) == 1
    assert len(project.training_runs()) == 1
    assert project.latest_benchmark_summary() == "dice 0.8"
    assert project.checkpoint_entries()[0]["architecture"]["backend"] == "basic_unet"
    assert project.checkpoint_entries()[0]["architecture"]["spatial_dims"] == "2d"
    assert project.checkpoint_entries()[0]["architecture_summary"]["double_conv_layers"] == 18


def test_dataset_source_new_masks_since_last_checkpoint(tmp_path: Path):
    project = TrainingProject.create_or_open(tmp_path / "training_project")
    first = project.add_pair(np.zeros((2, 2)), np.ones((2, 2)))
    project.register_checkpoint(
        checkpoint_bytes=b"checkpoint",
        parent_checkpoint_id="",
        training_mode="scratch",
        dataset_pair_ids=[first["pair_id"]],
    )
    second = project.add_pair(np.zeros((2, 2)), np.ones((2, 2)))

    new_pairs = project.selected_dataset_pairs("New masks since last checkpoint")
    assert [pair["pair_id"] for pair in new_pairs] == [second["pair_id"]]


def test_binary_mask_preparation_merges_all_nonzero_labels(tmp_path: Path):
    project = TrainingProject.create_or_open(tmp_path / "training_project")
    mask = np.array([[0, 1, 2], [3, 0, 4]], dtype=np.uint16)

    pair = project.add_pair(np.zeros((2, 3), dtype=np.uint8), mask)

    stored = project.dataset_pairs()[0]
    assert stored["pair_id"] == pair["pair_id"]
    assert stored["mask_preparation"]["label_transform"] == "nonzero_to_foreground"
    assert stored["mask_preparation"]["source_labels"] == [0, 1, 2, 3, 4]
    assert stored["mask_preparation"]["saved_labels"] == [0, 1]


def test_architecture_and_starting_weights_reload(tmp_path: Path):
    project = TrainingProject.create_or_open(tmp_path / "training_project")
    project.save_architecture_config(
        {
            "backend": "basic_unet",
            "spatial_dims": "2d",
            "preset": "standard_unet",
            "depth": 5,
            "base_channels": 16,
            "normalization": "instance",
            "upsampling": "bilinear",
            "input_channels": 3,
            "output_mode": "multiclass",
            "num_classes": 4,
            "class_labels": {"0": "background", "1": "cell", "2": "nucleus", "3": "debris"},
        }
    )
    project.save_starting_weights_config({"mode": "scratch"})

    reopened = TrainingProject.create_or_open(project.root)
    architecture = reopened.load_architecture_config()
    starting_weights = reopened.load_starting_weights_config()
    assert architecture["depth"] == 5
    assert architecture["output_channels"] == 4
    assert architecture["loss"] == "cross_entropy_dice"
    assert starting_weights["mode"] == "scratch"


def test_basic_unet_descriptor_counts_layers():
    description = describe_basic_unet({"depth": 4, "base_channels": 32, "output_mode": "binary"})

    assert description["feature_channels"] == [32, 64, 128, 256, 512]
    assert description["double_conv_layers"] == 18
    assert description["upsampling_layers"] == 4
    assert description["final_projection_layers"] == 1
