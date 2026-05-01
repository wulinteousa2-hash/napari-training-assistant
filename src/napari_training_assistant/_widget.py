"""Qt widget for persistent SAM3-to-U-Net training projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_training_assistant.project import TrainingProject


TRAINING_MODES = (
    "Continue from latest checkpoint",
    "Continue from selected checkpoint",
    "Retrain from scratch",
)
DATASET_SOURCES = (
    "All accepted masks",
    "New masks since last checkpoint",
    "Selected masks only",
)
ARCHITECTURE_BACKENDS = ("basic_unet",)
SPATIAL_DIMS = ("2d", "3d - future")
UNET_PRESETS = ("standard_unet", "resunet_2d - future", "attention_unet_2d - future")
UNET_NORMALIZATION = ("batch", "instance", "group", "none")
UNET_UPSAMPLING = ("transpose", "bilinear")
OUTPUT_MODES = ("binary", "multiclass")
STARTING_WEIGHT_MODES = (
    "scratch",
    "latest_project_checkpoint",
    "selected_project_checkpoint",
    "imported_pretrained_unet",
)
MASK_PREPARATION_MODES = (
    "merge_nonzero_to_foreground",
    "keep_labels_as_multiclass",
)


class TrainingAssistantWidget(QWidget):
    """Main napari dock widget."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self.project: TrainingProject | None = None
        self._build_ui()
        self._set_project_actions_enabled(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        columns = QGridLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.addWidget(self._build_project_section())
        left.addWidget(self._build_dataset_section())
        left.addWidget(self._build_checkpoint_section())
        left.addWidget(self._build_prediction_section())
        left.addStretch(1)
        right.addWidget(self._build_training_settings_section())
        right.addWidget(self._build_architecture_section())
        right.addWidget(self._build_starting_weights_section())
        right.addWidget(self._build_training_actions_section())
        right.addStretch(1)
        columns.addLayout(left, 0, 0)
        columns.addLayout(right, 0, 1)
        columns.setColumnStretch(0, 1)
        columns.setColumnStretch(1, 1)
        layout.addLayout(columns)
        layout.addStretch(1)

    def _build_project_section(self) -> QGroupBox:
        box = QGroupBox("Project")
        layout = QVBoxLayout(box)

        select_row = QHBoxLayout()
        self.select_project_button = QPushButton("Select / Create Training Project Folder")
        self.select_project_button.clicked.connect(self.select_project_folder)
        select_row.addWidget(self.select_project_button)
        layout.addLayout(select_row)

        form = QFormLayout()
        self.project_path_label = QLabel("No project selected")
        self.project_path_label.setWordWrap(True)
        self.dataset_count_label = QLabel("0")
        self.latest_checkpoint_label = QLabel("None")
        self.latest_benchmark_label = QLabel("None")
        self.project_state_label = QLabel("missing")
        form.addRow("Current project path", self.project_path_label)
        form.addRow("Dataset count", self.dataset_count_label)
        form.addRow("Latest checkpoint", self.latest_checkpoint_label)
        form.addRow("Latest benchmark summary", self.latest_benchmark_label)
        form.addRow("Project state", self.project_state_label)
        layout.addLayout(form)
        return box

    def _build_dataset_section(self) -> QGroupBox:
        box = QGroupBox("Dataset history")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self.image_layer_combo = QComboBox()
        self.mask_layer_combo = QComboBox()
        self.mask_layer_combo.currentTextChanged.connect(lambda _text: self._refresh_detected_labels())
        self.refresh_layers_button = QPushButton("Refresh layers")
        self.refresh_layers_button.clicked.connect(self.refresh_layer_choices)
        row.addWidget(self.image_layer_combo)
        row.addWidget(self.mask_layer_combo)
        row.addWidget(self.refresh_layers_button)
        layout.addLayout(row)

        self.add_mask_button = QPushButton("Add accepted image/mask pair")
        self.add_mask_button.clicked.connect(self.add_current_mask_pair)
        layout.addWidget(self.add_mask_button)

        preparation = QGroupBox("Mask preparation")
        preparation_layout = QFormLayout(preparation)
        self.mask_preparation_combo = QComboBox()
        self.mask_preparation_combo.addItems(MASK_PREPARATION_MODES)
        self.mask_preparation_combo.currentTextChanged.connect(self.persist_mask_preparation)
        self.target_class_name_edit = QTextEdit()
        self.target_class_name_edit.setFixedHeight(32)
        self.target_class_name_edit.setPlainText("foreground")
        self.target_class_name_edit.textChanged.connect(self.persist_mask_preparation)
        self.detected_labels_label = QLabel("Detected labels: none")
        self.detected_labels_label.setWordWrap(True)
        self.mask_cleanup_note = QLabel(
            "Binary quick fix: merge all nonzero SAM-style instance labels into one foreground class."
        )
        self.mask_cleanup_note.setWordWrap(True)
        preparation_layout.addRow("Quick fix", self.mask_preparation_combo)
        preparation_layout.addRow("Target class", self.target_class_name_edit)
        preparation_layout.addRow("", self.detected_labels_label)
        preparation_layout.addRow("", self.mask_cleanup_note)
        layout.addWidget(preparation)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.dataset_list)
        return box

    def _build_training_settings_section(self) -> QGroupBox:
        box = QGroupBox("Training settings")
        layout = QFormLayout(box)

        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItems(TRAINING_MODES)
        self.dataset_source_combo = QComboBox()
        self.dataset_source_combo.addItems(DATASET_SOURCES)
        self.dataset_source_combo.currentTextChanged.connect(self._show_dataset_source_warning)

        self.new_masks_warning = QLabel(
            "Training only on new masks may cause the model to forget earlier examples. "
            "Use All accepted masks for safer updates."
        )
        self.new_masks_warning.setWordWrap(True)
        self.new_masks_warning.hide()

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(16, 8192)
        self.patch_size_spin.setSingleStep(16)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(8)
        self.learning_rate_spin.setRange(0.00000001, 10.0)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.validation_split_spin = QDoubleSpinBox()
        self.validation_split_spin.setDecimals(3)
        self.validation_split_spin.setRange(0.0, 0.9)
        self.validation_split_spin.setSingleStep(0.05)
        self.image_normalization_combo = QComboBox()
        self.image_normalization_combo.addItems(["percentile", "minmax", "zscore", "none"])
        self.notes_edit = QTextEdit()
        self.notes_edit.setFixedHeight(64)

        for widget in (
            self.training_mode_combo,
            self.dataset_source_combo,
            self.patch_size_spin,
            self.batch_size_spin,
            self.epochs_spin,
            self.learning_rate_spin,
            self.validation_split_spin,
            self.image_normalization_combo,
        ):
            signal = getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None)
            signal.connect(self.persist_training_settings)
        self.notes_edit.textChanged.connect(self.persist_training_settings)

        layout.addRow("Training mode", self.training_mode_combo)
        layout.addRow("Dataset source", self.dataset_source_combo)
        layout.addRow("", self.new_masks_warning)
        layout.addRow("Patch size", self.patch_size_spin)
        layout.addRow("Batch size", self.batch_size_spin)
        layout.addRow("Epochs", self.epochs_spin)
        layout.addRow("Learning rate", self.learning_rate_spin)
        layout.addRow("Validation split", self.validation_split_spin)
        layout.addRow("Image normalization", self.image_normalization_combo)
        layout.addRow("Notes", self.notes_edit)

        self.save_settings_button = QPushButton("Save training settings")
        self.save_settings_button.clicked.connect(self.persist_training_settings)
        layout.addRow("", self.save_settings_button)
        return box

    def _build_architecture_section(self) -> QGroupBox:
        box = QGroupBox("U-Net architecture")
        layout = QFormLayout(box)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(ARCHITECTURE_BACKENDS)
        self.spatial_dims_combo = QComboBox()
        self.spatial_dims_combo.addItems(SPATIAL_DIMS)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(UNET_PRESETS)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(2, 6)
        self.base_channels_spin = QSpinBox()
        self.base_channels_spin.setRange(8, 256)
        self.base_channels_spin.setSingleStep(8)
        self.arch_normalization_combo = QComboBox()
        self.arch_normalization_combo.addItems(UNET_NORMALIZATION)
        self.upsampling_combo = QComboBox()
        self.upsampling_combo.addItems(UNET_UPSAMPLING)
        self.input_channels_spin = QSpinBox()
        self.input_channels_spin.setRange(1, 64)
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(OUTPUT_MODES)
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 1024)
        self.output_channels_label = QLabel("1")
        self.class_labels_edit = QTextEdit()
        self.class_labels_edit.setFixedHeight(64)
        self.class_labels_edit.setPlainText("0: background\n1: foreground")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.architecture_status_label = QLabel("Compatibility: valid")
        self.architecture_status_label.setWordWrap(True)

        for widget in (
            self.backend_combo,
            self.spatial_dims_combo,
            self.preset_combo,
            self.depth_spin,
            self.base_channels_spin,
            self.arch_normalization_combo,
            self.upsampling_combo,
            self.input_channels_spin,
            self.output_mode_combo,
            self.num_classes_spin,
            self.threshold_spin,
        ):
            signal = getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None)
            signal.connect(self.persist_architecture_settings)
        self.class_labels_edit.textChanged.connect(self.persist_architecture_settings)

        layout.addRow("Backend", self.backend_combo)
        layout.addRow("Dimensionality", self.spatial_dims_combo)
        layout.addRow("Preset", self.preset_combo)
        layout.addRow("Depth", self.depth_spin)
        layout.addRow("Base channels", self.base_channels_spin)
        layout.addRow("Network normalization", self.arch_normalization_combo)
        layout.addRow("Upsampling", self.upsampling_combo)
        layout.addRow("Input channels", self.input_channels_spin)
        layout.addRow("Output mode", self.output_mode_combo)
        layout.addRow("Number of classes", self.num_classes_spin)
        layout.addRow("Model output channels", self.output_channels_label)
        layout.addRow("Class labels", self.class_labels_edit)
        layout.addRow("Binary threshold", self.threshold_spin)
        layout.addRow("", self.architecture_status_label)
        return box

    def _build_starting_weights_section(self) -> QGroupBox:
        box = QGroupBox("Starting weights")
        layout = QFormLayout(box)
        self.starting_weights_combo = QComboBox()
        self.starting_weights_combo.addItems(STARTING_WEIGHT_MODES)
        self.starting_weights_combo.currentTextChanged.connect(self.persist_starting_weights_settings)
        self.import_model_button = QPushButton("Import .pt / .pth")
        self.import_model_button.clicked.connect(self.import_pretrained_model)
        self.imported_model_label = QLabel("No imported model selected")
        self.imported_model_label.setWordWrap(True)
        self.compatibility_label = QLabel("Compatibility: valid")
        self.compatibility_label.setWordWrap(True)
        layout.addRow("Start from", self.starting_weights_combo)
        layout.addRow("", self.import_model_button)
        layout.addRow("Imported model", self.imported_model_label)
        layout.addRow("", self.compatibility_label)
        return box

    def _build_training_actions_section(self) -> QGroupBox:
        box = QGroupBox("Training actions")
        layout = QVBoxLayout(box)
        self.train_button = QPushButton("Train U-Net")
        self.train_button.clicked.connect(self.train_unet)
        self.retrain_button = QPushButton("Retrain from scratch")
        self.retrain_button.clicked.connect(self.retrain_from_scratch)
        layout.addWidget(self.train_button)
        layout.addWidget(self.retrain_button)
        return box

    def _build_checkpoint_section(self) -> QGroupBox:
        box = QGroupBox("Model checkpoints")
        layout = QVBoxLayout(box)
        self.checkpoint_list = QListWidget()
        self.checkpoint_list.itemSelectionChanged.connect(self._refresh_compatibility_status)
        layout.addWidget(self.checkpoint_list)
        return box

    def _build_prediction_section(self) -> QGroupBox:
        box = QGroupBox("Prediction outputs")
        layout = QVBoxLayout(box)
        self.save_predictions_check = QCheckBox("Save predictions to project")
        self.save_predictions_check.setChecked(True)
        layout.addWidget(self.save_predictions_check)
        self.prediction_layer_combo = QComboBox()
        layout.addWidget(self.prediction_layer_combo)
        self.save_prediction_button = QPushButton("Save selected prediction")
        self.save_prediction_button.clicked.connect(self.save_selected_prediction)
        layout.addWidget(self.save_prediction_button)
        self.prediction_list = QListWidget()
        layout.addWidget(self.prediction_list)
        return box

    def select_project_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select / Create Training Project Folder")
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        self.project = TrainingProject.create_or_open(path)
        self._set_project_actions_enabled(True)
        self._load_settings_from_project()
        self.refresh_layer_choices()
        self.refresh_project_summary()

    def require_project(self) -> TrainingProject | None:
        if self.project is not None:
            return self.project
        QMessageBox.warning(
            self,
            "Training project required",
            "Select or create a Training Project Folder before adding masks, training U-Net, or saving predictions.",
        )
        return None

    def refresh_project_summary(self) -> None:
        if self.project is None:
            self._set_project_actions_enabled(False)
            return
        project = self.project
        config = project.load_config()
        state = project.validate()
        latest = project.latest_checkpoint()
        self.project_path_label.setText(str(project.root))
        self.dataset_count_label.setText(str(project.dataset_count()))
        self.latest_checkpoint_label.setText(
            latest["checkpoint_id"] if latest else config.get("latest_checkpoint_path") or "None"
        )
        self.latest_benchmark_label.setText(project.latest_benchmark_summary() or "None")
        self.project_state_label.setText(state.label)
        self._refresh_dataset_list()
        self._refresh_checkpoint_list()
        self._refresh_prediction_list()

    def refresh_layer_choices(self) -> None:
        layer_names = []
        if self.viewer is not None:
            layer_names = [layer.name for layer in self.viewer.layers]
        current_image = self.image_layer_combo.currentText()
        current_mask = self.mask_layer_combo.currentText()
        for combo in (self.image_layer_combo, self.mask_layer_combo, self.prediction_layer_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(layer_names)
            combo.blockSignals(False)
        self._restore_combo_text(self.image_layer_combo, current_image)
        self._restore_combo_text(self.mask_layer_combo, current_mask)
        self._refresh_detected_labels()

    def add_current_mask_pair(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if self.viewer is None:
            QMessageBox.warning(self, "No viewer", "A napari viewer is required to add layer data.")
            return
        image_layer = self._layer_by_name(self.image_layer_combo.currentText())
        mask_layer = self._layer_by_name(self.mask_layer_combo.currentText())
        if image_layer is None or mask_layer is None:
            QMessageBox.warning(self, "Layer required", "Select both an image layer and a mask layer.")
            return
        try:
            project.add_pair(
                np.asarray(image_layer.data),
                np.asarray(mask_layer.data),
                image_layer_name=image_layer.name,
                mask_layer_name=mask_layer.name,
                mask_preparation=self._current_mask_preparation(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Mask labels do not match settings", str(exc))
            return
        self.refresh_project_summary()

    def persist_training_settings(self, *args: Any) -> None:
        if self.project is None:
            return
        self.project.update_settings(
            default_training_mode=self.training_mode_combo.currentText(),
            default_dataset_source=self.dataset_source_combo.currentText(),
            patch_size=self.patch_size_spin.value(),
            batch_size=self.batch_size_spin.value(),
            epochs=self.epochs_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            validation_split=self.validation_split_spin.value(),
            normalization_mode=self.image_normalization_combo.currentText(),
            notes=self.notes_edit.toPlainText(),
        )

    def persist_mask_preparation(self, *args: Any) -> None:
        if self.project is None:
            return
        self.project.update_settings(mask_preparation=self._current_mask_preparation())
        self._refresh_detected_labels()

    def persist_architecture_settings(self, *args: Any) -> None:
        self._sync_architecture_controls()
        if self.project is None:
            return
        architecture = self._current_architecture()
        self.project.save_architecture_config(architecture)
        self._refresh_compatibility_status()

    def persist_starting_weights_settings(self, *args: Any) -> None:
        if self.project is None:
            return
        self.project.save_starting_weights_config(self._current_starting_weights())
        self._refresh_compatibility_status()

    def import_pretrained_model(self) -> None:
        project = self.require_project()
        if project is None:
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import pretrained U-Net checkpoint",
            "",
            "PyTorch checkpoints (*.pt *.pth);;All files (*)",
        )
        if not path:
            return
        entry = project.import_pretrained_model(path, architecture=self._current_architecture())
        self.imported_model_label.setText(f"{entry['model_id']} | {entry['model_name']}")
        self.starting_weights_combo.setCurrentText("imported_pretrained_unet")
        project.save_starting_weights_config(
            {
                **self._current_starting_weights(),
                "imported_model_id": entry["model_id"],
                "imported_model_path": entry["model_path"],
                "compatibility_status": "valid",
                "compatibility_message": "Imported with the current U-Net architecture.",
            }
        )
        self._refresh_compatibility_status()

    def train_unet(self) -> None:
        project = self.require_project()
        if project is None:
            return
        self.persist_training_settings()
        self.persist_architecture_settings()
        self.persist_starting_weights_settings()
        config = project.load_config()
        architecture = project.load_architecture_config()
        starting_weights = project.load_starting_weights_config()
        if architecture.get("spatial_dims") != "2d":
            QMessageBox.warning(self, "3D U-Net not implemented", "3D U-Net is reserved in the project schema but not implemented yet.")
            return
        selected_checkpoint = self._selected_checkpoint()
        training_mode_label = self.training_mode_combo.currentText()
        parent_checkpoint_id = ""
        if training_mode_label == "Continue from latest checkpoint":
            latest = project.latest_checkpoint()
            parent_checkpoint_id = latest.get("checkpoint_id", "") if latest else ""
            training_mode = "continue"
        elif training_mode_label == "Continue from selected checkpoint":
            if not selected_checkpoint:
                QMessageBox.warning(self, "Checkpoint required", "Select a checkpoint to continue from.")
                return
            compatible, message = project.architecture_compatibility(
                architecture, selected_checkpoint.get("architecture", {})
            )
            if not compatible:
                QMessageBox.warning(self, "Checkpoint architecture mismatch", message)
                return
            parent_checkpoint_id = selected_checkpoint.get("checkpoint_id", "")
            training_mode = "continue"
        else:
            training_mode = "scratch"

        selected_pair_ids = self._selected_pair_ids()
        pairs = project.selected_dataset_pairs(
            self.dataset_source_combo.currentText(),
            selected_pair_ids=selected_pair_ids,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        if not pairs:
            QMessageBox.warning(self, "Dataset required", "No dataset pairs match the selected dataset source.")
            return

        pair_ids = [pair["pair_id"] for pair in pairs]
        patch_size = max(config.get("patch_size", 256), 1)
        number_of_patches = len(pair_ids) * max(1, config.get("epochs", 1)) * max(1, 256 // patch_size)
        checkpoint_payload = (
            "napari-training-assistant checkpoint placeholder\n"
            f"training_mode={training_mode}\n"
            f"parent_checkpoint_id={parent_checkpoint_id}\n"
            f"dataset_pair_ids={','.join(pair_ids)}\n"
        ).encode("utf-8")
        project.register_checkpoint(
            checkpoint_bytes=checkpoint_payload,
            parent_checkpoint_id=parent_checkpoint_id,
            training_mode=training_mode,
            dataset_pair_ids=pair_ids,
            number_of_patches=number_of_patches,
            train_validation_split=config.get("validation_split", 0.2),
            loss_metrics={},
            dice_iou_metrics={},
            benchmark_summary=f"{training_mode_label}; {len(pair_ids)} images",
            architecture=architecture,
            starting_weights=starting_weights,
        )
        self.refresh_project_summary()

    def retrain_from_scratch(self) -> None:
        self.training_mode_combo.setCurrentText("Retrain from scratch")
        self.train_unet()

    def save_selected_prediction(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if not self.save_predictions_check.isChecked():
            return
        layer = self._layer_by_name(self.prediction_layer_combo.currentText())
        if layer is None:
            QMessageBox.warning(self, "Layer required", "Select a prediction layer to save.")
            return
        project.save_prediction(np.asarray(layer.data))
        self.refresh_project_summary()

    def _set_project_actions_enabled(self, enabled: bool) -> None:
        for widget in (
            self.image_layer_combo,
            self.mask_layer_combo,
            self.refresh_layers_button,
            self.add_mask_button,
            self.mask_preparation_combo,
            self.target_class_name_edit,
            self.training_mode_combo,
            self.dataset_source_combo,
            self.patch_size_spin,
            self.batch_size_spin,
            self.epochs_spin,
            self.learning_rate_spin,
            self.validation_split_spin,
            self.image_normalization_combo,
            self.backend_combo,
            self.spatial_dims_combo,
            self.preset_combo,
            self.depth_spin,
            self.base_channels_spin,
            self.arch_normalization_combo,
            self.upsampling_combo,
            self.input_channels_spin,
            self.output_mode_combo,
            self.num_classes_spin,
            self.class_labels_edit,
            self.threshold_spin,
            self.starting_weights_combo,
            self.import_model_button,
            self.notes_edit,
            self.save_settings_button,
            self.checkpoint_list,
            self.train_button,
            self.retrain_button,
            self.save_predictions_check,
            self.prediction_layer_combo,
            self.save_prediction_button,
            self.dataset_list,
            self.prediction_list,
        ):
            widget.setEnabled(enabled)

    def _load_settings_from_project(self) -> None:
        if self.project is None:
            return
        config = self.project.load_config()
        self._restore_combo_text(self.training_mode_combo, config.get("default_training_mode", TRAINING_MODES[0]))
        self._restore_combo_text(self.dataset_source_combo, config.get("default_dataset_source", DATASET_SOURCES[0]))
        self.patch_size_spin.setValue(int(config.get("patch_size", 256)))
        self.batch_size_spin.setValue(int(config.get("batch_size", 4)))
        self.epochs_spin.setValue(int(config.get("epochs", 10)))
        self.learning_rate_spin.setValue(float(config.get("learning_rate", 0.0001)))
        self.validation_split_spin.setValue(float(config.get("validation_split", 0.2)))
        self._restore_combo_text(self.image_normalization_combo, config.get("normalization_mode", "percentile"))
        self.notes_edit.setPlainText(config.get("notes", ""))
        mask_preparation = config.get("mask_preparation", {})
        self._restore_combo_text(
            self.mask_preparation_combo,
            mask_preparation.get("mode", "merge_nonzero_to_foreground"),
        )
        self.target_class_name_edit.setPlainText(mask_preparation.get("target_class_name", "foreground"))
        architecture = self.project.load_architecture_config()
        self._restore_combo_text(self.backend_combo, architecture.get("backend", "basic_unet"))
        self._restore_combo_text(self.spatial_dims_combo, architecture.get("spatial_dims", "2d"))
        self._restore_combo_text(self.preset_combo, architecture.get("preset", "standard_unet"))
        self.depth_spin.setValue(int(architecture.get("depth", 4)))
        self.base_channels_spin.setValue(int(architecture.get("base_channels", 32)))
        self._restore_combo_text(self.arch_normalization_combo, architecture.get("normalization", "batch"))
        self._restore_combo_text(self.upsampling_combo, architecture.get("upsampling", "transpose"))
        self.input_channels_spin.setValue(int(architecture.get("input_channels", 1)))
        self._restore_combo_text(self.output_mode_combo, architecture.get("output_mode", "binary"))
        self.num_classes_spin.setValue(int(architecture.get("num_classes", 2)))
        self.class_labels_edit.setPlainText(self._class_labels_to_text(architecture.get("class_labels", {})))
        self.threshold_spin.setValue(float(architecture.get("threshold", 0.5)))
        starting_weights = self.project.load_starting_weights_config()
        self._restore_combo_text(self.starting_weights_combo, starting_weights.get("mode", "latest_project_checkpoint"))
        imported_text = starting_weights.get("imported_model_id") or "No imported model selected"
        self.imported_model_label.setText(imported_text)
        self._show_dataset_source_warning(self.dataset_source_combo.currentText())
        self._sync_architecture_controls()
        self._refresh_compatibility_status()

    def _refresh_dataset_list(self) -> None:
        if self.project is None:
            return
        self.dataset_list.clear()
        for pair in self.project.dataset_pairs():
            text = f"{pair['pair_id']} | {pair.get('image_layer_name', '')} -> {pair.get('mask_layer_name', '')}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, pair["pair_id"])
            self.dataset_list.addItem(item)

    def _refresh_checkpoint_list(self) -> None:
        if self.project is None:
            return
        self.checkpoint_list.clear()
        for checkpoint in self.project.checkpoint_entries():
            parent = checkpoint.get("parent_checkpoint_id") or "none"
            arch = checkpoint.get("architecture", {})
            text = (
                f"{checkpoint['checkpoint_id']} | parent: {parent} | "
                f"{checkpoint.get('training_mode', '')} | {checkpoint.get('number_of_images', 0)} images | "
                f"{arch.get('spatial_dims', '?')} {arch.get('preset', '?')}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, checkpoint)
            self.checkpoint_list.addItem(item)

    def _refresh_prediction_list(self) -> None:
        if self.project is None:
            return
        self.prediction_list.clear()
        for path in sorted((self.project.root / "predictions").glob("*.tif")):
            self.prediction_list.addItem(str(path.relative_to(self.project.root)))

    def _current_mask_preparation(self) -> dict[str, Any]:
        return {
            "mode": self.mask_preparation_combo.currentText(),
            "target_class_name": self.target_class_name_edit.toPlainText().strip() or "foreground",
            "manual_label_map": {},
            "strict_multiclass_validation": True,
        }

    def _current_architecture(self) -> dict[str, Any]:
        output_mode = self.output_mode_combo.currentText()
        num_classes = self.num_classes_spin.value()
        output_channels = 1 if output_mode == "binary" else num_classes
        return {
            "backend": self.backend_combo.currentText(),
            "spatial_dims": self._clean_future_value(self.spatial_dims_combo.currentText()),
            "preset": self._clean_future_value(self.preset_combo.currentText()),
            "depth": self.depth_spin.value(),
            "base_channels": self.base_channels_spin.value(),
            "normalization": self.arch_normalization_combo.currentText(),
            "upsampling": self.upsampling_combo.currentText(),
            "input_channels": self.input_channels_spin.value(),
            "output_mode": output_mode,
            "num_classes": num_classes,
            "output_channels": output_channels,
            "class_labels": self._parse_class_labels(self.class_labels_edit.toPlainText(), output_mode, num_classes),
            "activation": "sigmoid" if output_mode == "binary" else "softmax",
            "loss": "bce_dice" if output_mode == "binary" else "cross_entropy_dice",
            "threshold": self.threshold_spin.value(),
        }

    def _current_starting_weights(self) -> dict[str, Any]:
        existing = self.project.load_starting_weights_config() if self.project else {}
        return {
            **existing,
            "mode": self.starting_weights_combo.currentText(),
            "checkpoint_id": self._selected_checkpoint_id_for_starting_weights(),
        }

    def _selected_checkpoint_id_for_starting_weights(self) -> str:
        selected = self._selected_checkpoint()
        if self.starting_weights_combo.currentText() == "selected_project_checkpoint" and selected:
            return selected.get("checkpoint_id", "")
        if self.starting_weights_combo.currentText() == "latest_project_checkpoint" and self.project:
            latest = self.project.latest_checkpoint()
            return latest.get("checkpoint_id", "") if latest else ""
        return ""

    def _sync_architecture_controls(self) -> None:
        is_binary = self.output_mode_combo.currentText() == "binary"
        self.num_classes_spin.setEnabled(not is_binary)
        if is_binary and self.num_classes_spin.value() != 2:
            self.num_classes_spin.blockSignals(True)
            self.num_classes_spin.setValue(2)
            self.num_classes_spin.blockSignals(False)
        output_channels = 1 if is_binary else self.num_classes_spin.value()
        self.output_channels_label.setText(str(output_channels))
        self.threshold_spin.setEnabled(is_binary)
        future_selected = "future" in self.spatial_dims_combo.currentText() or "future" in self.preset_combo.currentText()
        if future_selected:
            self.architecture_status_label.setText("Compatibility: invalid - selected U-Net option is reserved for a later implementation.")
        else:
            self.architecture_status_label.setText("Compatibility: valid")

    def _refresh_compatibility_status(self) -> None:
        if self.project is None:
            return
        architecture = self._current_architecture()
        mode = self.starting_weights_combo.currentText()
        if "future" in self.spatial_dims_combo.currentText() or "future" in self.preset_combo.currentText():
            message = "Compatibility: invalid - selected U-Net option is reserved for a later implementation."
            self.architecture_status_label.setText(message)
            self.compatibility_label.setText(message)
            self.train_button.setEnabled(False)
            return
        candidate = None
        if mode == "selected_project_checkpoint":
            candidate = self._selected_checkpoint()
            if candidate is None:
                message = "Compatibility: select a checkpoint to continue."
                self.compatibility_label.setText(message)
                self.train_button.setEnabled(False)
                return
        elif mode == "latest_project_checkpoint":
            candidate = self.project.latest_checkpoint()
        elif mode == "imported_pretrained_unet":
            imported_id = self.project.load_starting_weights_config().get("imported_model_id", "")
            candidate = self._imported_model_by_id(imported_id)
            if candidate is None:
                message = "Compatibility: import a supported U-Net checkpoint first."
                self.compatibility_label.setText(message)
                self.train_button.setEnabled(False)
                return

        if candidate and candidate.get("architecture"):
            compatible, reason = self.project.architecture_compatibility(architecture, candidate["architecture"])
            prefix = "Compatibility: valid" if compatible else "Compatibility: invalid"
            self.compatibility_label.setText(f"{prefix} - {reason}")
            self.train_button.setEnabled(compatible)
            return
        self.compatibility_label.setText("Compatibility: valid")
        self.train_button.setEnabled(self.project is not None)

    def _refresh_detected_labels(self) -> None:
        layer = self._layer_by_name(self.mask_layer_combo.currentText())
        if layer is None:
            self.detected_labels_label.setText("Detected labels: none")
            return
        labels = [int(value) for value in np.unique(np.asarray(layer.data))]
        preview = ", ".join(str(value) for value in labels[:20])
        if len(labels) > 20:
            preview += ", ..."
        self.detected_labels_label.setText(f"Detected labels: {preview}")

    def _imported_model_by_id(self, model_id: str) -> dict[str, Any] | None:
        if not model_id or self.project is None:
            return None
        for entry in self.project.load_model_registry().get("models", []):
            if entry.get("model_id") == model_id:
                return entry
        return None

    @staticmethod
    def _parse_class_labels(text: str, output_mode: str, num_classes: int) -> dict[str, str]:
        labels: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key.isdigit():
                labels[str(int(key))] = value.strip() or f"class_{key}"
        if output_mode == "binary":
            return {"0": labels.get("0", "background"), "1": labels.get("1", "foreground")}
        for index in range(num_classes):
            labels.setdefault(str(index), "background" if index == 0 else f"class_{index}")
        return {str(index): labels[str(index)] for index in range(num_classes)}

    @staticmethod
    def _class_labels_to_text(labels: dict[str, str]) -> str:
        if not labels:
            labels = {"0": "background", "1": "foreground"}
        return "\n".join(f"{key}: {labels[key]}" for key in sorted(labels, key=lambda item: int(item)))

    @staticmethod
    def _clean_future_value(value: str) -> str:
        return value.split(" - ", 1)[0]

    def _selected_pair_ids(self) -> list[str]:
        return [item.data(Qt.UserRole) for item in self.dataset_list.selectedItems()]

    def _selected_checkpoint(self) -> dict[str, Any] | None:
        selected = self.checkpoint_list.selectedItems()
        if not selected:
            return None
        return selected[0].data(Qt.UserRole)

    def _layer_by_name(self, name: str):
        if self.viewer is None or not name:
            return None
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    @staticmethod
    def _restore_combo_text(combo: QComboBox, text: str) -> None:
        if not text:
            return
        index = combo.findText(text)
        if index < 0:
            for item_index in range(combo.count()):
                if combo.itemText(item_index).split(" - ", 1)[0] == text:
                    index = item_index
                    break
        if index >= 0:
            combo.setCurrentIndex(index)

    def _show_dataset_source_warning(self, text: str) -> None:
        self.new_masks_warning.setVisible(text == "New masks since last checkpoint")
