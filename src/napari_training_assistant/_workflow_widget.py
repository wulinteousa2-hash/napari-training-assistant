"""Tabbed workflow UI for persistent SAM3-to-U-Net training projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_training_assistant.project import TrainingProject


TRAINING_MODES = (
    ("Continue latest", "Continue from latest checkpoint"),
    ("Continue selected", "Continue from selected checkpoint"),
    ("From scratch", "Retrain from scratch"),
)
DATASET_SOURCES = (
    ("All accepted masks", "All accepted masks"),
    ("New masks only", "New masks since last checkpoint"),
    ("Selected masks", "Selected masks only"),
)
ARCHITECTURE_BACKENDS = (("Basic U-Net", "basic_unet"),)
SPATIAL_DIMS = (("2D", "2d"), ("3D U-Net - future", "3d"))
UNET_PRESETS = (
    ("Standard U-Net", "standard_unet"),
    ("ResUNet - future", "resunet_2d"),
    ("Attention U-Net - future", "attention_unet_2d"),
)
UNET_NORMALIZATION = (
    ("BatchNorm", "batch"),
    ("InstanceNorm", "instance"),
    ("GroupNorm", "group"),
    ("None", "none"),
)
UNET_UPSAMPLING = (
    ("Transposed convolution", "transpose"),
    ("Bilinear + convolution", "bilinear"),
)
OUTPUT_MODES = (
    ("Binary foreground/background", "binary"),
    ("Multiclass semantic labels", "multiclass"),
)
STARTING_WEIGHT_MODES = (
    ("Scratch", "scratch"),
    ("Latest project checkpoint", "latest_project_checkpoint"),
    ("Selected project checkpoint", "selected_project_checkpoint"),
    ("Imported pretrained U-Net", "imported_pretrained_unet"),
)
MASK_PREPARATION_MODES = (
    ("Merge all labels into foreground", "merge_nonzero_to_foreground"),
    ("Keep labels as multiclass", "keep_labels_as_multiclass"),
)
SAM3_MODES = (
    ("2D box", "2d_box"),
    ("2D points", "2d_points"),
    ("Live points", "live_points"),
    ("2D exemplar", "2d_exemplar"),
    ("3D / multiplex", "3d_multiplex"),
)
SAM3_DEVICES = (
    ("Auto", "auto"),
    ("CUDA", "cuda"),
    ("CPU for 2D", "cpu"),
)


class TrainingAssistantWidget(QWidget):
    """Tabbed workflow dock widget."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self.project: TrainingProject | None = None
        self._build_ui()
        self._set_project_actions_enabled(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._build_project_bar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_sam3_tab(), "SAM3")
        self.tabs.addTab(self._build_dataset_tab(), "Dataset")
        self.tabs.addTab(self._build_train_tab(), "Train")
        self.tabs.addTab(self._build_checkpoints_tab(), "Checkpoints")
        self.tabs.addTab(self._build_predict_tab(), "Predict")
        self.tabs.addTab(self._build_advanced_tab(), "Advanced")
        layout.addWidget(self.tabs)

    def _build_sam3_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_box = QGroupBox("SAM3 model")
        model_layout = QGridLayout(model_box)
        self.sam3_mode_combo = QComboBox()
        self._add_options(self.sam3_mode_combo, SAM3_MODES)
        self.sam3_mode_combo.currentTextChanged.connect(self.on_sam3_mode_changed)
        self.sam3_device_combo = QComboBox()
        self._add_options(self.sam3_device_combo, SAM3_DEVICES)
        self.sam3_device_combo.currentTextChanged.connect(self.persist_sam3_settings)
        self.sam3_2d_model_edit = QLineEdit()
        self.sam3_2d_model_edit.setPlaceholderText("Folder containing sam3.pt or model.safetensors")
        self.sam3_2d_model_edit.editingFinished.connect(self.persist_sam3_settings)
        self.sam3_3d_model_edit = QLineEdit()
        self.sam3_3d_model_edit.setPlaceholderText("Folder containing sam3.1_multiplex.pt")
        self.sam3_3d_model_edit.editingFinished.connect(self.persist_sam3_settings)
        self.sam3_2d_model_button = QPushButton("Select 2D")
        self.sam3_2d_model_button.clicked.connect(lambda: self.browse_sam3_model_dir("2d"))
        self.sam3_3d_model_button = QPushButton("Select 3D")
        self.sam3_3d_model_button.clicked.connect(lambda: self.browse_sam3_model_dir("3d"))
        self.sam3_status_label = QLabel("SAM3 model not configured.")
        self.sam3_status_label.setWordWrap(True)
        model_layout.addWidget(QLabel("Mode"), 0, 0)
        model_layout.addWidget(self.sam3_mode_combo, 0, 1)
        model_layout.addWidget(QLabel("Device"), 0, 2)
        model_layout.addWidget(self.sam3_device_combo, 0, 3)
        model_layout.addWidget(QLabel("2D model"), 1, 0)
        model_layout.addWidget(self.sam3_2d_model_edit, 1, 1, 1, 2)
        model_layout.addWidget(self.sam3_2d_model_button, 1, 3)
        model_layout.addWidget(QLabel("3D model"), 2, 0)
        model_layout.addWidget(self.sam3_3d_model_edit, 2, 1, 1, 2)
        model_layout.addWidget(self.sam3_3d_model_button, 2, 3)
        model_layout.addWidget(self.sam3_status_label, 3, 0, 1, 4)
        model_layout.setColumnStretch(1, 1)
        layout.addWidget(model_box)

        prompt_box = QGroupBox("Prompt")
        prompt_layout = QGridLayout(prompt_box)
        self.sam3_image_layer_combo = QComboBox()
        self.sam3_image_layer_combo.currentTextChanged.connect(self.persist_sam3_settings)
        self.sam3_refresh_layers_button = QPushButton("Refresh")
        self.sam3_refresh_layers_button.clicked.connect(self.refresh_layer_choices)
        self.sam3_prepare_prompt_button = QPushButton("Prepare prompt layer")
        self.sam3_prepare_prompt_button.clicked.connect(self.prepare_sam3_prompt_layer)
        self.sam3_prompt_status_label = QLabel("Choose a mode to prepare prompt layers.")
        self.sam3_prompt_status_label.setWordWrap(True)
        prompt_layout.addWidget(QLabel("Image layer"), 0, 0)
        prompt_layout.addWidget(self.sam3_image_layer_combo, 0, 1)
        prompt_layout.addWidget(self.sam3_refresh_layers_button, 0, 2)
        prompt_layout.addWidget(self.sam3_prepare_prompt_button, 1, 0, 1, 3)
        prompt_layout.addWidget(self.sam3_prompt_status_label, 2, 0, 1, 3)
        prompt_layout.setColumnStretch(1, 1)
        layout.addWidget(prompt_box)

        preview_box = QGroupBox("Preview")
        preview_layout = QGridLayout(preview_box)
        self.sam3_run_preview_button = QPushButton("Run preview")
        self.sam3_run_preview_button.clicked.connect(self.run_sam3_preview)
        self.sam3_clear_preview_button = QPushButton("Clear preview")
        self.sam3_clear_preview_button.clicked.connect(self.clear_sam3_preview)
        self.sam3_accept_preview_button = QPushButton("Accept preview to Dataset")
        self.sam3_accept_preview_button.clicked.connect(self.accept_sam3_preview_to_dataset)
        self.sam3_preview_layer_label = QLabel("Preview labels: SAM3 preview labels")
        self.sam3_preview_layer_label.setWordWrap(True)
        preview_layout.addWidget(self.sam3_run_preview_button, 0, 0)
        preview_layout.addWidget(self.sam3_clear_preview_button, 0, 1)
        preview_layout.addWidget(self.sam3_accept_preview_button, 1, 0, 1, 2)
        preview_layout.addWidget(self.sam3_preview_layer_label, 2, 0, 1, 2)
        layout.addWidget(preview_box)
        layout.addStretch(1)
        return tab

    def _build_project_bar(self) -> QWidget:
        box = QGroupBox("Project")
        layout = QGridLayout(box)
        layout.setColumnStretch(1, 1)
        self.select_project_button = QPushButton("Select / Create Project")
        self.select_project_button.clicked.connect(self.select_project_folder)
        self.project_path_label = QLabel("No project selected")
        self.project_path_label.setWordWrap(True)
        self.project_state_label = QLabel("missing")
        self.dataset_count_label = QLabel("0")
        self.latest_checkpoint_label = QLabel("None")
        self.latest_benchmark_label = QLabel("None")
        layout.addWidget(self.select_project_button, 0, 0)
        layout.addWidget(self.project_path_label, 0, 1, 1, 5)
        layout.addWidget(QLabel("State"), 1, 0)
        layout.addWidget(self.project_state_label, 1, 1)
        layout.addWidget(QLabel("Dataset"), 1, 2)
        layout.addWidget(self.dataset_count_label, 1, 3)
        layout.addWidget(QLabel("Latest"), 1, 4)
        layout.addWidget(self.latest_checkpoint_label, 1, 5)
        layout.addWidget(QLabel("Benchmark"), 2, 0)
        layout.addWidget(self.latest_benchmark_label, 2, 1, 1, 5)
        return box

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QGroupBox("Add accepted mask")
        controls_layout = QGridLayout(controls)
        self.image_layer_combo = QComboBox()
        self.mask_layer_combo = QComboBox()
        self.mask_layer_combo.currentTextChanged.connect(lambda _text: self._refresh_detected_labels())
        self.refresh_layers_button = QPushButton("Refresh")
        self.refresh_layers_button.clicked.connect(self.refresh_layer_choices)
        self.add_mask_button = QPushButton("Add to dataset")
        self.add_mask_button.clicked.connect(self.add_current_mask_pair)
        controls_layout.addWidget(QLabel("Image layer"), 0, 0)
        controls_layout.addWidget(self.image_layer_combo, 0, 1)
        controls_layout.addWidget(QLabel("Mask layer"), 1, 0)
        controls_layout.addWidget(self.mask_layer_combo, 1, 1)
        controls_layout.addWidget(self.refresh_layers_button, 0, 2)
        controls_layout.addWidget(self.add_mask_button, 1, 2)
        controls_layout.setColumnStretch(1, 1)
        layout.addWidget(controls)

        preparation = QGroupBox("Mask preparation")
        preparation_layout = QFormLayout(preparation)
        self.mask_preparation_combo = QComboBox()
        self._add_options(self.mask_preparation_combo, MASK_PREPARATION_MODES)
        self.mask_preparation_combo.currentTextChanged.connect(self.persist_mask_preparation)
        self.target_class_name_edit = QTextEdit()
        self.target_class_name_edit.setFixedHeight(30)
        self.target_class_name_edit.setPlainText("foreground")
        self.target_class_name_edit.textChanged.connect(self.persist_mask_preparation)
        self.detected_labels_label = QLabel("Detected labels: none")
        self.detected_labels_label.setWordWrap(True)
        self.mask_cleanup_note = QLabel(
            "Binary masks save all nonzero instance labels as one foreground class."
        )
        self.mask_cleanup_note.setWordWrap(True)
        preparation_layout.addRow("Mode", self.mask_preparation_combo)
        preparation_layout.addRow("Target class", self.target_class_name_edit)
        preparation_layout.addRow("", self.detected_labels_label)
        preparation_layout.addRow("", self.mask_cleanup_note)
        layout.addWidget(preparation)

        self.dataset_table = self._make_table(("Pair", "Image", "Mask", "Class", "Labels", "Used"))
        self.dataset_table.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.dataset_table, 1)
        return tab

    def _build_train_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        settings = QGroupBox("Train U-Net")
        settings_layout = QFormLayout(settings)
        self.training_mode_combo = QComboBox()
        self._add_options(self.training_mode_combo, TRAINING_MODES)
        self.dataset_source_combo = QComboBox()
        self._add_options(self.dataset_source_combo, DATASET_SOURCES)
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
        self.starting_weights_combo = QComboBox()
        self._add_options(self.starting_weights_combo, STARTING_WEIGHT_MODES)
        self.starting_weights_combo.currentTextChanged.connect(self.persist_starting_weights_settings)
        self.compatibility_label = QLabel("Compatibility: valid")
        self.compatibility_label.setWordWrap(True)
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
        settings_layout.addRow("Training mode", self.training_mode_combo)
        settings_layout.addRow("Dataset source", self.dataset_source_combo)
        settings_layout.addRow("", self.new_masks_warning)
        settings_layout.addRow("Starting point", self.starting_weights_combo)
        settings_layout.addRow("Patch size", self.patch_size_spin)
        settings_layout.addRow("Batch size", self.batch_size_spin)
        settings_layout.addRow("Epochs", self.epochs_spin)
        settings_layout.addRow("Learning rate", self.learning_rate_spin)
        settings_layout.addRow("Validation split", self.validation_split_spin)
        settings_layout.addRow("Image normalization", self.image_normalization_combo)
        settings_layout.addRow("", self.compatibility_label)
        layout.addWidget(settings)
        self.train_summary_label = QLabel("No project selected.")
        self.train_summary_label.setWordWrap(True)
        layout.addWidget(self.train_summary_label)
        actions = QHBoxLayout()
        self.train_button = QPushButton("Train U-Net")
        self.train_button.clicked.connect(self.train_unet)
        self.retrain_button = QPushButton("Retrain from scratch")
        self.retrain_button.clicked.connect(self.retrain_from_scratch)
        actions.addWidget(self.train_button)
        actions.addWidget(self.retrain_button)
        layout.addLayout(actions)
        layout.addStretch(1)
        return tab

    def _build_checkpoints_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.checkpoint_summary_label = QLabel("No checkpoints yet.")
        self.checkpoint_summary_label.setWordWrap(True)
        layout.addWidget(self.checkpoint_summary_label)
        self.checkpoint_table = self._make_table(("Run", "Mode", "Parent", "Images", "Architecture", "Summary"))
        self.checkpoint_table.itemSelectionChanged.connect(self._refresh_compatibility_status)
        layout.addWidget(self.checkpoint_table, 1)
        self.use_selected_checkpoint_button = QPushButton("Use selected checkpoint for training")
        self.use_selected_checkpoint_button.clicked.connect(self.use_selected_checkpoint_for_training)
        layout.addWidget(self.use_selected_checkpoint_button)
        return tab

    def _build_predict_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        box = QGroupBox("Prediction outputs")
        form = QFormLayout(box)
        self.save_predictions_check = QCheckBox("Save predictions to project")
        self.save_predictions_check.setChecked(True)
        self.prediction_layer_combo = QComboBox()
        self.save_prediction_button = QPushButton("Save selected prediction")
        self.save_prediction_button.clicked.connect(self.save_selected_prediction)
        form.addRow("", self.save_predictions_check)
        form.addRow("Prediction layer", self.prediction_layer_combo)
        form.addRow("", self.save_prediction_button)
        layout.addWidget(box)
        self.prediction_table = self._make_table(("Output", "Path"))
        layout.addWidget(self.prediction_table, 1)
        return tab

    def _build_advanced_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_architecture_section())
        layout.addWidget(self._build_starting_weights_section())
        notes = QGroupBox("Project notes")
        notes_layout = QFormLayout(notes)
        self.notes_edit = QTextEdit()
        self.notes_edit.setFixedHeight(70)
        self.notes_edit.textChanged.connect(self.persist_training_settings)
        self.save_settings_button = QPushButton("Save settings")
        self.save_settings_button.clicked.connect(self.persist_all_settings)
        notes_layout.addRow("Notes", self.notes_edit)
        notes_layout.addRow("", self.save_settings_button)
        layout.addWidget(notes)
        layout.addStretch(1)
        return tab

    def _build_architecture_section(self) -> QGroupBox:
        box = QGroupBox("U-Net architecture")
        layout = QFormLayout(box)
        self.backend_combo = QComboBox()
        self._add_options(self.backend_combo, ARCHITECTURE_BACKENDS)
        self.spatial_dims_combo = QComboBox()
        self._add_options(self.spatial_dims_combo, SPATIAL_DIMS)
        self.preset_combo = QComboBox()
        self._add_options(self.preset_combo, UNET_PRESETS)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(2, 6)
        self.base_channels_spin = QSpinBox()
        self.base_channels_spin.setRange(8, 256)
        self.base_channels_spin.setSingleStep(8)
        self.arch_normalization_combo = QComboBox()
        self._add_options(self.arch_normalization_combo, UNET_NORMALIZATION)
        self.upsampling_combo = QComboBox()
        self._add_options(self.upsampling_combo, UNET_UPSAMPLING)
        self.input_channels_spin = QSpinBox()
        self.input_channels_spin.setRange(1, 64)
        self.output_mode_combo = QComboBox()
        self._add_options(self.output_mode_combo, OUTPUT_MODES)
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 1024)
        self.output_channels_label = QLabel("1")
        self.class_labels_edit = QTextEdit()
        self.class_labels_edit.setFixedHeight(62)
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
        box = QGroupBox("Imported starting weights")
        layout = QFormLayout(box)
        self.import_model_button = QPushButton("Import .pt / .pth")
        self.import_model_button.clicked.connect(self.import_pretrained_model)
        self.imported_model_label = QLabel("No imported model selected")
        self.imported_model_label.setWordWrap(True)
        layout.addRow("", self.import_model_button)
        layout.addRow("Imported model", self.imported_model_label)
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
        self.project_path_label.setText(self._short_path(project.root))
        self.project_path_label.setToolTip(str(project.root))
        self.dataset_count_label.setText(str(project.dataset_count()))
        self.latest_checkpoint_label.setText(
            latest["checkpoint_id"] if latest else config.get("latest_checkpoint_path") or "None"
        )
        self.latest_benchmark_label.setText(project.latest_benchmark_summary() or "None")
        self.project_state_label.setText(state.label)
        self._refresh_dataset_table()
        self._refresh_checkpoint_table()
        self._refresh_prediction_table()
        self._refresh_train_summary()
        self._refresh_compatibility_status()

    def refresh_layer_choices(self) -> None:
        layer_names = []
        if self.viewer is not None:
            layer_names = [layer.name for layer in self.viewer.layers]
        current_sam3_image = self.sam3_image_layer_combo.currentText()
        current_image = self.image_layer_combo.currentText()
        current_mask = self.mask_layer_combo.currentText()
        current_prediction = self.prediction_layer_combo.currentText()
        for combo in (
            self.sam3_image_layer_combo,
            self.image_layer_combo,
            self.mask_layer_combo,
            self.prediction_layer_combo,
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(layer_names)
            combo.blockSignals(False)
        self._restore_combo_text(self.sam3_image_layer_combo, current_sam3_image)
        self._restore_combo_text(self.image_layer_combo, current_image)
        self._restore_combo_text(self.mask_layer_combo, current_mask)
        self._restore_combo_text(self.prediction_layer_combo, current_prediction)
        self._refresh_detected_labels()

    def browse_sam3_model_dir(self, kind: str) -> None:
        project = self.require_project()
        if project is None:
            return
        path = QFileDialog.getExistingDirectory(self, "Select SAM3 model folder")
        if not path:
            return
        if kind == "2d":
            self.sam3_2d_model_edit.setText(path)
        else:
            self.sam3_3d_model_edit.setText(path)
        self.persist_sam3_settings()

    def on_sam3_mode_changed(self, *args: Any) -> None:
        self.persist_sam3_settings()
        self.prepare_sam3_prompt_layer()

    def persist_sam3_settings(self, *args: Any) -> None:
        if self.project is None:
            return
        self.project.save_sam3_config(self._current_sam3_config())
        self._refresh_sam3_status()

    def prepare_sam3_prompt_layer(self) -> None:
        if self.viewer is None:
            return
        config = self._current_sam3_config()
        mode = config["default_mode"]
        image_layer = self._layer_by_name(self.sam3_image_layer_combo.currentText())
        ndim = int(getattr(image_layer, "ndim", 2) or 2) if image_layer is not None else 2
        ndim = max(2, ndim)
        if mode == "2d_points":
            layer = self._ensure_points_layer(config["points_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add")
            self.sam3_prompt_status_label.setText("Ready: add positive/negative points, then run preview.")
        elif mode == "live_points":
            layer = self._ensure_points_layer(config["live_points_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add")
            self.sam3_prompt_status_label.setText("Ready: live point layer selected. Preview will update once inference is connected.")
        elif mode == "2d_box":
            layer = self._ensure_shapes_layer(config["boxes_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add_rectangle")
            self.sam3_prompt_status_label.setText("Ready: draw one or more boxes, then run preview.")
        elif mode == "2d_exemplar":
            layer = self._ensure_shapes_layer(config["exemplar_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add_rectangle")
            self.sam3_prompt_status_label.setText("Ready: draw exemplar boxes around example objects, then run preview.")
        else:
            layer = self._ensure_points_layer(config["multiplex_prompt_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add")
            self.sam3_prompt_status_label.setText("Ready: add 3D/multiplex seed prompts. SAM3.1 CUDA inference is wired later.")
        self.persist_sam3_settings()

    def run_sam3_preview(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if self.viewer is None:
            QMessageBox.warning(self, "No viewer", "A napari viewer is required for SAM3 preview.")
            return
        image_layer = self._layer_by_name(self.sam3_image_layer_combo.currentText())
        if image_layer is None:
            QMessageBox.warning(self, "Image layer required", "Select an image layer for SAM3 preview.")
            return
        config = self._current_sam3_config()
        status_ok, status = self._sam3_model_status(config)
        if not status_ok:
            QMessageBox.warning(self, "SAM3 model folder required", status)
            return
        preview = self._ensure_preview_labels_layer(image_layer, config["preview_labels_layer_name"])
        self._activate_layer(preview, "paint")
        self.sam3_prompt_status_label.setText(
            "SAM3 preview inference is not wired yet. Prompt and preview layers are ready."
        )
        self.refresh_layer_choices()

    def clear_sam3_preview(self) -> None:
        if self.viewer is None:
            return
        config = self._current_sam3_config()
        for name in (config["preview_labels_layer_name"], config["propagated_labels_layer_name"]):
            layer = self._layer_by_name(name)
            if layer is not None:
                self.viewer.layers.remove(layer)
        self.refresh_layer_choices()

    def accept_sam3_preview_to_dataset(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if self.viewer is None:
            QMessageBox.warning(self, "No viewer", "A napari viewer is required to accept SAM3 preview masks.")
            return
        config = self._current_sam3_config()
        image_layer = self._layer_by_name(self.sam3_image_layer_combo.currentText())
        preview_layer = self._layer_by_name(config["preview_labels_layer_name"])
        if image_layer is None or preview_layer is None:
            QMessageBox.warning(self, "Preview required", "Run or create a SAM3 preview labels layer before accepting.")
            return
        project.add_pair(
            np.asarray(image_layer.data),
            np.asarray(preview_layer.data),
            image_layer_name=image_layer.name,
            mask_layer_name=preview_layer.name,
            mask_preparation=self._current_mask_preparation(),
            metadata={"source": "sam3_preview", "sam3": config},
        )
        self.image_layer_combo.setCurrentText(image_layer.name)
        self.mask_layer_combo.setCurrentText(preview_layer.name)
        self.refresh_project_summary()
        self.tabs.setCurrentIndex(1)

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

    def persist_all_settings(self, *args: Any) -> None:
        self.persist_training_settings()
        self.persist_sam3_settings()
        self.persist_mask_preparation()
        self.persist_architecture_settings()
        self.persist_starting_weights_settings()

    def persist_training_settings(self, *args: Any) -> None:
        if self.project is None:
            return
        self.project.update_settings(
            default_training_mode=self._combo_data(self.training_mode_combo),
            default_dataset_source=self._combo_data(self.dataset_source_combo),
            patch_size=self.patch_size_spin.value(),
            batch_size=self.batch_size_spin.value(),
            epochs=self.epochs_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            validation_split=self.validation_split_spin.value(),
            normalization_mode=self.image_normalization_combo.currentText(),
            notes=self.notes_edit.toPlainText(),
        )
        self._refresh_train_summary()

    def persist_mask_preparation(self, *args: Any) -> None:
        if self.project is None:
            return
        self.project.update_settings(mask_preparation=self._current_mask_preparation())
        self._refresh_detected_labels()

    def persist_architecture_settings(self, *args: Any) -> None:
        self._sync_architecture_controls()
        if self.project is None:
            return
        self.project.save_architecture_config(self._current_architecture())
        self._refresh_compatibility_status()
        self._refresh_train_summary()

    def persist_starting_weights_settings(self, *args: Any) -> None:
        if self.project is None:
            return
        self.project.save_starting_weights_config(self._current_starting_weights())
        self._refresh_compatibility_status()
        self._refresh_train_summary()

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
        self.starting_weights_combo.setCurrentIndex(self._find_combo_data(self.starting_weights_combo, "imported_pretrained_unet"))
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

    def use_selected_checkpoint_for_training(self) -> None:
        if self._selected_checkpoint() is None:
            QMessageBox.warning(self, "Checkpoint required", "Select a checkpoint first.")
            return
        self.starting_weights_combo.setCurrentIndex(self._find_combo_data(self.starting_weights_combo, "selected_project_checkpoint"))
        self.training_mode_combo.setCurrentIndex(self._find_combo_data(self.training_mode_combo, "Continue from selected checkpoint"))
        self.tabs.setCurrentIndex(1)
        self.persist_starting_weights_settings()

    def train_unet(self) -> None:
        project = self.require_project()
        if project is None:
            return
        self.persist_all_settings()
        config = project.load_config()
        architecture = project.load_architecture_config()
        starting_weights = project.load_starting_weights_config()
        if architecture.get("spatial_dims") != "2d":
            QMessageBox.warning(
                self,
                "3D U-Net not implemented",
                "3D U-Net is reserved in the project schema but not implemented yet.",
            )
            return
        selected_checkpoint = self._selected_checkpoint()
        training_mode_label = self._combo_data(self.training_mode_combo)
        parent_checkpoint_id = ""
        if training_mode_label == "Continue from latest checkpoint":
            latest = project.latest_checkpoint()
            parent_checkpoint_id = latest.get("checkpoint_id", "") if latest else ""
            training_mode = "continue"
        elif training_mode_label == "Continue from selected checkpoint":
            if not selected_checkpoint:
                QMessageBox.warning(self, "Checkpoint required", "Select a checkpoint to continue from.")
                self.tabs.setCurrentIndex(2)
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
            self._combo_data(self.dataset_source_combo),
            selected_pair_ids=selected_pair_ids,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        if not pairs:
            QMessageBox.warning(self, "Dataset required", "No dataset pairs match the selected dataset source.")
            self.tabs.setCurrentIndex(0)
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
        self.tabs.setCurrentIndex(2)

    def retrain_from_scratch(self) -> None:
        self.training_mode_combo.setCurrentIndex(self._find_combo_data(self.training_mode_combo, "Retrain from scratch"))
        self.starting_weights_combo.setCurrentIndex(self._find_combo_data(self.starting_weights_combo, "scratch"))
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
            self.tabs,
            self.sam3_mode_combo,
            self.sam3_device_combo,
            self.sam3_2d_model_edit,
            self.sam3_3d_model_edit,
            self.sam3_2d_model_button,
            self.sam3_3d_model_button,
            self.sam3_image_layer_combo,
            self.sam3_refresh_layers_button,
            self.sam3_prepare_prompt_button,
            self.sam3_run_preview_button,
            self.sam3_clear_preview_button,
            self.sam3_accept_preview_button,
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
            self.checkpoint_table,
            self.use_selected_checkpoint_button,
            self.train_button,
            self.retrain_button,
            self.save_predictions_check,
            self.prediction_layer_combo,
            self.save_prediction_button,
            self.dataset_table,
            self.prediction_table,
        ):
            widget.setEnabled(enabled)

    def _load_settings_from_project(self) -> None:
        if self.project is None:
            return
        config = self.project.load_config()
        self._restore_combo_data(self.training_mode_combo, config.get("default_training_mode", "Continue from latest checkpoint"))
        self._restore_combo_data(self.dataset_source_combo, config.get("default_dataset_source", "All accepted masks"))
        self.patch_size_spin.setValue(int(config.get("patch_size", 256)))
        self.batch_size_spin.setValue(int(config.get("batch_size", 4)))
        self.epochs_spin.setValue(int(config.get("epochs", 10)))
        self.learning_rate_spin.setValue(float(config.get("learning_rate", 0.0001)))
        self.validation_split_spin.setValue(float(config.get("validation_split", 0.2)))
        self._restore_combo_text(self.image_normalization_combo, config.get("normalization_mode", "percentile"))
        self.notes_edit.setPlainText(config.get("notes", ""))
        mask_preparation = config.get("mask_preparation", {})
        self._restore_combo_data(
            self.mask_preparation_combo,
            mask_preparation.get("mode", "merge_nonzero_to_foreground"),
        )
        self.target_class_name_edit.setPlainText(mask_preparation.get("target_class_name", "foreground"))
        architecture = self.project.load_architecture_config()
        self._restore_combo_data(self.backend_combo, architecture.get("backend", "basic_unet"))
        self._restore_combo_data(self.spatial_dims_combo, architecture.get("spatial_dims", "2d"))
        self._restore_combo_data(self.preset_combo, architecture.get("preset", "standard_unet"))
        self.depth_spin.setValue(int(architecture.get("depth", 4)))
        self.base_channels_spin.setValue(int(architecture.get("base_channels", 32)))
        self._restore_combo_data(self.arch_normalization_combo, architecture.get("normalization", "batch"))
        self._restore_combo_data(self.upsampling_combo, architecture.get("upsampling", "transpose"))
        self.input_channels_spin.setValue(int(architecture.get("input_channels", 1)))
        self._restore_combo_data(self.output_mode_combo, architecture.get("output_mode", "binary"))
        self.num_classes_spin.setValue(int(architecture.get("num_classes", 2)))
        self.class_labels_edit.setPlainText(self._class_labels_to_text(architecture.get("class_labels", {})))
        self.threshold_spin.setValue(float(architecture.get("threshold", 0.5)))
        starting_weights = self.project.load_starting_weights_config()
        self._restore_combo_data(self.starting_weights_combo, starting_weights.get("mode", "latest_project_checkpoint"))
        imported_text = starting_weights.get("imported_model_id") or "No imported model selected"
        self.imported_model_label.setText(imported_text)
        sam3_config = self.project.load_sam3_config()
        self._restore_combo_data(self.sam3_mode_combo, sam3_config.get("default_mode", "2d_box"))
        self._restore_combo_data(self.sam3_device_combo, sam3_config.get("device", "auto"))
        self.sam3_2d_model_edit.setText(sam3_config.get("sam3_2d_model_dir", ""))
        self.sam3_3d_model_edit.setText(sam3_config.get("sam3_3d_model_dir", ""))
        self._restore_combo_text(self.sam3_image_layer_combo, sam3_config.get("last_image_layer", ""))
        self._show_dataset_source_warning(self._combo_data(self.dataset_source_combo))
        self._sync_architecture_controls()
        self._refresh_compatibility_status()
        self._refresh_sam3_status()

    def _refresh_dataset_table(self) -> None:
        if self.project is None:
            return
        pairs = self.project.dataset_pairs()
        self.dataset_table.setRowCount(len(pairs))
        for row, pair in enumerate(pairs):
            preparation = pair.get("mask_preparation", {})
            used = ", ".join(pair.get("used_in_checkpoints", [])) or "no"
            values = (
                self._short_id(pair.get("pair_id", "")),
                pair.get("image_layer_name", ""),
                pair.get("mask_layer_name", ""),
                preparation.get("target_class_name", ""),
                self._labels_summary(preparation.get("saved_labels", [])),
                used,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, pair.get("pair_id", ""))
                if column == 0:
                    item.setToolTip(pair.get("pair_id", ""))
                self.dataset_table.setItem(row, column, item)
        self.dataset_table.resizeColumnsToContents()

    def _refresh_checkpoint_table(self) -> None:
        if self.project is None:
            return
        checkpoints = self.project.checkpoint_entries()
        self.checkpoint_table.setRowCount(len(checkpoints))
        latest = self.project.latest_checkpoint()
        latest_id = latest.get("checkpoint_id", "") if latest else ""
        for row, checkpoint in enumerate(checkpoints):
            arch = checkpoint.get("architecture", {})
            values = (
                checkpoint.get("checkpoint_id", ""),
                checkpoint.get("training_mode", ""),
                checkpoint.get("parent_checkpoint_id", "") or "none",
                checkpoint.get("number_of_images", 0),
                f"{arch.get('spatial_dims', '?')} {arch.get('preset', '?')}",
                checkpoint.get("benchmark_summary", ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, checkpoint)
                self.checkpoint_table.setItem(row, column, item)
        self.checkpoint_table.resizeColumnsToContents()
        self.checkpoint_summary_label.setText(
            f"Latest: {latest_id or 'None'} | Total checkpoints: {len(checkpoints)}"
        )

    def _refresh_prediction_table(self) -> None:
        if self.project is None:
            return
        outputs = sorted((self.project.root / "predictions").glob("*.tif"))
        self.prediction_table.setRowCount(len(outputs))
        for row, path in enumerate(outputs):
            relative = path.relative_to(self.project.root)
            self.prediction_table.setItem(row, 0, QTableWidgetItem(path.name))
            self.prediction_table.setItem(row, 1, QTableWidgetItem(str(relative)))
        self.prediction_table.resizeColumnsToContents()

    def _refresh_train_summary(self) -> None:
        if self.project is None:
            self.train_summary_label.setText("No project selected.")
            return
        latest = self.project.latest_checkpoint()
        architecture = self._current_architecture()
        self.train_summary_label.setText(
            "Ready state: "
            f"{self.project.dataset_count()} dataset pairs | "
            f"{self._combo_data(self.training_mode_combo)} | "
            f"{self._combo_data(self.dataset_source_combo)} | "
            f"{architecture['spatial_dims']} {architecture['preset']} | "
            f"latest {latest.get('checkpoint_id', 'None') if latest else 'None'}"
        )

    def _current_mask_preparation(self) -> dict[str, Any]:
        return {
            "mode": self._combo_data(self.mask_preparation_combo),
            "target_class_name": self.target_class_name_edit.toPlainText().strip() or "foreground",
            "manual_label_map": {},
            "strict_multiclass_validation": True,
        }

    def _current_sam3_config(self) -> dict[str, Any]:
        existing = self.project.load_sam3_config() if self.project else {}
        return {
            **existing,
            "default_mode": self._combo_data(self.sam3_mode_combo),
            "sam3_2d_model_dir": self.sam3_2d_model_edit.text().strip(),
            "sam3_3d_model_dir": self.sam3_3d_model_edit.text().strip(),
            "device": self._combo_data(self.sam3_device_combo),
            "last_image_layer": self.sam3_image_layer_combo.currentText(),
        }

    def _current_architecture(self) -> dict[str, Any]:
        output_mode = self._combo_data(self.output_mode_combo)
        num_classes = self.num_classes_spin.value()
        output_channels = 1 if output_mode == "binary" else num_classes
        return {
            "backend": self._combo_data(self.backend_combo),
            "spatial_dims": self._combo_data(self.spatial_dims_combo),
            "preset": self._combo_data(self.preset_combo),
            "depth": self.depth_spin.value(),
            "base_channels": self.base_channels_spin.value(),
            "normalization": self._combo_data(self.arch_normalization_combo),
            "upsampling": self._combo_data(self.upsampling_combo),
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
            "mode": self._combo_data(self.starting_weights_combo),
            "checkpoint_id": self._selected_checkpoint_id_for_starting_weights(),
        }

    def _selected_checkpoint_id_for_starting_weights(self) -> str:
        selected = self._selected_checkpoint()
        mode = self._combo_data(self.starting_weights_combo)
        if mode == "selected_project_checkpoint" and selected:
            return selected.get("checkpoint_id", "")
        if mode == "latest_project_checkpoint" and self.project:
            latest = self.project.latest_checkpoint()
            return latest.get("checkpoint_id", "") if latest else ""
        return ""

    def _sync_architecture_controls(self) -> None:
        is_binary = self._combo_data(self.output_mode_combo) == "binary"
        self.num_classes_spin.setEnabled(not is_binary)
        if is_binary and self.num_classes_spin.value() != 2:
            self.num_classes_spin.blockSignals(True)
            self.num_classes_spin.setValue(2)
            self.num_classes_spin.blockSignals(False)
        output_channels = 1 if is_binary else self.num_classes_spin.value()
        self.output_channels_label.setText(str(output_channels))
        self.threshold_spin.setEnabled(is_binary)
        future_selected = self._combo_data(self.spatial_dims_combo) != "2d" or self._combo_data(self.preset_combo) != "standard_unet"
        if future_selected:
            self.architecture_status_label.setText("Compatibility: invalid - selected U-Net option is reserved for a later implementation.")
        else:
            self.architecture_status_label.setText("Compatibility: valid")

    def _refresh_compatibility_status(self) -> None:
        if self.project is None:
            return
        architecture = self._current_architecture()
        mode = self._combo_data(self.starting_weights_combo)
        if self._combo_data(self.spatial_dims_combo) != "2d" or self._combo_data(self.preset_combo) != "standard_unet":
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
        self.detected_labels_label.setText(f"Detected labels: {self._labels_summary(labels, max_items=18)}")

    def _refresh_sam3_status(self) -> None:
        if self.project is None:
            self.sam3_status_label.setText("SAM3 model not configured.")
            return
        ok, message = self._sam3_model_status(self._current_sam3_config())
        prefix = "Ready" if ok else "Needs setup"
        self.sam3_status_label.setText(f"{prefix}: {message}")
        self.sam3_preview_layer_label.setText(
            f"Preview labels: {self._current_sam3_config()['preview_labels_layer_name']}"
        )

    def _sam3_model_status(self, config: dict[str, Any]) -> tuple[bool, str]:
        mode = config.get("default_mode", "2d_box")
        if mode == "3d_multiplex":
            model_dir = Path(config.get("sam3_3d_model_dir", "")).expanduser()
            expected = ("sam3.1_multiplex.pt",)
            label = "SAM3.1 multiplex"
        else:
            model_dir = Path(config.get("sam3_2d_model_dir", "")).expanduser()
            expected = ("sam3.pt", "model.safetensors")
            label = "SAM3.0 image"
        if not str(model_dir).strip() or str(model_dir) == ".":
            return False, f"Select a {label} model folder."
        if not model_dir.exists() or not model_dir.is_dir():
            return False, f"{label} folder does not exist: {model_dir}"
        found = [name for name in expected if (model_dir / name).exists()]
        if not found:
            return False, f"{label} folder must contain one of: {', '.join(expected)}"
        if mode == "3d_multiplex" and config.get("device") == "cpu":
            return False, "SAM3.1 multiplex requires CUDA; CPU is only for 2D."
        return True, f"{label} model folder found: {', '.join(found)}"

    def _imported_model_by_id(self, model_id: str) -> dict[str, Any] | None:
        if not model_id or self.project is None:
            return None
        for entry in self.project.load_model_registry().get("models", []):
            if entry.get("model_id") == model_id:
                return entry
        return None

    def _selected_pair_ids(self) -> list[str]:
        pair_ids = []
        for index in self.dataset_table.selectionModel().selectedRows():
            item = self.dataset_table.item(index.row(), 0)
            if item is not None:
                pair_ids.append(item.data(Qt.UserRole))
        return pair_ids

    def _selected_checkpoint(self) -> dict[str, Any] | None:
        selected = self.checkpoint_table.selectionModel().selectedRows()
        if not selected:
            return None
        item = self.checkpoint_table.item(selected[0].row(), 0)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _layer_by_name(self, name: str):
        if self.viewer is None or not name:
            return None
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _ensure_points_layer(self, name: str, *, ndim: int):
        layer = self._layer_by_name(name)
        if layer is not None:
            return layer
        data = np.empty((0, ndim), dtype=float)
        return self.viewer.add_points(
            data,
            name=name,
            size=8,
            face_color="lime",
            edge_color="black",
            properties={"polarity": np.array([], dtype=object)},
        )

    def _ensure_shapes_layer(self, name: str, *, ndim: int):
        layer = self._layer_by_name(name)
        if layer is not None:
            return layer
        return self.viewer.add_shapes(
            data=[],
            name=name,
            ndim=ndim,
            edge_color="yellow",
            face_color="transparent",
            opacity=0.7,
        )

    def _ensure_preview_labels_layer(self, image_layer, name: str):
        layer = self._layer_by_name(name)
        image = np.asarray(image_layer.data)
        shape = image.shape[-3:] if image.ndim > 3 else image.shape
        if image.ndim >= 2:
            shape = image.shape[-2:] if self._combo_data(self.sam3_mode_combo) != "3d_multiplex" else image.shape
        if layer is not None:
            return layer
        labels = np.zeros(shape, dtype=np.uint16)
        return self.viewer.add_labels(labels, name=name)

    def _activate_layer(self, layer, mode: str) -> None:
        if self.viewer is None or layer is None:
            return
        self.viewer.layers.selection.active = layer
        if hasattr(layer, "mode"):
            try:
                layer.mode = mode
            except Exception:
                pass

    @staticmethod
    def _make_table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        for column in range(len(headers) - 1):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        return table

    @staticmethod
    def _add_options(combo: QComboBox, options: tuple[tuple[str, str], ...]) -> None:
        for label, value in options:
            combo.addItem(label, value)

    @staticmethod
    def _combo_data(combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data) if data is not None else combo.currentText()

    @staticmethod
    def _find_combo_data(combo: QComboBox, value: str) -> int:
        for index in range(combo.count()):
            if combo.itemData(index) == value or combo.itemText(index) == value:
                return index
        return 0

    def _restore_combo_data(self, combo: QComboBox, value: str) -> None:
        combo.setCurrentIndex(self._find_combo_data(combo, value))

    @staticmethod
    def _restore_combo_text(combo: QComboBox, text: str) -> None:
        if not text:
            return
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

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
    def _labels_summary(labels: list[int], max_items: int = 10) -> str:
        if not labels:
            return "none"
        preview = ", ".join(str(value) for value in labels[:max_items])
        if len(labels) > max_items:
            preview += ", ..."
        return preview

    @staticmethod
    def _short_id(value: str) -> str:
        parts = value.split("_")
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}"
        return value

    @staticmethod
    def _short_path(path: Path) -> str:
        text = str(path)
        if len(text) <= 72:
            return text
        return f"...{text[-69:]}"

    def _show_dataset_source_warning(self, text: str) -> None:
        self.new_masks_warning.setVisible(
            self._combo_data(self.dataset_source_combo) == "New masks since last checkpoint"
        )
