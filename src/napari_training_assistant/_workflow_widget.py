"""Tabbed workflow UI for persistent SAM3-to-U-Net training projects."""

from __future__ import annotations

from collections import deque
import time
import threading
from os import path
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QSettings, QThread, QTimer, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_training_assistant.sam3_backend import (
    SAM3BackendError,
    SAM3PreviewEngine,
)
from napari_training_assistant.sam3_backend.sam31_persistent_worker import (
    Sam31Job,
    Sam31Worker,
)
from napari_training_assistant.sam3_backend.device import resolve_device
from napari_training_assistant.io.loaders import load_image_any

SETTINGS_ORG = "napari"
SETTINGS_APP = "napari-training-assistant"
LAST_PROJECT_PATH_KEY = "last_training_project_path"


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
    ("Merge SAM instances into target class", "merge_nonzero_to_target_class"),
    ("Binary foreground/background", "merge_nonzero_to_foreground"),
    ("Keep semantic class labels", "keep_labels_as_multiclass"),
)
SAM3_MODES = (
    ("2D box", "2d_box"),
    ("2D points", "2d_points"),
    ("Live points", "live_points"),
    ("2D exemplar", "2d_exemplar"),
    ("3D / multiplex", "3d_multiplex"),
)
SAM3_DEVICES = (
    ("CUDA", "cuda"),
)
SAM3_PROPAGATION_DIRECTIONS = (
    ("both", "both"),
    ("forward", "forward"),
    ("backward", "backward"),
)


class TrainingAssistantWidget(QWidget):


    """Tabbed workflow dock widget."""

    sam3_activity_message = Signal(str)
    sam31_job_requested = Signal(object)

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self.project: TrainingProject | None = None
        self.sam3_engine = SAM3PreviewEngine()
        self._sam3_worker: Any | None = None
        self._sam31_thread: QThread | None = None
        self._sam31_worker: Sam31Worker | None = None
        self._sam3_worker_failed = False
        self._sam3_worker_activity = "preview"
        self._sam3_point_polarity = "positive"
        self._sam3_live_points_layer = None
        self._sam3_live_events_suspended = 0
        self._sam3_layer_writer: Any | None = None
        self._sam3_frame_queue = deque()
        self._sam3_frame_write_timer = QTimer(self)
        self._sam3_frame_write_timer.setInterval(50)
        self._sam3_frame_write_timer.timeout.connect(self._drain_sam3_frame_queue)
        self._loading_project_settings = False
        self._build_ui()
        self.sam3_activity_message.connect(self._log_sam3_activity)
        self.sam3_engine.reference_backend.logger = self.sam3_activity_message.emit
        self._build_sam3_shortcuts()
        self._set_project_actions_enabled(False)
        self._try_auto_open_last_project()
        

    def _settings(self) -> QSettings:
        return QSettings(SETTINGS_ORG, SETTINGS_APP)


    def _last_project_path(self) -> str:
        path = self._settings().value(LAST_PROJECT_PATH_KEY, "", type=str)
        return path or str(Path.home())


    def _remember_project_path(self, path: str | Path) -> None:
        self._settings().setValue(
            LAST_PROJECT_PATH_KEY,
            str(Path(path).expanduser().resolve()),
        )

    

    def _try_auto_open_last_project(self) -> None:
        path = self._last_project_path()
        if not path:
            return

        project_path = Path(path).expanduser()
        if not project_path.exists():
            return

        try:
            self.open_project(project_path)
        except Exception:
            # Do not crash napari if the remembered project is invalid.
            self.project = None
            self._set_project_actions_enabled(False)


    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._build_project_bar())
        layout.addWidget(self._build_model_task_bar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_sam3_tab(), "SAM3")
        self.tabs.addTab(self._build_dataset_tab(), "Dataset")
        self.tabs.addTab(self._build_train_tab(), "Train")
        self.tabs.addTab(self._build_checkpoints_tab(), "Checkpoints")
        self.tabs.addTab(self._build_predict_tab(), "Predict")
        self.tabs.addTab(self._build_advanced_tab(), "Advanced")
        layout.addWidget(self.tabs)

    def _build_sam3_shortcuts(self) -> None:
        self.sam3_live_debounce_timer = QTimer(self)
        self.sam3_live_debounce_timer.setSingleShot(True)
        self.sam3_live_debounce_timer.setInterval(250)
        self.sam3_live_debounce_timer.timeout.connect(self._run_live_sam3_preview_if_enabled)

        self.sam3_toggle_point_shortcut = QShortcut(QKeySequence("T"), self)
        self.sam3_toggle_point_shortcut.activated.connect(self.toggle_sam3_next_point_mode)
        self.sam3_flip_point_shortcut = QShortcut(QKeySequence("Shift+T"), self)
        self.sam3_flip_point_shortcut.activated.connect(self.flip_sam3_existing_point_polarity)

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
        self.sam3_direction_combo = QComboBox()
        self._add_options(self.sam3_direction_combo, SAM3_PROPAGATION_DIRECTIONS)
        self.sam3_direction_combo.currentTextChanged.connect(self.persist_sam3_settings)
        self.sam3_windows_compat_check = QCheckBox("Windows compatibility mode")
        self.sam3_windows_compat_check.setToolTip(
            "Optional slower SAM3.1 fallback mode for Windows troubleshooting only."
        )
        self.sam3_windows_compat_check.stateChanged.connect(self.persist_sam3_settings)
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



        model_layout.addWidget(QLabel("Mode"), 0, 0)
        model_layout.addWidget(self.sam3_mode_combo, 0, 1)
        model_layout.addWidget(QLabel("Device"), 0, 2)
        model_layout.addWidget(self.sam3_device_combo, 0, 3)
        model_layout.addWidget(QLabel("3D direction"), 1, 0)
        model_layout.addWidget(self.sam3_direction_combo, 1, 1, 1, 3)
        model_layout.addWidget(self.sam3_windows_compat_check, 2, 0, 1, 4)
        model_layout.addWidget(QLabel("2D model"), 3, 0)
        model_layout.addWidget(self.sam3_2d_model_edit, 3, 1, 1, 2)
        model_layout.addWidget(self.sam3_2d_model_button, 3, 3)
        model_layout.addWidget(QLabel("3D model"), 4, 0)
        model_layout.addWidget(self.sam3_3d_model_edit, 4, 1, 1, 2)
        model_layout.addWidget(self.sam3_3d_model_button, 4, 3)
        self.sam3_status_label = QLabel("SAM3 model not configured.")
        self.sam3_status_label.setWordWrap(True)
        model_layout.addWidget(self.sam3_status_label, 5, 0, 1, 4) 

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
        self.sam3_positive_point_button = QPushButton("Point +")
        self.sam3_positive_point_button.clicked.connect(lambda: self.set_sam3_point_polarity("positive"))
        self.sam3_negative_point_button = QPushButton("Point -")
        self.sam3_negative_point_button.clicked.connect(lambda: self.set_sam3_point_polarity("negative"))
        self.sam3_prompt_status_label = QLabel("Choose a mode to prepare prompt layers.")
        self.sam3_prompt_status_label.setWordWrap(True)
        prompt_layout.addWidget(QLabel("Image layer"), 0, 0)
        prompt_layout.addWidget(self.sam3_image_layer_combo, 0, 1)
        prompt_layout.addWidget(self.sam3_refresh_layers_button, 0, 2)
        prompt_layout.addWidget(self.sam3_prepare_prompt_button, 1, 0, 1, 3)
        prompt_layout.addWidget(self.sam3_positive_point_button, 2, 0)
        prompt_layout.addWidget(self.sam3_negative_point_button, 2, 1)
        prompt_layout.addWidget(self.sam3_prompt_status_label, 3, 0, 1, 3)
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
        self.sam3_progress_bar = QProgressBar()
        self.sam3_progress_bar.setRange(0, 100)
        self.sam3_progress_bar.setValue(0)
        self.sam3_progress_bar.setFormat("Idle")
        self.sam3_activity_log = QTextEdit()
        self.sam3_activity_log.setReadOnly(True)
        self.sam3_activity_log.setFixedHeight(90)
        self.sam3_activity_log.setPlaceholderText("SAM3 activity log")
        self.sam3_no_write_benchmark_check = QCheckBox("SAM3.1 no-write benchmark")
        self.sam3_no_write_benchmark_check.setToolTip(
            "Diagnostic mode: run SAM3.1 propagation without writing frames to napari."
        )
        self.sam3_no_write_benchmark_check.stateChanged.connect(self.persist_sam3_settings)
        self.sam3_debug_diagnostics_check = QCheckBox("SAM3.1 debug diagnostics")
        self.sam3_debug_diagnostics_check.setToolTip(
            "Log SAM3.1 predictor/session internals and per-frame propagation timing."
        )
        self.sam3_debug_diagnostics_check.stateChanged.connect(self.persist_sam3_settings)
        preview_layout.addWidget(self.sam3_run_preview_button, 0, 0)
        preview_layout.addWidget(self.sam3_clear_preview_button, 0, 1)
        preview_layout.addWidget(self.sam3_accept_preview_button, 1, 0, 1, 2)
        preview_layout.addWidget(self.sam3_preview_layer_label, 2, 0, 1, 2)
        preview_layout.addWidget(self.sam3_no_write_benchmark_check, 3, 0, 1, 2)
        preview_layout.addWidget(self.sam3_debug_diagnostics_check, 4, 0, 1, 2)
        preview_layout.addWidget(self.sam3_progress_bar, 5, 0, 1, 2)
        preview_layout.addWidget(self.sam3_activity_log, 6, 0, 1, 2)
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

    def _build_model_task_bar(self) -> QWidget:
        box = QGroupBox("Model Task")
        layout = QGridLayout(box)
        layout.setColumnStretch(1, 1)
        self.model_task_title_label = QLabel("Model Task")
        self.model_task_combo = QComboBox()
        self.model_task_combo.currentIndexChanged.connect(self.on_model_task_selected)
        self.new_task_button = QPushButton("New Task")
        self.new_task_button.clicked.connect(self.new_model_task)
        self.duplicate_task_button = QPushButton("Duplicate")
        self.duplicate_task_button.clicked.connect(self.duplicate_model_task)
        self.rename_task_button = QPushButton("Rename")
        self.rename_task_button.clicked.connect(self.rename_model_task)
        self.import_paired_dataset_button = QPushButton("Import paired dataset")
        self.import_paired_dataset_button.clicked.connect(self.import_paired_dataset)
        self.model_task_summary_label = QLabel(
            "Active: none | binary | classes: none | pairs: 0 | latest: none | start: fresh"
        )
        self.model_task_summary_label.setWordWrap(True)
        layout.addWidget(self.model_task_title_label, 0, 0)
        layout.addWidget(self.model_task_combo, 0, 1)
        layout.addWidget(self.new_task_button, 0, 2)
        layout.addWidget(self.duplicate_task_button, 0, 3)
        layout.addWidget(self.rename_task_button, 0, 4)
        layout.addWidget(self.import_paired_dataset_button, 0, 5)
        layout.addWidget(self.model_task_summary_label, 1, 0, 1, 6)
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
        self.open_pair_button = QPushButton("Open selected pair")
        self.open_pair_button.clicked.connect(self.open_selected_dataset_pair)
        controls_layout.addWidget(QLabel("Image layer"), 0, 0)
        controls_layout.addWidget(self.image_layer_combo, 0, 1)
        controls_layout.addWidget(QLabel("Mask layer"), 1, 0)
        controls_layout.addWidget(self.mask_layer_combo, 1, 1)
        controls_layout.addWidget(self.refresh_layers_button, 0, 2)
        controls_layout.addWidget(self.add_mask_button, 1, 2)
        controls_layout.addWidget(self.open_pair_button, 2, 2)
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
            "SAM masks often store object instances, not semantic classes. The recommended mode merges all nonzero instance IDs into the selected target class."
        )
        self.mask_cleanup_note.setWordWrap(True)
        preparation_layout.addRow("Mode", self.mask_preparation_combo)
        preparation_layout.addRow("Target class", self.target_class_name_edit)
        preparation_layout.addRow("", self.detected_labels_label)
        preparation_layout.addRow("", self.mask_cleanup_note)
        layout.addWidget(preparation)

        self.dataset_table = self._make_table(("Pair", "Source", "Image", "Mask", "Shape", "Labels", "Used"))
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

        activity_box = QGroupBox("Training activity")
        activity_layout = QVBoxLayout(activity_box)
        self.unet_progress_bar = QProgressBar()
        self.unet_progress_bar.setRange(0, 100)
        self.unet_progress_bar.setValue(0)
        self.unet_progress_bar.setFormat("Idle")
        self.unet_activity_log = QTextEdit()
        self.unet_activity_log.setReadOnly(True)
        self.unet_activity_log.setFixedHeight(160)
        self.unet_activity_log.setPlaceholderText("U-Net training activity log")
        activity_layout.addWidget(self.unet_progress_bar)
        activity_layout.addWidget(self.unet_activity_log)
        layout.addWidget(activity_box)

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

        run_box = QGroupBox("Run U-Net prediction")
        run_layout = QFormLayout(run_box)
        self.predict_image_layer_combo = QComboBox()
        self.predict_strategy_combo = QComboBox()
        self._add_options(
            self.predict_strategy_combo,
            (("Auto", "auto"), ("Full image", "full"), ("Tiled", "tiled")),
        )
        self.run_layer_prediction_button = QPushButton("Run prediction on selected layer")
        self.run_layer_prediction_button.clicked.connect(self.run_prediction_on_selected_layer)
        self.run_folder_prediction_button = QPushButton("Run prediction on input folder")
        self.run_folder_prediction_button.clicked.connect(self.run_prediction_on_input_folder)
        run_layout.addRow("Image layer", self.predict_image_layer_combo)
        run_layout.addRow("Strategy", self.predict_strategy_combo)
        run_layout.addRow("", self.run_layer_prediction_button)
        run_layout.addRow("", self.run_folder_prediction_button)
        layout.addWidget(run_box)

        save_box = QGroupBox("Save prediction layers")
        save_layout = QFormLayout(save_box)
        self.save_predictions_check = QCheckBox("Save predictions to project")
        self.save_predictions_check.setChecked(True)
        self.prediction_layer_combo = QComboBox()
        self.save_prediction_button = QPushButton("Save selected prediction layer")
        self.save_prediction_button.clicked.connect(self.save_selected_prediction)
        self.save_all_prediction_layers_button = QPushButton("Save all prediction-like layers")
        self.save_all_prediction_layers_button.clicked.connect(self.save_all_prediction_layers)
        save_layout.addRow("", self.save_predictions_check)
        save_layout.addRow("Prediction layer", self.prediction_layer_combo)
        save_layout.addRow("", self.save_prediction_button)
        save_layout.addRow("", self.save_all_prediction_layers_button)
        layout.addWidget(save_box)

        activity_box = QGroupBox("Prediction activity")
        activity_layout = QVBoxLayout(activity_box)
        self.prediction_progress_bar = QProgressBar()
        self.prediction_progress_bar.setRange(0, 100)
        self.prediction_progress_bar.setValue(0)
        self.prediction_progress_bar.setFormat("Idle")
        self.prediction_activity_log = QTextEdit()
        self.prediction_activity_log.setReadOnly(True)
        self.prediction_activity_log.setFixedHeight(130)
        self.prediction_activity_log.setPlaceholderText("Prediction activity log")
        activity_layout.addWidget(self.prediction_progress_bar)
        activity_layout.addWidget(self.prediction_activity_log)
        layout.addWidget(activity_box)

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
        path = QFileDialog.getExistingDirectory(
            self,
            "Select / Create Training Project Folder",
            self._last_project_path(),
        )

        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        self.project = TrainingProject.create_or_open(path)
        self._remember_project_path(path)
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

    def on_model_task_selected(self) -> None:
        if self.project is None or self._loading_project_settings:
            return
        task_id = self._combo_data(self.model_task_combo)
        if not task_id:
            return
        try:
            self.project.set_active_task(task_id)
        except ValueError as exc:
            QMessageBox.warning(self, "Model task not found", str(exc))
            return
        self._apply_active_task_to_controls()
        self._refresh_after_task_change()

    def new_model_task(self) -> None:
        project = self.require_project()
        if project is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("New Model Task")
        layout = QFormLayout(dialog)
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("myelin_binary")
        output_combo = QComboBox()
        output_combo.addItems(["binary", "multiclass"])
        labels_edit = QTextEdit()
        labels_edit.setFixedHeight(90)
        labels_edit.setPlainText("0: background\n1: foreground")
        starting_combo = QComboBox()
        starting_combo.addItems([
            "fresh_empty_model",
            "continue_latest_checkpoint",
            "copy_from_current_task_config",
        ])

        def on_mode_changed(text: str) -> None:
            if text == "binary":
                labels_edit.setPlainText("0: background\n1: foreground")
            elif labels_edit.toPlainText().strip() == "0: background\n1: foreground":
                labels_edit.setPlainText("0: background\n1: foreground\n2: class_2")

        output_combo.currentTextChanged.connect(on_mode_changed)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow("Task name", name_edit)
        layout.addRow("Output mode", output_combo)
        layout.addRow("Class labels", labels_edit)
        layout.addRow("Starting mode", starting_combo)
        layout.addRow("", buttons)
        result = dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()
        if result != QDialog.Accepted:
            return
        name = name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Task name required", "Enter a model task name.")
            return
        output_mode = output_combo.currentText()
        labels = self._parse_class_labels(
            labels_edit.toPlainText(),
            output_mode,
            2 if output_mode == "binary" else max(2, len(labels_edit.toPlainText().splitlines())),
        )
        copy_from = project.active_task_id() if starting_combo.currentText() == "copy_from_current_task_config" else None
        try:
            entry = project.create_task(name, output_mode, labels, copy_from_task_id=copy_from)
            config = project.active_task_config()
            config["starting_mode"] = starting_combo.currentText()
            project.save_active_task_config(config)
        except Exception as exc:
            QMessageBox.warning(self, "Could not create task", str(exc))
            return
        project.set_active_task(entry["task_id"])
        self._refresh_after_task_change()

    def duplicate_model_task(self) -> None:
        project = self.require_project()
        if project is None:
            return
        source = project.active_task_config()
        new_name, ok = QInputDialog.getText(
            self,
            "Duplicate Model Task",
            "New task name",
            text=f"{source.get('display_name', 'Task')} copy",
        )
        if not ok or not new_name.strip():
            return
        try:
            entry = project.duplicate_task(project.active_task_id(), new_name.strip())
            project.set_active_task(entry["task_id"])
        except Exception as exc:
            QMessageBox.warning(self, "Could not duplicate task", str(exc))
            return
        self._refresh_after_task_change()

    def rename_model_task(self) -> None:
        project = self.require_project()
        if project is None:
            return
        config = project.active_task_config()
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Model Task",
            "Task name",
            text=config.get("display_name", ""),
        )
        if not ok or not new_name.strip():
            return
        try:
            project.rename_task(project.active_task_id(), new_name.strip())
        except Exception as exc:
            QMessageBox.warning(self, "Could not rename task", str(exc))
            return
        self._refresh_after_task_change()

    def import_paired_dataset(self) -> None:
        project = self.require_project()
        if project is None:
            return
        image_dir = QFileDialog.getExistingDirectory(self, "Select image folder", str(project.root))
        if not image_dir:
            return
        mask_dir = QFileDialog.getExistingDirectory(self, "Select mask folder", str(project.root))
        if not mask_dir:
            return
        image_paths = {path.stem: path for path in sorted(Path(image_dir).iterdir()) if path.is_file()}
        mask_paths = {path.stem: path for path in sorted(Path(mask_dir).iterdir()) if path.is_file()}
        matches = [(stem, image_paths[stem], mask_paths[stem]) for stem in sorted(image_paths.keys() & mask_paths.keys())]
        if not matches:
            QMessageBox.warning(self, "No matched pairs", "No image/mask files had matching stems.")
            return
        unmatched_images = len(set(image_paths) - set(mask_paths))
        unmatched_masks = len(set(mask_paths) - set(image_paths))
        message = (
            f"Import {len(matches)} matched image/mask pairs into the active Model Task?"
            f"\nUnmatched images: {unmatched_images}. Unmatched masks: {unmatched_masks}."
        )
        if QMessageBox.question(self, "Import paired dataset", message) != QMessageBox.Yes:
            return
        imported = 0
        for stem, image_path, mask_path in matches:
            project.import_existing_pair(
                image_path,
                mask_path,
                metadata={"matched_stem": stem},
            )
            imported += 1
        self._refresh_after_task_change()
        QMessageBox.information(self, "Import complete", f"Imported {imported} pairs into the active Model Task.")

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
        self.dataset_count_label.setText(str(project.active_task_dataset_count()))
        self.latest_checkpoint_label.setText(
            latest["checkpoint_id"] if latest else config.get("latest_checkpoint_path") or "None"
        )
        self.latest_benchmark_label.setText(project.latest_benchmark_summary() or "None")
        self.project_state_label.setText(state.label)
        self._refresh_model_task_combo()
        self._refresh_model_task_summary()
        self._refresh_dataset_table()
        self._refresh_checkpoint_table()
        self._refresh_prediction_table()
        self._refresh_train_summary()
        self._refresh_compatibility_status()

    def _refresh_after_task_change(self) -> None:
        self._apply_active_task_to_controls()
        self.refresh_project_summary()

    def _refresh_model_task_combo(self) -> None:
        if self.project is None:
            return
        active = self.project.active_task_id()
        self.model_task_combo.blockSignals(True)
        self.model_task_combo.clear()
        for entry in self.project.task_entries():
            self.model_task_combo.addItem(entry.get("display_name", entry["task_id"]), entry["task_id"])
        index = self._find_combo_data(self.model_task_combo, active)
        self.model_task_combo.setCurrentIndex(index)
        self.model_task_combo.blockSignals(False)

    def _refresh_model_task_summary(self) -> None:
        if self.project is None:
            self.model_task_summary_label.setText(
                "Active: none | binary | classes: none | pairs: 0 | latest: none | start: fresh"
            )
            return
        config = self.project.active_task_config()
        labels = self._class_summary(config.get("class_labels", {}))
        latest = self.project.latest_checkpoint()
        latest_text = latest.get("checkpoint_id", "none") if latest else "none"
        start_mode = config.get("starting_mode", "fresh_empty_model")
        start_text = "continue" if start_mode in {"continue_latest_checkpoint", "continue_latest"} else "fresh"
        task_root = self.project.active_task_root()
        checkpoints = task_root / "checkpoints"
        predictions = task_root / "predictions"
        self.model_task_summary_label.setText(
            f"Active: {config.get('display_name', config.get('task_id', 'task'))} | "
            f"{config.get('output_mode', 'binary')} | classes: {labels or 'none'} | "
            f"pairs: {self.project.active_task_dataset_count()} | latest: {latest_text} | "
            f"start: {start_text} | checkpoints: {self._short_path(checkpoints)} | "
            f"predictions: {self._short_path(predictions)}"
        )

    def _apply_active_task_to_controls(self) -> None:
        if self.project is None or not hasattr(self, "output_mode_combo"):
            return
        task_config = self.project.active_task_config()
        architecture = {**self.project.load_architecture_config(), **task_config.get("architecture", {})}
        output_mode = task_config.get("output_mode", architecture.get("output_mode", "binary"))
        class_labels = task_config.get("class_labels", architecture.get("class_labels", {}))
        architecture["output_mode"] = output_mode
        architecture["class_labels"] = class_labels
        architecture["num_classes"] = max([int(key) for key in class_labels] or [1]) + 1
        architecture["output_channels"] = 1 if output_mode == "binary" else architecture["num_classes"]
        self._loading_project_settings = True
        try:
            self._restore_combo_data(self.output_mode_combo, output_mode)
            self.num_classes_spin.setValue(int(architecture.get("num_classes", 2)))
            self.class_labels_edit.setPlainText(self._class_labels_to_text(class_labels))
            starting_mode = task_config.get("starting_mode", "fresh_empty_model")
            self._restore_combo_data(
                self.starting_weights_combo,
                "latest_project_checkpoint" if starting_mode == "continue_latest_checkpoint" else "scratch",
            )
        finally:
            self._loading_project_settings = False
        self._sync_architecture_controls()

    def refresh_layer_choices(self) -> None:
        layer_names = []
        if self.viewer is not None:
            layer_names = [layer.name for layer in self.viewer.layers]
        current_sam3_image = self.sam3_image_layer_combo.currentText()
        current_image = self.image_layer_combo.currentText()
        current_mask = self.mask_layer_combo.currentText()
        current_prediction = self.prediction_layer_combo.currentText()
        current_predict_image = self.predict_image_layer_combo.currentText()
        for combo in (
            self.sam3_image_layer_combo,
            self.image_layer_combo,
            self.mask_layer_combo,
            self.prediction_layer_combo,
            self.predict_image_layer_combo,
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(layer_names)
            combo.blockSignals(False)
        self._restore_combo_text(self.sam3_image_layer_combo, current_sam3_image)
        self._restore_combo_text(self.image_layer_combo, current_image)
        self._restore_combo_text(self.mask_layer_combo, current_mask)
        self._restore_combo_text(self.prediction_layer_combo, current_prediction)
        self._restore_combo_text(self.predict_image_layer_combo, current_predict_image)
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

        if self._combo_data(self.sam3_mode_combo) == "3d_multiplex":
            self.sam3_run_preview_button.setText("Run 3D propagation")
        else:
            self.sam3_run_preview_button.setText("Run preview")

        if self._loading_project_settings:
            return
        if self._layer_by_name(self.sam3_image_layer_combo.currentText()) is None:
            self._sync_live_sam3_points_layer(None)
            self.sam3_prompt_status_label.setText("Select an image layer, then prepare a SAM3 prompt layer.")
            return
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
        if image_layer is None:
            self._sync_live_sam3_points_layer(None)
            self.sam3_prompt_status_label.setText("Select an image layer before preparing SAM3 prompts.")
            return
        ndim = int(getattr(image_layer, "ndim", 2) or 2)
        ndim = max(2, ndim)
        if mode == "2d_points":
            layer = self._ensure_points_layer(config["points_layer_name"], ndim=ndim)
            self._set_current_sam3_point_polarity(layer)
            self._sync_live_sam3_points_layer(None)
            self._activate_layer(layer, "add")
            self.sam3_prompt_status_label.setText("Ready: add positive/negative points, then run preview.")
        elif mode == "live_points":
            layer = self._ensure_points_layer(config["live_points_layer_name"], ndim=ndim)
            self._set_current_sam3_point_polarity(layer)
            self._sync_live_sam3_points_layer(layer)
            self._activate_layer(layer, "add")
            self.sam3_prompt_status_label.setText("Ready: Live Points armed. T toggles next point; Shift+T flips selected/latest point.")
        elif mode == "2d_box":
            self._sync_live_sam3_points_layer(None)
            layer = self._ensure_shapes_layer(config["boxes_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add_rectangle")
            self.sam3_prompt_status_label.setText("Ready: draw one or more boxes, then run preview.")
        elif mode == "2d_exemplar":
            self._sync_live_sam3_points_layer(None)
            layer = self._ensure_shapes_layer(config["exemplar_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add_rectangle")
            self.sam3_prompt_status_label.setText("Ready: draw one or more exemplar boxes, then run preview.")
        else:
            self._sync_live_sam3_points_layer(None)
            layer = self._ensure_shapes_layer(config["multiplex_prompt_layer_name"], ndim=ndim)
            self._activate_layer(layer, "add_rectangle")
            self.sam3_prompt_status_label.setText("Ready: draw SAM3.1 multiplex box prompts on the source frame.")
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

        mode = config["default_mode"]

        if mode == "3d_multiplex":
            model_dir = config["sam3_3d_model_dir"]
            prompt_layer = self._layer_by_name(config["multiplex_prompt_layer_name"])
        elif mode == "2d_box":
            model_dir = config["sam3_2d_model_dir"]
            prompt_layer = self._layer_by_name(config["boxes_layer_name"])
        elif mode == "2d_exemplar":
            model_dir = config["sam3_2d_model_dir"]
            prompt_layer = self._layer_by_name(config["exemplar_layer_name"])
        elif mode == "live_points":
            model_dir = config["sam3_2d_model_dir"]
            prompt_layer = self._layer_by_name(config["live_points_layer_name"])
        else:
            model_dir = config["sam3_2d_model_dir"]
            prompt_layer = self._layer_by_name(config["points_layer_name"])

        if prompt_layer is None:
            QMessageBox.warning(
                self,
                "Prompt layer required",
                "Click Prepare prompt layer and add a box or point before running SAM3 preview.",
            )
            return

        if self._sam3_worker is not None:
            self._log_sam3_activity("SAM3 is already running. Wait for the current run to finish before starting another.")
            QMessageBox.information(
                self,
                "SAM3 already running",
                "A SAM3 job is already running. Wait for it to finish before starting another run.",
            )
            return

        from napari.qt.threading import thread_worker

        image_data = image_layer.data if mode == "3d_multiplex" else np.asarray(image_layer.data)
        if mode in {"2d_box", "2d_exemplar", "3d_multiplex"}:
            prompt_data = list(prompt_layer.data)
        elif mode in {"2d_points", "live_points"}:
            prompt_data = {
                "data": np.asarray(prompt_layer.data),
                "properties": dict(getattr(prompt_layer, "properties", {}) or {}),
            }
        else:
            prompt_data = np.asarray(prompt_layer.data)
        image_layer_name = image_layer.name
        preview_layer_name = (
            config["propagated_labels_layer_name"]
            if mode == "3d_multiplex"
            else config["preview_labels_layer_name"]
        )
        threshold = 0.35
        compile_model = bool(config.get("compile_model", False))
        device = "cuda"
        propagation_direction = str(config.get("propagation_direction", "both") or "both")
        windows_compatibility_mode = bool(config.get("sam31_windows_compatibility_mode", False))
        no_write_benchmark = bool(config.get("sam31_no_write_benchmark", False))
        debug_diagnostics = bool(config.get("sam31_debug_diagnostics", False))
        runtime_mode = str(config.get("sam31_runtime_mode", "performance") or "performance")

        if mode == "3d_multiplex":
            try:
                from napari_sam3_assistant.core.models import Sam3Task
                from napari_sam3_assistant.services.prompt_collector import PromptCollector

                bundle = PromptCollector().collect(
                    self.viewer,
                    image_layer_name=image_layer_name,
                    task=Sam3Task.SEGMENT_3D,
                    shapes_layer_name=config["multiplex_prompt_layer_name"],
                    collect_exemplar_rois=False,
                )
                adapter = self.sam3_engine._ensure_multiplex_adapter(
                    model_dir=model_dir,
                    device="cuda",
                    threshold=threshold,
                    compile_model=compile_model,
                    windows_compatibility_mode=windows_compatibility_mode,
                )
                self._log_sam3_activity(
                    "SAM3.1 image source: "
                    f"{self.sam3_engine.reference_backend.describe_image_source(image_data)}"
                )
                if debug_diagnostics:
                    self.sam3_engine.reference_backend.log_prompt_diagnostics(bundle)
            except Exception as exc:
                QMessageBox.warning(self, "SAM3.1 multiplex setup failed", str(exc))
                return


            self._ensure_sam31_persistent_worker()
            job = Sam31Job(
                adapter=adapter,
                backend=self.sam3_engine.reference_backend,
                image_data=image_data,
                bundle=bundle,
                propagation_direction=propagation_direction,
                windows_compatibility_mode=windows_compatibility_mode,
                debug_diagnostics=debug_diagnostics,
                no_write_benchmark=no_write_benchmark,
            )
            self._sam3_frame_queue.clear()
            self._sam3_worker = self._sam31_worker
            self._sam3_worker_activity = "3D propagation"
            self._sam3_worker_failed = False
            self._set_sam3_running(True)
            self._log_sam3_activity(
                "Submitted SAM3.1 multiplex propagation to persistent worker "
                f"from frame {bundle.image.frame_index or 0} ({propagation_direction}, "
                f"runtime_mode={runtime_mode}, "
                f"no_write={no_write_benchmark})."
            )
            self.sam31_job_requested.emit(job)
            return

        @thread_worker
        def run_preview_worker():
            yield (5, "Starting SAM3 preview. First model load can take several minutes.")
            result = self.sam3_engine.run_preview(
                image=image_data,
                mode=mode,
                model_dir=model_dir,
                device=device,
                prompt_data=prompt_data,
                image_layer_name=image_layer_name,
                dims_current_step=tuple(self.viewer.dims.current_step),
                threshold=threshold,
                compile_model=compile_model,
                propagation_direction=propagation_direction,
                windows_compatibility_mode=windows_compatibility_mode,
            )
            yield ("result", result)

        worker = run_preview_worker()
        worker.yielded.connect(
            lambda payload: self._on_sam3_worker_yielded(
                payload,
                image_layer_name=image_layer_name,
                preview_layer_name=preview_layer_name,
            )
        )
        worker.errored.connect(self._on_sam3_worker_error)
        worker.finished.connect(self._on_sam3_worker_finished)
        self._sam3_worker = worker
        self._sam3_worker_activity = "preview"
        self._sam3_worker_failed = False
        self._set_sam3_running(True)
        self._log_sam3_activity(
            f"Running SAM3 preview: mode={mode}, device={device}, threshold={threshold:.2f}."
        )
        worker.start()

    def _on_sam3_worker_yielded(
        self,
        payload: Any,
        *,
        image_layer_name: str,
        preview_layer_name: str,
    ) -> None:
        if isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "result":
            self._write_sam3_preview_result(
                payload[1],
                image_layer_name=image_layer_name,
                preview_layer_name=preview_layer_name,
            )
            return
        if isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "frame_result":
            self._write_sam3_frame_result(
                payload[1],
                image_layer_name=image_layer_name,
                preview_layer_name=preview_layer_name,
            )
            return
        self._on_sam3_progress(payload)

    def _ensure_sam31_persistent_worker(self) -> Sam31Worker:
        """Create or return the long-lived SAM3.1 multiplex worker.

        The worker lives in one QThread so the expensive SAM3.1 video predictor
        can be reused across repeated propagation runs.
        """
        if self._sam31_worker is not None and self._sam31_thread is not None:
            return self._sam31_worker

        thread = QThread(self)
        worker = Sam31Worker()
        worker.moveToThread(thread)

        self.sam31_job_requested.connect(worker.run_job)
        worker.result_ready.connect(
            lambda result: self._queue_sam3_frame_result(
                result,
                image_layer_name=str(result.metadata.get("image_layer", "")),
                preview_layer_name=self._current_sam3_config()["propagated_labels_layer_name"],
            )
        )
        worker.session_ready.connect(self._on_sam3_video_session_returned)
        worker.log_message.connect(self._log_sam3_activity)
        worker.failed.connect(lambda message: self._on_sam3_worker_error(RuntimeError(message)))
        worker.finished.connect(self._on_sam3_worker_finished)
        thread.finished.connect(worker.deleteLater)
        thread.start()

        self._sam31_thread = thread
        self._sam31_worker = worker
        return worker

    def _shutdown_sam31_persistent_worker(self) -> None:
        """Stop the persistent SAM3.1 worker. Call only on unload/close."""
        try:
            self.sam31_job_requested.disconnect()
        except Exception:
            pass

        if self._sam31_thread is not None:
            self._sam31_thread.quit()
            self._sam31_thread.wait(3000)

        self._sam31_worker = None
        self._sam31_thread = None

    def closeEvent(self, event: Any) -> None:
        self._shutdown_sam31_persistent_worker()
        super().closeEvent(event)

    def _queue_sam3_frame_result(
        self,
        result,
        *,
        image_layer_name: str,
        preview_layer_name: str,
    ) -> None:
        """Queue one SAM3.1 video frame for controlled napari layer writing.

        This avoids making the SAM3 propagation worker directly compete with
        napari layer updates. The worker can continue producing frames while
        the Qt event loop writes them at a bounded rate.
        """
        self._sam3_frame_queue.append((result, image_layer_name, preview_layer_name))
        if not self._sam3_frame_write_timer.isActive():
            self._sam3_frame_write_timer.start()

    def _drain_sam3_frame_queue(self) -> None:
        """Write queued SAM3.1 frame results to napari in small batches."""
        max_frames_per_tick = 2
        written = 0
        while self._sam3_frame_queue and written < max_frames_per_tick:
            result, image_layer_name, preview_layer_name = self._sam3_frame_queue.popleft()
            self._write_sam3_frame_result(
                result,
                image_layer_name=image_layer_name,
                preview_layer_name=preview_layer_name,
            )
            written += 1

        if not self._sam3_frame_queue:
            self._sam3_frame_write_timer.stop()

    def _write_sam3_preview_result(
        self,
        result,
        *,
        image_layer_name: str,
        preview_layer_name: str,
    ) -> None:
        image_layer = self._layer_by_name(image_layer_name)
        if image_layer is None:
            self._log_sam3_activity(f"SAM3 preview finished, but image layer is gone: {image_layer_name}")
            return
        preview = self._ensure_preview_labels_layer(image_layer, preview_layer_name)
        preview.data = result.labels
        preview.metadata["sam3_preview"] = result.metadata

        self._activate_layer(preview, "paint")
        if self._combo_data(self.sam3_mode_combo) == "live_points":
            self._activate_live_sam3_points_layer()
        self.sam3_prompt_status_label.setText("SAM3 preview updated.")
        self._on_sam3_progress((100, "SAM3 preview updated."))
        backend = result.metadata.get("backend_status", "unknown")
        nonempty_frames = result.metadata.get("nonempty_frames")
        frame_summary = (
            f"; nonempty_frames={int(nonempty_frames)}"
            if nonempty_frames is not None
            else ""
        )
        self._log_sam3_activity(
            f"SAM3 preview updated: objects={int(np.max(result.labels)) if result.labels.size else 0}{frame_summary}; backend={backend}."
        )
        self.refresh_layer_choices()

    def _write_sam3_frame_result(
        self,
        result,
        *,
        image_layer_name: str,
        preview_layer_name: str,
    ) -> None:
        write_t0 = time.perf_counter()
        image_layer = self._layer_by_name(image_layer_name)
        if image_layer is None:
            self._log_sam3_activity(f"SAM3.1 frame arrived, but image layer is gone: {image_layer_name}")
            return

        image_shape = tuple(int(value) for value in getattr(image_layer.data, "shape"))
        output_shape = self._sam3_video_output_shape(image_layer.name, image_shape)

        try:
            writer = self._sam3_reference_layer_writer()
            writer.write_video_frame_result(
                result,
                output_shape,
                labels_name=preview_layer_name,
            )
        except Exception as exc:
            self._log_sam3_activity(f"SAM3.1 frame write failed: {exc}")
            return

        write_elapsed = time.perf_counter() - write_t0
        frame_index = result.metadata.get("frame_index", getattr(result, "frame_index", None))
        stage = result.metadata.get("stage", "")

        if frame_index is not None:
            try:
                total_frames = max(int(output_shape[0]), 1)
                progress = min(99, max(0, int((int(frame_index) + 1) / total_frames * 100)))
                self.sam3_progress_bar.setRange(0, 100)
                self.sam3_progress_bar.setValue(progress)
                self.sam3_progress_bar.setFormat(f"SAM3.1 wrote frame {frame_index}")
            except Exception:
                pass

        self._log_sam3_activity(self._sam3_result_summary(result))

        if write_elapsed > 0.2:
            self._log_sam3_activity(
                f"SAM3.1 slow frame write: frame={frame_index}, write_time={write_elapsed:.3f} sec"
            )

        if frame_index is not None and stage == "prompt":
            self._log_sam3_activity(f"SAM3.1 prompt frame {frame_index} updated.")

    @staticmethod
    def _sam3_result_summary(result) -> str:
        frame_index = getattr(result, "frame_index", None)
        session_id = getattr(result, "session_id", None)
        if getattr(result, "object_ids", None) is not None:
            count = len(np.asarray(result.object_ids).reshape(-1))
        elif getattr(result, "masks", None) is not None:
            masks = np.asarray(result.masks)
            count = int(masks.shape[0]) if masks.ndim >= 3 else int(bool(masks.any()))
        elif getattr(result, "labels", None) is not None:
            labels = np.asarray(result.labels)
            count = int(np.max(labels)) if labels.size else 0
        else:
            count = 0
        frame = f" frame={frame_index}" if frame_index is not None else ""
        session = f" session={session_id}" if session_id else ""
        return f"SAM3 result:{frame}{session} objects={count}"

    def _sam3_reference_layer_writer(self):
        if self._sam3_layer_writer is None:
            from napari_sam3_assistant.services.layer_writer import LayerWriter

            self._sam3_layer_writer = LayerWriter(self.viewer)
        return self._sam3_layer_writer

    def clear_sam3_preview(self) -> None:
        if self.viewer is None:
            return
        config = self._current_sam3_config()
        for name in (config["preview_labels_layer_name"], config["propagated_labels_layer_name"]):
            layer = self._layer_by_name(name)
            if layer is not None:
                self.viewer.layers.remove(layer)
        self.refresh_layer_choices()
        self._log_sam3_activity("Cleared SAM3 preview layers.")

    def _on_sam3_progress(self, payload: Any) -> None:
        if isinstance(payload, tuple) and len(payload) == 2:
            value, message = payload
            try:
                if self.sam3_progress_bar.maximum() != 0:
                    self.sam3_progress_bar.setValue(int(value))
            except Exception:
                pass
            self.sam3_progress_bar.setFormat(str(message))
            self.sam3_prompt_status_label.setText(str(message))
            self._log_sam3_activity(str(message))
            return
        self._log_sam3_activity(str(payload))

    def _on_sam3_worker_error(self, error: Any) -> None:
        self._sam3_worker_failed = True
        activity = self._sam3_worker_activity
        message = str(error)
        self.sam3_prompt_status_label.setText(f"SAM3 {activity} failed.")
        self._log_sam3_activity(f"SAM3 {activity} failed: {message}")
        QMessageBox.warning(self, f"SAM3 {activity} failed", message)

    def _on_sam3_model_loaded(self, message: str) -> None:
        self.sam3_prompt_status_label.setText(str(message))
        self._on_sam3_progress((100, str(message)))

    def _on_sam3_video_session_returned(self, session: Any) -> None:
        self.sam3_engine.reference_backend.video_session = session
        session_id = getattr(session, "session_id", "")
        if session_id:
            self._log_sam3_activity(f"Video session ready: {session_id}")

    def _on_sam3_worker_finished(self) -> None:
        activity = self._sam3_worker_activity
        self._sam3_worker = None

        # Make sure any queued SAM3.1 frames are written before showing Complete.
        while self._sam3_frame_queue:
            self._drain_sam3_frame_queue()

        self._set_sam3_running(False)
        if self._sam3_worker_failed:
            self.sam3_progress_bar.setFormat("Failed")
            self._log_sam3_activity(f"SAM3 {activity} stopped after failure.")
        else:
            self.sam3_progress_bar.setRange(0, 100)
            self.sam3_progress_bar.setValue(100)
            self.sam3_progress_bar.setFormat("Complete")
            self._log_sam3_activity(f"SAM3 {activity} finished.")

    def _set_sam3_running(self, running: bool) -> None:
        activity = self._sam3_worker_activity

        self.sam3_run_preview_button.setEnabled(not running)
        self.sam3_clear_preview_button.setEnabled(not running)
        self.sam3_accept_preview_button.setEnabled(not running)
        self.sam3_progress_bar.setRange(0, 0 if running else 100)
        self.sam3_progress_bar.setFormat(f"SAM3 {activity} running..." if running else "Idle")
        if not running and not self._sam3_worker_failed:
            self.sam3_progress_bar.setValue(0)
        self.setCursor(Qt.BusyCursor if running else Qt.ArrowCursor)

    def _log_sam3_activity(self, message: str) -> None:
        self.sam3_activity_log.append(message)

    def _activate_live_sam3_points_layer(self) -> None:
        layer = self._current_sam3_points_layer()
        if layer is not None:
            self._activate_layer(layer, "add")

    def set_sam3_point_polarity(self, polarity: str) -> None:
        self._sam3_point_polarity = "negative" if polarity == "negative" else "positive"
        layer = self._current_sam3_points_layer()
        if layer is not None:
            selected = sorted(getattr(layer, "selected_data", []))
            if selected:
                properties = dict(getattr(layer, "properties", {}) or {})
                values = self._sam3_point_polarity_values(layer)
                for index in selected:
                    values[index] = self._sam3_point_polarity
                properties["polarity"] = np.asarray(values, dtype=object)
                layer.properties = properties
                self._refresh_sam3_point_colors(layer)
                if self._live_sam3_points_enabled():
                    self._request_live_sam3_preview()
            self._set_current_sam3_point_polarity(layer)
        self._log_sam3_activity(f"Next SAM3 point mode: {self._sam3_point_polarity}.")

    def toggle_sam3_next_point_mode(self) -> None:
        if not self._point_shortcuts_enabled():
            return
        self.set_sam3_point_polarity(
            "negative" if self._sam3_point_polarity == "positive" else "positive"
        )

    def flip_sam3_existing_point_polarity(self) -> None:
        if not self._point_shortcuts_enabled():
            return
        layer = self._current_sam3_points_layer()
        if layer is None:
            self._log_sam3_activity("No SAM3 points layer selected.")
            return
        data = np.asarray(layer.data)
        if len(data) == 0:
            self._log_sam3_activity("No SAM3 points available to flip.")
            return
        indices = sorted(getattr(layer, "selected_data", [])) or [len(data) - 1]
        properties = dict(getattr(layer, "properties", {}) or {})
        values = self._sam3_point_polarity_values(layer)
        for index in indices:
            values[index] = "negative" if values[index] == "positive" else "positive"
        properties["polarity"] = np.asarray(values, dtype=object)
        layer.properties = properties
        self._refresh_sam3_point_colors(layer)
        self._log_sam3_activity(f"Flipped {len(indices)} SAM3 point(s).")
        if self._live_sam3_points_enabled():
            self._request_live_sam3_preview()

    def _point_shortcuts_enabled(self) -> bool:
        return self._combo_data(self.sam3_mode_combo) in {"2d_points", "live_points"}

    def _live_sam3_points_enabled(self) -> bool:
        return (
            self.viewer is not None
            and self._combo_data(self.sam3_mode_combo) == "live_points"
            and self._sam3_worker is None
        )

    def _run_live_sam3_preview_if_enabled(self) -> None:
        if self._live_sam3_points_enabled():
            self.run_sam3_preview()

    def _request_live_sam3_preview(self) -> None:
        if self._live_sam3_points_enabled():
            self.sam3_live_debounce_timer.start()

    def _sync_live_sam3_points_layer(self, layer) -> None:
        if self._sam3_live_points_layer is layer:
            return
        self._disconnect_live_sam3_points_layer()
        self._sam3_live_points_layer = layer
        if layer is None:
            return
        events = getattr(layer, "events", None)
        if events is None:
            return
        for event_name in ("data", "properties", "set_data"):
            event = getattr(events, event_name, None)
            if event is None:
                continue
            try:
                event.connect(self._on_live_sam3_points_changed)
            except Exception:
                pass

    def _disconnect_live_sam3_points_layer(self) -> None:
        layer = self._sam3_live_points_layer
        if layer is None:
            return
        events = getattr(layer, "events", None)
        if events is not None:
            for event_name in ("data", "properties", "set_data"):
                event = getattr(events, event_name, None)
                if event is None:
                    continue
                try:
                    event.disconnect(self._on_live_sam3_points_changed)
                except Exception:
                    pass
        self._sam3_live_points_layer = None

    def _on_live_sam3_points_changed(self, event: Any = None) -> None:
        if self._sam3_live_events_suspended:
            return
        self._request_live_sam3_preview()

    def _current_sam3_points_layer(self):
        if self.viewer is None:
            return None
        config = self._current_sam3_config()
        mode = self._combo_data(self.sam3_mode_combo)
        name = config["live_points_layer_name"] if mode == "live_points" else config["points_layer_name"]
        return self._layer_by_name(name)

    def _set_current_sam3_point_polarity(self, layer) -> None:
        if layer is None:
            return
        self._sam3_live_events_suspended += 1
        try:
            layer.current_properties = {
                "polarity": np.asarray([self._sam3_point_polarity], dtype=object)
            }
        finally:
            self._sam3_live_events_suspended = max(0, self._sam3_live_events_suspended - 1)

    def _sam3_point_polarity_values(self, layer) -> list[str]:
        properties = dict(getattr(layer, "properties", {}) or {})
        values = [str(value) for value in list(properties.get("polarity", []))]
        if len(values) < len(layer.data):
            values.extend(["positive"] * (len(layer.data) - len(values)))
        return [
            "negative" if str(value).strip().lower() == "negative" else "positive"
            for value in values[: len(layer.data)]
        ]

    def _refresh_sam3_point_colors(self, layer) -> None:
        try:
            layer.refresh_colors()
        except Exception:
            pass

    def accept_sam3_preview_to_dataset(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if self.viewer is None:
            QMessageBox.warning(self, "No viewer", "A napari viewer is required to accept SAM3 preview masks.")
            return
        config = self._current_sam3_config()
        image_layer = self._layer_by_name(self.sam3_image_layer_combo.currentText())
        preview_layer_name = (
            config["propagated_labels_layer_name"]
            if config.get("default_mode") == "3d_multiplex"
            else config["preview_labels_layer_name"]
        )
        preview_layer = self._layer_by_name(preview_layer_name)
        if image_layer is None or preview_layer is None:
            QMessageBox.warning(
                self,
                "Preview required",
                f"Run or create the SAM3 labels layer before accepting: {preview_layer_name}",
            )
            return
        project.add_pair(
            np.asarray(image_layer.data),
            np.asarray(preview_layer.data),
            image_layer_name=image_layer.name,
            mask_layer_name=preview_layer.name,
            source="sam3_preview",
            mask_preparation=self._current_mask_preparation(),
            metadata={"source": "sam3_preview", "sam3": config},
        )
        self.image_layer_combo.setCurrentText(image_layer.name)
        self.mask_layer_combo.setCurrentText(preview_layer.name)
        self._refresh_after_task_change()
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
                source="manual_label",
                mask_preparation=self._current_mask_preparation(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Mask labels do not match settings", str(exc))
            return
        self._refresh_after_task_change()

    def open_selected_dataset_pair(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if self.viewer is None:
            QMessageBox.warning(self, "No viewer", "A napari viewer is required to open dataset pairs.")
            return
        selected = self.dataset_table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "Dataset pair required", "Select one dataset pair to open.")
            return
        item = self.dataset_table.item(selected[0].row(), 0)
        pair_id = item.data(Qt.UserRole) if item is not None else ""
        pair = next((candidate for candidate in project.dataset_pairs() if candidate.get("pair_id") == pair_id), None)
        if pair is None:
            QMessageBox.warning(self, "Dataset pair missing", "The selected dataset pair was not found.")
            return
        image_path = self._project_path(pair.get("image_path", ""))
        mask_path = self._project_path(pair.get("mask_path", ""))
        if image_path is None or mask_path is None or not image_path.exists() or not mask_path.exists():
            QMessageBox.warning(self, "Dataset file missing", "The selected image or mask file no longer exists.")
            return
        try:
            image = load_image_any(image_path)
            mask = load_image_any(mask_path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open dataset pair", str(exc))
            return
        prefix = self._short_id(pair_id)
        image_layer = self.viewer.add_image(image, name=f"{prefix} image")
        labels_layer = self.viewer.add_labels(mask, name=f"{prefix} mask")
        self.refresh_layer_choices()
        self.image_layer_combo.setCurrentText(image_layer.name)
        self.mask_layer_combo.setCurrentText(labels_layer.name)
        self.sam3_image_layer_combo.setCurrentText(image_layer.name)
        self.tabs.setCurrentIndex(1)

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

    def _clear_unet_activity_log(self) -> None:
        self.unet_activity_log.clear()

    def _log_unet_activity(self, message: str) -> None:
        self.unet_activity_log.append(str(message))

    @staticmethod
    def _format_unet_metric(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    def _set_unet_training_running(self, running: bool) -> None:
        self.train_button.setEnabled(not running)
        self.retrain_button.setEnabled(not running)
        if running:
            self.unet_progress_bar.setRange(0, 0)
            self.unet_progress_bar.setFormat("Preparing...")
            self.setCursor(Qt.BusyCursor)
        else:
            self.unet_progress_bar.setRange(0, 100)
            self.setCursor(Qt.ArrowCursor)

    def _on_unet_training_progress(self, epoch: int, total_epochs: int, row: dict[str, Any]) -> None:
        total_epochs = max(int(total_epochs), 1)
        epoch = int(epoch)
        percent = int(round(epoch / total_epochs * 100))
        self.unet_progress_bar.setRange(0, 100)
        self.unet_progress_bar.setValue(max(0, min(100, percent)))
        self.unet_progress_bar.setFormat(f"Epoch {epoch}/{total_epochs}")
        fmt = self._format_unet_metric
        self._log_unet_activity(
            f"Epoch {epoch}/{total_epochs} | "
            f"train_loss={fmt(row.get('train_loss'))} | "
            f"val_loss={fmt(row.get('val_loss'))} | "
            f"train_dice={fmt(row.get('train_dice'))} | "
            f"val_dice={fmt(row.get('val_dice'))} | "
            f"train_iou={fmt(row.get('train_iou'))} | "
            f"val_iou={fmt(row.get('val_iou'))} | "
            f"train_f1={fmt(row.get('train_f1'))} | "
            f"val_f1={fmt(row.get('val_f1'))}"
        )

    def _on_unet_training_complete(self, summary: dict[str, Any]) -> None:
        self.refresh_project_summary()
        self.tabs.setCurrentIndex(3)
        checkpoint_id = summary.get("checkpoint_id", "checkpoint")
        patches = summary.get("number_of_patches", 0)
        self.train_summary_label.setText(
            f"Training complete: {checkpoint_id} with {patches} patches."
        )
        self.unet_progress_bar.setRange(0, 100)
        self.unet_progress_bar.setValue(100)
        self.unet_progress_bar.setFormat("Complete")
        self._log_unet_activity(f"Training complete: {checkpoint_id}")
        self._log_unet_activity(
            f"Patches: {patches} | "
            f"Best epoch: {summary.get('best_epoch', '')} | "
            f"Best val Dice: {self._format_unet_metric(summary.get('best_val_dice', ''))} | "
            f"Best val IoU: {self._format_unet_metric(summary.get('best_val_iou', ''))}"
        )
        if summary.get("run_dir"):
            self._log_unet_activity(f"Run folder: {summary.get('run_dir')}")
        QMessageBox.information(
            self,
            "U-Net training complete",
            f"Saved {checkpoint_id} and updated project history.",
        )

    def _on_unet_worker_yielded(self, payload: Any) -> None:
        if not isinstance(payload, tuple):
            self._log_unet_activity(str(payload))
            return
        if len(payload) >= 1 and payload[0] == "progress":
            _, epoch, total_epochs, row = payload
            self._on_unet_training_progress(epoch, total_epochs, row)
            return
        if len(payload) >= 1 and payload[0] == "result":
            _, summary = payload
            self._on_unet_training_complete(summary)
            return
        self._log_unet_activity(str(payload))

    def train_unet(self) -> None:
        project = self.require_project()
        if project is None:
            return
        self.persist_all_settings()
        architecture = project.load_architecture_config()
        selected_checkpoint = self._selected_checkpoint()
        training_mode_label = self._combo_data(self.training_mode_combo)
        parent_checkpoint_id = ""
        if training_mode_label == "Continue from latest checkpoint":
            latest = project.latest_checkpoint()
            parent_checkpoint_id = latest.get("checkpoint_id", "") if latest else ""
            if latest and latest.get("architecture"):
                compatible, message = project.architecture_compatibility(
                    architecture, latest.get("architecture", {})
                )
                if not compatible:
                    QMessageBox.warning(self, "Checkpoint architecture mismatch", message)
                    return
            self.starting_weights_combo.setCurrentIndex(
                self._find_combo_data(self.starting_weights_combo, "latest_project_checkpoint")
            )
        elif training_mode_label == "Continue from selected checkpoint":
            if not selected_checkpoint:
                QMessageBox.warning(self, "Checkpoint required", "Select a checkpoint to continue from.")
                self.tabs.setCurrentIndex(3)
                return
            compatible, message = project.architecture_compatibility(
                architecture, selected_checkpoint.get("architecture", {})
            )
            if not compatible:
                QMessageBox.warning(self, "Checkpoint architecture mismatch", message)
                return
            parent_checkpoint_id = selected_checkpoint.get("checkpoint_id", "")
            self.starting_weights_combo.setCurrentIndex(
                self._find_combo_data(self.starting_weights_combo, "selected_project_checkpoint")
            )
        else:
            self.starting_weights_combo.setCurrentIndex(
                self._find_combo_data(self.starting_weights_combo, "scratch")
            )
        self.persist_starting_weights_settings()
        selected_pair_ids = self._selected_pair_ids()
        dataset_source = self._combo_data(self.dataset_source_combo)
        pairs = project.selected_dataset_pairs(
            dataset_source,
            selected_pair_ids=selected_pair_ids,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        if not pairs:
            QMessageBox.warning(self, "Dataset required", "No dataset pairs match the selected dataset source.")
            self.tabs.setCurrentIndex(1)
            return
        pair_ids = [pair["pair_id"] for pair in pairs]

        if getattr(self, "_unet_worker", None) is not None:
            QMessageBox.information(self, "Training already running", "A U-Net training run is already active.")
            return

        from napari.qt.threading import thread_worker
        from napari_training_assistant.unet_backend.project_runner import (
            run_unet_training_for_project,
        )

        config = project.load_config()
        task_config = project.active_task_config()
        self._clear_unet_activity_log()
        self._set_unet_training_running(True)
        self.train_summary_label.setText(f"Training U-Net on {len(pair_ids)} dataset pairs...")
        self._log_unet_activity("Starting U-Net training.")
        self._log_unet_activity(f"Task: {task_config.get('display_name', project.active_task_id())}")
        self._log_unet_activity(f"Dataset source: {dataset_source}")
        self._log_unet_activity(f"Dataset pairs: {len(pair_ids)}")
        self._log_unet_activity(
            f"Patch size: {config.get('patch_size', 256)} | "
            f"batch size: {config.get('batch_size', 4)} | "
            f"epochs: {config.get('epochs', 10)} | "
            f"lr: {config.get('learning_rate', 0.0001)} | "
            f"validation split: {config.get('validation_split', 0.2)}"
        )

        @thread_worker
        def run_training_worker():
            queue = deque()
            result_holder: dict[str, Any] = {}
            error_holder: dict[str, Exception] = {}

            def progress_cb(epoch: int, total_epochs: int, row: dict[str, Any]) -> None:
                queue.append(("progress", epoch, total_epochs, row))

            def train_target() -> None:
                try:
                    result_holder["summary"] = run_unet_training_for_project(
                        project,
                        selected_pair_ids=pair_ids,
                        progress_cb=progress_cb,
                    )
                except Exception as exc:
                    error_holder["error"] = exc

            thread = threading.Thread(target=train_target, daemon=True)
            thread.start()

            while thread.is_alive():
                while queue:
                    yield queue.popleft()
                time.sleep(0.1)

            thread.join()

            while queue:
                yield queue.popleft()

            if "error" in error_holder:
                raise error_holder["error"]

            yield ("result", result_holder.get("summary", {}))

        def on_error(error: Any) -> None:
            self._unet_worker = None
            self.refresh_project_summary()
            message = str(error)
            self.train_summary_label.setText(f"Training failed: {message}")
            self.unet_progress_bar.setRange(0, 100)
            self.unet_progress_bar.setFormat("Failed")
            self._log_unet_activity(f"Training failed: {message}")
            QMessageBox.critical(self, "U-Net training failed", message)

        def on_finished() -> None:
            self._unet_worker = None
            self._set_unet_training_running(False)
            self.refresh_project_summary()

        worker = run_training_worker()
        worker.yielded.connect(self._on_unet_worker_yielded)
        worker.errored.connect(on_error)
        worker.finished.connect(on_finished)
        self._unet_worker = worker
        worker.start()

    def retrain_from_scratch(self) -> None:
        self.training_mode_combo.setCurrentIndex(self._find_combo_data(self.training_mode_combo, "Retrain from scratch"))
        self.starting_weights_combo.setCurrentIndex(self._find_combo_data(self.starting_weights_combo, "scratch"))
        self.train_unet()

    def _log_prediction_activity(self, message: str) -> None:
        self.prediction_activity_log.append(str(message))

    def _set_prediction_running(self, running: bool) -> None:
        for widget in (
            self.run_layer_prediction_button,
            self.run_folder_prediction_button,
            self.save_prediction_button,
            self.save_all_prediction_layers_button,
        ):
            widget.setEnabled(not running)
        if running:
            self.prediction_progress_bar.setRange(0, 0)
            self.prediction_progress_bar.setFormat("Running...")
            self.setCursor(Qt.BusyCursor)
        else:
            self.prediction_progress_bar.setRange(0, 100)
            self.setCursor(Qt.ArrowCursor)

    def _prediction_run_dir(self) -> Path | None:
        project = self.require_project()
        if project is None:
            return None
        checkpoint = self._selected_checkpoint() or project.latest_checkpoint()
        checkpoint_id = checkpoint.get("checkpoint_id", "") if checkpoint else ""
        runs_root = project.active_task_root() / "history" / "unet_runs"
        if not runs_root.exists():
            QMessageBox.warning(self, "No U-Net run", "Train U-Net before running prediction.")
            return None

        candidates = sorted(
            [path for path in runs_root.iterdir() if path.is_dir() and (path / "best_model.pt").exists()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if checkpoint_id:
            for run_dir in candidates:
                summary_path = run_dir / "summary.json"
                try:
                    summary = TrainingProject.read_json(summary_path) if summary_path.exists() else {}
                except Exception:
                    summary = {}
                if summary.get("checkpoint_id") == checkpoint_id:
                    return run_dir
        if candidates:
            return candidates[0]
        QMessageBox.warning(self, "No U-Net run", "No run folder with best_model.pt was found for the active Model Task.")
        return None

    @staticmethod
    def _safe_prediction_name(name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
        cleaned = cleaned.strip("_")
        return cleaned or "prediction"

    def _save_prediction_array(self, array: np.ndarray, *, source_name: str) -> Path | None:
        project = self.require_project()
        if project is None or not self.save_predictions_check.isChecked():
            return None
        filename = f"{self._safe_prediction_name(source_name)}_pred.tif"
        return project.save_prediction(np.asarray(array), name=filename)

    def run_prediction_on_selected_layer(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if self.viewer is None:
            QMessageBox.warning(self, "No viewer", "A napari viewer is required to run prediction on a layer.")
            return
        run_dir = self._prediction_run_dir()
        if run_dir is None:
            return
        layer = self._layer_by_name(self.predict_image_layer_combo.currentText())
        if layer is None:
            QMessageBox.warning(self, "Image layer required", "Select an image layer for prediction.")
            return
        if getattr(self, "_prediction_worker", None) is not None:
            QMessageBox.information(self, "Prediction already running", "A prediction job is already active.")
            return

        from napari.qt.threading import thread_worker

        image = np.asarray(layer.data)
        image_name = layer.name
        strategy = self._combo_data(self.predict_strategy_combo) or "auto"
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.prediction_activity_log.clear()
        self._set_prediction_running(True)
        self._log_prediction_activity(f"Running U-Net prediction on layer: {image_name}")
        self._log_prediction_activity(f"Run folder: {run_dir}")
        self._log_prediction_activity(f"Strategy: {strategy} | device: {device}")

        @thread_worker
        def prediction_worker():
            yield (5, "Loading U-Net model...")
            from napari_training_assistant.unet_backend.predictor import (
                _predict_with_model,
                load_model_from_run_folder,
            )
            model, cfg = load_model_from_run_folder(run_dir, device=device)
            yield (20, "Model loaded. Running prediction...")
            pred, used_strategy = _predict_with_model(model, cfg, image, device=device, strategy=strategy)
            yield (90, f"Prediction complete with {used_strategy} strategy.")
            yield ("result", pred, used_strategy)

        worker = prediction_worker()
        worker.yielded.connect(lambda payload: self._on_layer_prediction_yielded(payload, image_name=image_name))
        worker.errored.connect(self._on_prediction_error)
        worker.finished.connect(self._on_prediction_finished)
        self._prediction_worker = worker
        worker.start()

    def _on_layer_prediction_yielded(self, payload: Any, *, image_name: str) -> None:
        if isinstance(payload, tuple) and len(payload) == 3 and payload[0] == "result":
            _, pred, used_strategy = payload
            name = f"U-Net prediction - {image_name}"
            if self.viewer is not None:
                self.viewer.add_labels(np.asarray(pred), name=name)
            saved = self._save_prediction_array(pred, source_name=image_name)
            if saved is not None:
                self._log_prediction_activity(f"Saved prediction: {saved}")
            self.prediction_progress_bar.setRange(0, 100)
            self.prediction_progress_bar.setValue(100)
            self.prediction_progress_bar.setFormat("Complete")
            self._log_prediction_activity(f"Added prediction layer: {name} ({used_strategy})")
            self.refresh_layer_choices()
            self._refresh_after_task_change()
            return
        self._on_prediction_progress(payload)

    def run_prediction_on_input_folder(self) -> None:
        project = self.require_project()
        if project is None:
            return
        run_dir = self._prediction_run_dir()
        if run_dir is None:
            return
        input_dir = QFileDialog.getExistingDirectory(self, "Select input image folder", str(project.root))
        if not input_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select output prediction folder", str(project.active_task_predictions_dir()))
        if not output_dir:
            output_dir = str(project.active_task_predictions_dir())
        if getattr(self, "_prediction_worker", None) is not None:
            QMessageBox.information(self, "Prediction already running", "A prediction job is already active.")
            return

        from napari.qt.threading import thread_worker

        strategy = self._combo_data(self.predict_strategy_combo) or "auto"
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prediction_activity_log.clear()
        self._set_prediction_running(True)
        self._log_prediction_activity(f"Running folder prediction: {input_dir}")
        self._log_prediction_activity(f"Output folder: {output_dir}")
        self._log_prediction_activity(f"Run folder: {run_dir}")

        @thread_worker
        def folder_prediction_worker():
            from napari_training_assistant.unet_backend.predictor import predict_folder_from_run_folder

            def progress_cb(index: int, total: int, path: str, status: str) -> None:
                yield_payloads.append(("progress", index, total, path, status))

            yield_payloads: deque = deque()

            def cb(index: int, total: int, path: str, status: str) -> None:
                yield_payloads.append(("progress", index, total, path, status))

            result_holder: dict[str, Any] = {}
            error_holder: dict[str, Exception] = {}

            def predict_target() -> None:
                try:
                    report, cfg = predict_folder_from_run_folder(
                        run_dir,
                        input_dir,
                        output_dir,
                        device=device,
                        strategy=strategy,
                        overwrite=True,
                        progress_cb=cb,
                    )
                    result_holder["report"] = report
                    result_holder["cfg"] = cfg
                except Exception as exc:
                    error_holder["error"] = exc

            thread = threading.Thread(target=predict_target, daemon=True)
            thread.start()
            while thread.is_alive():
                while yield_payloads:
                    yield yield_payloads.popleft()
                time.sleep(0.1)
            thread.join()
            while yield_payloads:
                yield yield_payloads.popleft()
            if "error" in error_holder:
                raise error_holder["error"]
            yield ("folder_result", result_holder.get("report", []))

        worker = folder_prediction_worker()
        worker.yielded.connect(self._on_folder_prediction_yielded)
        worker.errored.connect(self._on_prediction_error)
        worker.finished.connect(self._on_prediction_finished)
        self._prediction_worker = worker
        worker.start()

    def _on_folder_prediction_yielded(self, payload: Any) -> None:
        if isinstance(payload, tuple) and len(payload) == 5 and payload[0] == "progress":
            _, index, total, path, status = payload
            total = max(int(total), 1)
            index = int(index)
            percent = int(round(index / total * 100))
            self.prediction_progress_bar.setRange(0, 100)
            self.prediction_progress_bar.setValue(percent)
            self.prediction_progress_bar.setFormat(f"Predicting {index}/{total}")
            self._log_prediction_activity(f"{index}/{total} | {Path(path).name} | {status}")
            return
        if isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "folder_result":
            _, report = payload
            ok = sum(1 for item in report if str(item.get("status", "")) == "ok")
            errors = sum(1 for item in report if str(item.get("status", "")).startswith("error"))
            self.prediction_progress_bar.setRange(0, 100)
            self.prediction_progress_bar.setValue(100)
            self.prediction_progress_bar.setFormat("Complete")
            self._log_prediction_activity(f"Folder prediction complete: {ok} ok, {errors} errors, {len(report)} total.")
            self._refresh_after_task_change()
            return
        self._on_prediction_progress(payload)

    def _on_prediction_progress(self, payload: Any) -> None:
        if isinstance(payload, tuple) and len(payload) == 2:
            value, message = payload
            try:
                if self.prediction_progress_bar.maximum() != 0:
                    self.prediction_progress_bar.setValue(int(value))
            except Exception:
                pass
            self.prediction_progress_bar.setFormat(str(message))
            self._log_prediction_activity(str(message))
            return
        self._log_prediction_activity(str(payload))

    def _on_prediction_error(self, error: Any) -> None:
        self._prediction_worker = None
        message = str(error)
        self.prediction_progress_bar.setRange(0, 100)
        self.prediction_progress_bar.setFormat("Failed")
        self._log_prediction_activity(f"Prediction failed: {message}")
        QMessageBox.critical(self, "U-Net prediction failed", message)

    def _on_prediction_finished(self) -> None:
        self._prediction_worker = None
        self._set_prediction_running(False)
        self.refresh_project_summary()

    def _prediction_like_layers(self) -> list[Any]:
        if self.viewer is None:
            return []
        out = []
        for layer in self.viewer.layers:
            name = str(getattr(layer, "name", "")).lower()
            if "pred" in name or "prediction" in name or "u-net" in name or "unet" in name:
                out.append(layer)
        return out

    def save_all_prediction_layers(self) -> None:
        project = self.require_project()
        if project is None:
            return
        if not self.save_predictions_check.isChecked():
            return
        layers = self._prediction_like_layers()
        if not layers:
            QMessageBox.warning(
                self,
                "No prediction layers",
                "No prediction-like layers were found. Layer names should contain pred, prediction, U-Net, or unet.",
            )
            return
        saved = 0
        for layer in layers:
            self._save_prediction_array(np.asarray(layer.data), source_name=layer.name)
            saved += 1
        self._log_prediction_activity(f"Saved {saved} prediction layer(s) to the active Model Task.")
        self._refresh_after_task_change()

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
        saved = self._save_prediction_array(np.asarray(layer.data), source_name=layer.name)
        if saved is not None:
            self._log_prediction_activity(f"Saved selected prediction: {saved}")
        self._refresh_after_task_change()

    def _set_project_actions_enabled(self, enabled: bool) -> None:
        for widget in (
            self.tabs,
            self.model_task_combo,
            self.new_task_button,
            self.duplicate_task_button,
            self.rename_task_button,
            self.import_paired_dataset_button,
            self.sam3_mode_combo,
            self.sam3_device_combo,
            self.sam3_direction_combo,
            self.sam3_windows_compat_check,
            self.sam3_no_write_benchmark_check,
            self.sam3_2d_model_edit,
            self.sam3_3d_model_edit,
            self.sam3_2d_model_button,
            self.sam3_3d_model_button,

            self.sam3_image_layer_combo,
            self.sam3_refresh_layers_button,
            self.sam3_prepare_prompt_button,
            self.sam3_positive_point_button,
            self.sam3_negative_point_button,
            self.sam3_run_preview_button,
            self.sam3_clear_preview_button,
            self.sam3_accept_preview_button,
            self.sam3_progress_bar,
            self.sam3_activity_log,
            self.sam3_debug_diagnostics_check,
            self.image_layer_combo,
            self.mask_layer_combo,
            self.refresh_layers_button,
            self.add_mask_button,
            self.open_pair_button,
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
            self.unet_progress_bar,
            self.unet_activity_log,
            self.save_predictions_check,
            self.prediction_layer_combo,
            self.predict_image_layer_combo,
            self.predict_strategy_combo,
            self.run_layer_prediction_button,
            self.run_folder_prediction_button,
            self.save_prediction_button,
            self.save_all_prediction_layers_button,
            self.prediction_progress_bar,
            self.prediction_activity_log,
            self.dataset_table,
            self.prediction_table,
        ):
            widget.setEnabled(enabled)

    def _load_settings_from_project(self) -> None:
        if self.project is None:
            return
        self._loading_project_settings = True
        try:
            self._refresh_model_task_combo()
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
                mask_preparation.get("mode", "merge_nonzero_to_target_class"),
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
            self._restore_combo_data(self.sam3_device_combo, "cuda")
            self._restore_combo_data(self.sam3_direction_combo, sam3_config.get("propagation_direction", "both"))
            runtime_mode = str(sam3_config.get("sam31_runtime_mode", "") or "")
            self.sam3_windows_compat_check.setChecked(
                runtime_mode == "windows_compatibility"
                or bool(sam3_config.get("sam31_windows_compatibility_mode", False))
            )
            self.sam3_no_write_benchmark_check.setChecked(
                bool(sam3_config.get("sam31_no_write_benchmark", False))
            )
            self.sam3_debug_diagnostics_check.setChecked(
                bool(sam3_config.get("sam31_debug_diagnostics", False))
            )
            self.sam3_2d_model_edit.setText(sam3_config.get("sam3_2d_model_dir", ""))
            self.sam3_3d_model_edit.setText(sam3_config.get("sam3_3d_model_dir", ""))
            self._restore_combo_text(self.sam3_image_layer_combo, sam3_config.get("last_image_layer", ""))
        finally:
            self._loading_project_settings = False
        self._apply_active_task_to_controls()
        self._show_dataset_source_warning(self._combo_data(self.dataset_source_combo))
        self._sync_architecture_controls()
        self._refresh_compatibility_status()
        self._refresh_sam3_status()
        self.sam3_prompt_status_label.setText("Select an image layer, then prepare a SAM3 prompt layer.")

    def _refresh_dataset_table(self) -> None:
        if self.project is None:
            return
        pairs = self.project.dataset_pairs()
        self.dataset_table.setRowCount(len(pairs))
        for row, pair in enumerate(pairs):
            preparation = pair.get("mask_preparation", {})
            used = ", ".join(pair.get("used_in_checkpoints", [])) or "no"
            image_text = pair.get("image_layer_name") or Path(pair.get("image_path", "")).name
            mask_text = pair.get("mask_layer_name") or Path(pair.get("mask_path", "")).name
            shape = pair.get("shape", [])
            mask_shape = pair.get("mask_shape", [])
            shape_text = "x".join(str(value) for value in shape) if shape else ""
            if mask_shape and mask_shape != shape:
                shape_text = f"{shape_text} / mask {'x'.join(str(value) for value in mask_shape)}"
            labels = preparation.get("saved_labels", [])
            labels_text = self._mask_labels_text(preparation)
            values = (
                self._short_id(pair.get("pair_id", "")),
                pair.get("source", "manual_label"),
                image_text,
                mask_text,
                shape_text,
                labels_text,
                used,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, pair.get("pair_id", ""))
                if column == 0:
                    item.setToolTip(pair.get("pair_id", ""))
                elif column == 2:
                    item.setToolTip(pair.get("image_path", ""))
                elif column == 3:
                    item.setToolTip(pair.get("mask_path", ""))
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
        outputs = self.project.prediction_outputs()
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
            "auto_expand_multiclass_labels": True,
            "instance_label_threshold": 32,
        }

    def _current_sam3_config(self) -> dict[str, Any]:
        existing = self.project.load_sam3_config() if self.project else {}
        return {
            **existing,
            "default_mode": self._combo_data(self.sam3_mode_combo),
            "sam3_2d_model_dir": self.sam3_2d_model_edit.text().strip(),
            "sam3_3d_model_dir": self.sam3_3d_model_edit.text().strip(),
            "device": "cuda",
            "propagation_direction": self._combo_data(self.sam3_direction_combo),
            "sam31_windows_compatibility_mode": self.sam3_windows_compat_check.isChecked(),
            "sam31_runtime_mode": (
                "windows_compatibility"
                if self.sam3_windows_compat_check.isChecked()
                else "performance"
            ),
            "sam31_no_write_benchmark": self.sam3_no_write_benchmark_check.isChecked(),
            "sam31_debug_diagnostics": self.sam3_debug_diagnostics_check.isChecked(),
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
        nonzero = [label for label in labels if label != 0]
        if len(nonzero) > 32:
            self.detected_labels_label.setText(
                f"Detected labels: {len(nonzero)} instance IDs plus background"
            )
            return
        self.detected_labels_label.setText(f"Detected labels: {self._labels_summary(labels, max_items=18)}")

    def _refresh_sam3_status(self) -> None:
        if self.project is None:
            self.sam3_status_label.setText("SAM3 model not configured.")
            return
        config = self._current_sam3_config()
        ok, message = self._sam3_model_status(config)
        prefix = "Ready" if ok else "Needs setup"
        self.sam3_status_label.setText(f"{prefix}: {message}")
        preview_layer_name = (
            config["propagated_labels_layer_name"]
            if config.get("default_mode") == "3d_multiplex"
            else config["preview_labels_layer_name"]
        )
        self.sam3_preview_layer_label.setText(
            f"Preview labels: {preview_layer_name}"
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

    def _project_path(self, value: str | Path) -> Path | None:
        if self.project is None or not value:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.project.root / candidate

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
            self._set_current_sam3_point_polarity(layer)
            return layer
        data = np.empty((0, ndim), dtype=float)
        layer = self.viewer.add_points(
            data,
            name=name,
            size=8,
            face_color="polarity",
            face_color_cycle=["#2fb344", "#e03131"],
            border_color="black",
            properties={"polarity": np.array([], dtype=object)},
            property_choices={"polarity": ["positive", "negative"]},
        )
        self._set_current_sam3_point_polarity(layer)
        return layer

    def _ensure_shapes_layer(self, name: str, *, ndim: int):
        layer = self._layer_by_name(name)
        if layer is not None:
            if self._is_shapes_layer(layer):
                return layer
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                return layer
        return self.viewer.add_shapes(
            data=[],
            name=name,
            ndim=ndim,
            shape_type="rectangle",
            edge_color="yellow",
            face_color="transparent",
            opacity=0.7,
        )

    @staticmethod
    def _is_shapes_layer(layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower() == "shapes"

    def _ensure_preview_labels_layer(self, image_layer, name: str):
        layer = self._layer_by_name(name)
        shape_attr = getattr(image_layer.data, "shape", None)
        if shape_attr is None:
            shape_attr = np.asarray(image_layer.data).shape
        image_shape = tuple(int(value) for value in shape_attr)
        mode = self._combo_data(self.sam3_mode_combo)
        dtype = np.uint16
        if mode == "3d_multiplex":
            shape = self._sam3_video_output_shape(image_layer.name, image_shape)
            dtype = np.uint32
        else:
            image_ndim = len(image_shape)
            shape = image_shape[-3:] if image_ndim > 3 else image_shape
            if image_ndim >= 2:
                shape = image_shape[-2:]
        if layer is not None:
            layer_data = np.asarray(layer.data)
            if tuple(layer_data.shape) != tuple(shape) or layer_data.dtype != dtype:
                layer.data = np.zeros(shape, dtype=dtype)
                try:
                    layer.refresh()
                except Exception:
                    pass
            return layer
        labels = np.zeros(shape, dtype=dtype)
        return self.viewer.add_labels(labels, name=name)

    def _sam3_video_output_shape(self, layer_name: str, image_shape: tuple[int, ...]) -> tuple[int, int, int]:
        try:
            from napari_sam3_assistant.core.coordinates import (
                infer_image_selection,
                selection_video_output_shape,
            )

            selection = infer_image_selection(
                layer_name=layer_name,
                data_shape=image_shape,
                dims_current_step=tuple(self.viewer.dims.current_step) if self.viewer is not None else None,
            )
            return selection_video_output_shape(selection)
        except Exception:
            if len(image_shape) >= 4 and image_shape[-1] in (1, 3, 4):
                return (image_shape[0], image_shape[-3], image_shape[-2])
            if len(image_shape) >= 3:
                return (image_shape[0], image_shape[-2], image_shape[-1])
            return (1, image_shape[-2], image_shape[-1])

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
    def _class_summary(labels: dict[str, str], max_items: int = 6) -> str:
        if not labels:
            return "none"
        items = [
            f"{key}:{labels[key]}"
            for key in sorted(labels, key=lambda item: int(item))
        ]
        if len(items) <= max_items:
            return ", ".join(items)
        visible = ", ".join(items[:max_items])
        return f"{visible}, +{len(items) - max_items} more"

    @staticmethod
    def _labels_summary(labels: list[int], max_items: int = 10) -> str:
        if not labels:
            return "none"
        preview = ", ".join(str(value) for value in labels[:max_items])
        if len(labels) > max_items:
            preview += ", ..."
        return preview

    def _mask_labels_text(self, preparation: dict[str, Any]) -> str:
        source = preparation.get("source_labels", [])
        saved = preparation.get("saved_labels", [])
        if not source and not saved:
            return "not inspected"
        if preparation.get("source_label_type") == "instance":
            instance_count = len([label for label in source if int(label) != 0])
            target = preparation.get("target_class_name") or "target"
            return f"{instance_count} instances -> {target} ({self._labels_summary(saved)})"
        source_text = self._labels_summary(source)
        saved_text = self._labels_summary(saved)
        if source == saved or not source:
            return saved_text
        return f"{source_text} -> {saved_text}"

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
