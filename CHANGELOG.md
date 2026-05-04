# Changelog

## 0.3.0 - 2026-05-03

- Connected the U-Net training workflow to the project runner instead of only
  registering placeholder checkpoints.
- Added task-scoped U-Net training runs under
  `tasks/<task_id>/history/unet_runs/`, including `best_model.pt`,
  `config.json`, `summary.json`, and `history.csv`.
- Added real PyTorch training support using patch datasets, DataLoader-based
  train/validation splitting, Adam optimization, and checkpoint continuation.
- Added binary BCE + Dice loss and multiclass cross-entropy + Dice loss.
- Added Dice, IoU, and F1 metrics for binary and multiclass training.
- Added 2D and 3D U-Net backend model classes, with 2D as the primary supported
  user-facing workflow.
- Added patch extraction utilities with boundary-aware 2D/3D patch starts and
  optional empty-mask filtering.
- Added conservative 2D/3D augmentation helpers for flips, rotations, and mild
  intensity jitter.
- Added image loading support for TIFF, common image formats, and OME-Zarr
  through shared loader utilities.
- Added Model Task project storage so each segmentation goal owns its own
  dataset, checkpoints, predictions, training history, and benchmark history.
- Added a Model Task bar with task selection, new task creation, duplication,
  renaming, summary display, and paired-dataset import.
- Added paired image/mask dataset import into the active Model Task.
- Added source tracking for dataset pairs, including `sam3_preview`,
  `manual_label`, and `imported_pair`.
- Updated project summaries, dataset counts, checkpoint lookup, prediction
  outputs, and benchmark history to use the active Model Task.
- Added checkpoint registration from trained model output and updated
  task-local `latest.pt` after successful training.
- Added smarter image/mask pairing utilities for two-folder, one-folder, CSV,
  and auto pairing modes, including common suffix conventions such as `_image`,
  `_mask`, `_1`, `_2`, `_label`, and `_seg`; these utilities are available for
  dataset import workflows and future UI expansion.
- Updated README to describe the working training workflow, Model Tasks,
  task-scoped storage, and current 2D-first training scope.
- Add live U-Net training activity log
- Add U-Net prediction UI for layers and input folders

## 0.2.0 - 2026-05-02

- Added SAM3 preview inference for 2D box, 2D points, live points, 2D
  exemplar, and SAM3.1 3D/multiplex workflows.
- Added a SAM3 backend package that delegates SAM3.1 multiplex behavior to
  `napari-sam3-assistant` instead of reimplementing video propagation.
- Added CUDA-only SAM3.1 multiplex validation with SAM3.0 2D inference kept on
  the separate image path.
- Added a persistent SAM3.1 worker thread so the multiplex video predictor can
  stay loaded in one execution thread across propagation runs.
- Added queued SAM3.1 frame writing through a Qt timer to avoid napari/Qt label
  updates throttling SAM3.1 propagation.
- Added SAM3.1 propagation direction control for `both`, `forward`, and
  `backward`.
- Added optional SAM3.1 diagnostics for CUDA/runtime/session state and
  per-frame propagation timing.
- Added optional SAM3.1 no-write benchmark mode for isolating propagation speed
  from napari layer writes.
- Added progress and activity logging for SAM3 model loading, prompt insertion,
  propagation, and frame writes.
- Added SAM3.1 propagated-label output handling for 3D/multiplex results.
- Added tests covering SAM3 backend routing, prompt conversion, SAM3.1 adapter
  caching, predictor configuration, and video-session behavior.

## 0.1.0 - 2026-05-02

- Added persistent Training Project Folder support.
- Added project state tracking for valid, missing, and incomplete projects.
- Added persistent dataset manifest with accepted image/mask pairs.
- Added checkpoint metadata, training run history, benchmark CSV, prediction
  output storage, and latest checkpoint pointer.
- Added tabbed dock widget layout with a compact project status bar.
- Added workflow tabs for SAM3, Dataset, Train, Checkpoints, Predict, and
  Advanced.
- Added compact SAM3 tab with task-driven prompt layer creation for 2D box, 2D
  points, live points, 2D exemplar, and future 3D/multiplex workflows.
- Added persistent `sam3/sam3_config.json` with SAM3 mode, model folders,
  device choice, and layer names.
- Added SAM3 model-folder validation for SAM3.0 image checkpoints and SAM3.1
  multiplex checkpoints.
- Added Accept preview to Dataset flow for `SAM3 preview labels`.
- Moved advanced U-Net architecture and imported-weight controls out of the
  primary training view.
- Replaced dense dataset/checkpoint lists with compact tables.
- Added configurable U-Net architecture schema with 2D support and future-ready
  3D dimensionality metadata.
- Added default `basic_unet` descriptor: depth 4, base channels 32, feature
  channels `[32, 64, 128, 256, 512]`, 18 main convolution layers, 4 upsampling
  layers, and 1 final projection layer.
- Added binary and multiclass output-mode metadata.
- Added binary mask preparation quick fix that merges all nonzero source labels
  into one foreground class.
- Added imported pretrained U-Net model registry under `models/imported/`.
- Added checkpoint architecture compatibility checks.
- Added tests for project creation, reload, mask persistence, checkpoint
  history, latest checkpoint updates, architecture persistence, starting-weight
  persistence, and binary mask preparation.
