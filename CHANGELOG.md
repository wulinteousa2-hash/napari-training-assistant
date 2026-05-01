# Changelog

## 0.1.0 - Unreleased

- Added persistent Training Project Folder support.
- Added project state tracking for valid, missing, and incomplete projects.
- Added persistent dataset manifest with accepted image/mask pairs.
- Added checkpoint metadata, training run history, benchmark CSV, prediction
  output storage, and latest checkpoint pointer.
- Added tabbed dock widget layout with a compact project status bar.
- Added workflow tabs for Dataset, Train, Checkpoints, Predict, and Advanced.
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

Known limitation:

- The real PyTorch training loop is not connected yet. The current train action
  registers a placeholder checkpoint for persistence-flow validation.
