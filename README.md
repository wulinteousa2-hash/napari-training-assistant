# napari-training-assistant

A napari plugin that treats a Training Project Folder as the persistent workspace
for a SAM3-to-U-Net training loop.

The project folder owns accepted image/mask pairs, dataset manifests, training
settings, auto-configuration decisions, checkpoints, benchmark history,
prediction outputs, and logs.

## Current Scope

This plugin currently implements the persistent project workspace, U-Net
architecture configuration, checkpoint metadata, imported starting-weight
tracking, prediction output storage, and dataset mask preparation.

The actual PyTorch training runner is not wired in yet. The current `Train
U-Net` action registers a placeholder checkpoint so the persistence and UI flow
can be validated before connecting the real trainer.

## User Interface

The dock widget is organized as a tabbed workflow with a compact project status
bar at the top.

Always visible:

- Project selector
- Short project path
- Project state
- Dataset count
- Latest checkpoint
- Latest benchmark summary

Tabs:

- **Dataset**: choose image/mask layers, prepare masks, add accepted pairs, and
  inspect the compact dataset table.
- **Train**: choose training mode, dataset source, starting point, and core
  training parameters.
- **Checkpoints**: inspect checkpoint history and choose a checkpoint for
  continued training.
- **Predict**: save prediction layers into the project.
- **Advanced**: edit U-Net architecture, import pretrained weights, and update
  project notes.

Training-related actions are disabled until the user selects or creates a
Training Project Folder.

## Training Project Folder

The Training Project Folder is the persistent workspace. Users should select or
create it before adding masks, training U-Net, or saving predictions.

Expected structure:

```text
training_project/
    project_config.json

    architecture/
        architecture_config.json

    dataset/
        images/
        masks/
        manifest.json

    models/
        imported/
        model_registry.json
        starting_weights_config.json

    checkpoints/
        checkpoints.json
        latest.pt
        unet_run_001.pt
        unet_run_002.pt

    predictions/
        prediction_001.tif
        prediction_002.tif

    history/
        training_runs.json
        benchmark_history.csv

    logs/
        training.log
```

Reopening the same project restores dataset history, training settings, U-Net
architecture settings, starting-weight choice, checkpoint history, latest
checkpoint pointer, benchmark history, and previously selected layer names.

## U-Net Architecture

The trainable model is a U-Net-family segmentation model. SAM3 is treated as an
annotation source for generating candidate masks, not as the trainable model.

The default backend is:

```json
{
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
  "activation": "sigmoid",
  "loss": "bce_dice",
  "threshold": 0.5
}
```

For the default 2D U-Net:

- Feature channels are `[32, 64, 128, 256, 512]`.
- There are 4 encoder levels, 1 bottleneck, and 4 decoder levels.
- Each block uses 2 convolution layers.
- The architecture has 18 main convolution layers, 4 upsampling layers, and 1
  final projection layer.

The schema already records dimensionality so future 3D U-Net support can be
added without changing the project folder structure. The UI currently marks 3D
U-Net options as future/unsupported.

## Binary And Multiclass Masks

Binary mode is the default:

- `0` means background.
- nonzero source labels can be merged into `1` foreground.
- model output has 1 channel.
- prediction uses sigmoid plus threshold.

Multiclass mode is supported in the project schema:

- masks should contain integer labels from `0` to `num_classes - 1`.
- model output channels equal `num_classes`.
- prediction uses channel argmax.

## Mask Preparation Quick Fix

SAM-style masks may contain multiple instance IDs for the same semantic object:

```text
0 = background
1 = object instance A
2 = object instance B
3 = object instance C
```

For binary U-Net training, the default quick fix is:

```python
mask = (mask > 0).astype("uint8")
```

This saves the training mask as:

```text
0 = background
1 = foreground
```

The dataset manifest records the source labels, saved labels, target class name,
and label transform so the conversion is auditable.

## Starting Weights

Supported starting-weight choices in the UI:

- Train from scratch
- Continue from latest project checkpoint
- Continue from selected project checkpoint
- Start from imported pretrained U-Net

Imported `.pt` and `.pth` files are copied into `models/imported/` and tracked
in `models/model_registry.json`. A PyTorch checkpoint alone is not considered
self-describing; the project also stores the architecture config used with that
checkpoint.

Checkpoint compatibility is checked against:

- backend
- dimensionality
- preset
- depth
- base channels
- normalization
- upsampling
- input channels
- output mode
- number of classes
- output channels

## Checkpoint Metadata

Every successful training run should create a new numbered checkpoint. Previous
checkpoints are not overwritten by default. `latest.pt` is only updated after a
successful checkpoint registration.

Each checkpoint records:

- checkpoint ID
- parent checkpoint ID
- training mode
- dataset pair IDs used
- image and patch counts
- train/validation split
- loss metrics
- Dice/IoU metrics, when available
- timestamp
- checkpoint path
- architecture snapshot
- starting weights snapshot
- architecture summary

## Development Notes

The default U-Net descriptor and lazy PyTorch builder live in
`src/napari_training_assistant/unet.py`. The builder imports PyTorch lazily so
the plugin metadata and persistence layer can still import in environments where
Torch is not installed.
