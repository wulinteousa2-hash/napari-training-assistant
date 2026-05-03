# napari-training-assistant

A napari plugin that treats a Training Project Folder as the persistent workspace
for a SAM3-to-U-Net training loop.

The project folder owns accepted image/mask pairs, dataset manifests, training
settings, auto-configuration decisions, checkpoints, benchmark history,
prediction outputs, and logs.

## Current Scope

Version 0.3.0 implements the persistent project workspace, SAM3 preview
annotation flow, Model Task management, task-scoped dataset storage, U-Net
architecture configuration, working PyTorch U-Net training, checkpoint history,
imported starting-weight tracking, prediction output storage, and dataset mask
preparation.

The `Train U-Net` action now runs the project training pipeline, builds patch
datasets from the active Model Task, trains a U-Net with PyTorch, saves the best
model checkpoint, and records run metadata, metrics, benchmark history, and
training history in the project folder.


## Installation

For the full SAM3-to-U-Net workflow, first install and validate SAM3 support by following
the `napari-sam3-assistant` installation guide:

```text
https://github.com/wulinteousa2-hash/napari-sam3-assistant
```

Use that README as the reference for installing:

- `napari-sam3-assistant`
- the SAM3 Python backend
- local SAM3 model weights
- compatible CUDA/PyTorch dependencies for SAM3 and SAM3.1 multiplex workflows

After `napari-sam3-assistant` is working in your napari environment, install
`napari-training-assistant` into the same environment:

```bash
git clone https://github.com/wulinteousa2-hash/napari-training-assistant.git
cd napari-training-assistant
pip install -e .
```

Start napari and open the plugin from the napari Plugins menu:

```bash
napari
```

Inside `napari-training-assistant`, select the SAM3 model folders in the
**SAM3** tab:

- 2D SAM3 modes require a SAM3.0 image model folder containing `sam3.pt` or
  `model.safetensors`.
- SAM3.1 3D/multiplex mode requires a SAM3.1 model folder containing
  `sam3.1_multiplex.pt` and CUDA.

U-Net training from existing image/mask pairs can run without using SAM3 during
the workflow. However, for the full SAM3-assisted mask-generation workflow, this
plugin expects `napari-sam3-assistant` and a working SAM3 installation to already
be available in the same environment.

## Dependencies

For the full workflow, this plugin should be installed into the same environment
where `napari-sam3-assistant` and SAM3 already work. The SAM3 installation guide
is intentionally not duplicated here because SAM3 setup depends on OS, CUDA,
PyTorch, and model version.

The training assistant itself uses:

- `napari` and `qtpy` for the dock widget UI
- `numpy` for array handling
- `tifffile` and `Pillow` for image I/O
- `dask`, `zarr`, and `ome-zarr` for OME-Zarr loading
- `torch` for U-Net training and inference
- `napari-sam3-assistant` for SAM3-assisted prompt collection, SAM3.1 multiplex
  handoff, and napari layer writing
- a working SAM3 backend and local SAM3 model folders for SAM3 preview and
  propagation

Practical rule:

- Follow the `napari-sam3-assistant` README first.
- Confirm SAM3 works there.
- Then install `napari-training-assistant` in the same environment.

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

- **SAM3**: configure SAM3 model folders, choose a prompt mode, auto-create
  prompt layers, prepare preview labels, and accept preview masks into the
  persistent dataset.
- **Dataset**: choose image/mask layers, prepare masks, add accepted pairs,
  reopen selected pairs, and inspect the compact dataset table.
- **Train**: choose training mode, dataset source, starting point, and core
  training parameters.
- **Checkpoints**: inspect checkpoint history and choose a checkpoint for
  continued training.
- **Predict**: save prediction layers into the project.
- **Advanced**: edit U-Net architecture, import pretrained weights, and update
  project notes.

Training-related actions are disabled until the user selects or creates a
Training Project Folder.

## Model Task Workflow

A Model Task defines one segmentation target inside a Training Project. Examples
include `myelin + background`, `axon + background`, or a multiclass task such as
`background + myelin + axon`.

Each Model Task keeps its own dataset manifest, copied image/mask pairs,
checkpoints, predictions, benchmark history, and training-run history. This keeps
separate segmentation goals from mixing their data or model outputs.

The Model Task bar supports:

- creating a fresh task
- duplicating a task configuration
- renaming a task
- switching the active task
- importing an existing paired image/mask dataset into the active task

The active Model Task controls which dataset pairs are used for training and
where new checkpoints and predictions are saved.

## Training Project Folder

The Training Project Folder is the persistent workspace. Users should select or
create it before adding masks, training U-Net, or saving predictions.

Expected structure:

```text
training_project/
    project_config.json

    architecture/
        architecture_config.json

    sam3/
        sam3_config.json

    tasks/
        tasks.json

        default_binary/
            task_config.json

            dataset/
                images/
                masks/
                manifest.json

            checkpoints/
                checkpoints.json
                latest.pt

            predictions/
                prediction_001.tif
                prediction_002.tif

            history/
                training_runs.json
                benchmark_history.csv
                unet_runs/
                    unet_run_001_YYYYMMDDTHHMMSSZ/
                        best_model.pt
                        config.json
                        summary.json
                        history.csv

    models/
        imported/
        model_registry.json
        starting_weights_config.json

    checkpoints/
        latest.pt

    logs/
        training.log
```

The root-level project files preserve global settings. The `tasks/` folder owns
the active training data, task-specific checkpoints, predictions, and run
history.

Reopening the same project restores Model Tasks, dataset history, training
settings, U-Net architecture settings, starting-weight choice, checkpoint
history, latest checkpoint pointer, benchmark history, and previously selected
layer names.

## SAM3 Annotation Tab

The SAM3 tab is task-driven. The user selects an image layer and a prompt mode;
the plugin creates or reuses the expected napari prompt layer automatically.

Supported prompt modes in the compact UI:

- **2D box**: creates/selects `SAM3 boxes` as a Shapes layer in rectangle mode.
- **2D points**: creates/selects `SAM3 points` as a Points layer.
- **Live points**: creates/selects `SAM3 live points` for immediate
  point-driven preview.
- **2D exemplar**: creates/selects `SAM3 exemplar boxes` as a Shapes layer.
- **3D / multiplex**: creates/selects `SAM3 3D prompts` for SAM3.1 multiplex
  propagation.

Model folder expectations:

- 2D modes use a SAM3.0 image model folder containing `sam3.pt` or
  `model.safetensors`.
- 3D / multiplex mode uses a SAM3.1 model folder containing
  `sam3.1_multiplex.pt`.
- CPU is only treated as valid for 2D mode. SAM3.1 multiplex requires CUDA.

The SAM3 tab prepares layers, validates model folders, runs SAM3 previews, and
writes results to `SAM3 preview labels` for 2D modes or `SAM3 propagated labels`
for SAM3.1 multiplex propagation. SAM3.1 multiplex uses CUDA and delegates the
video propagation flow to `napari-sam3-assistant`; frame results are queued and
written back to napari labels through a Qt timer so layer refreshes do not
throttle propagation. The **Accept preview to Dataset** button routes the
preview through the same persistent mask-preparation path as the Dataset tab.

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

## PyTorch Training Output

The `Train U-Net` action runs PyTorch training for the active Model Task. The
training runner builds a patch dataset from selected image/mask pairs, trains
with the configured U-Net settings, saves the best model state, and registers a
checkpoint in the task history.

For each U-Net run, the plugin writes:

- `best_model.pt` for the best validation checkpoint
- `config.json` for the resolved run configuration
- `summary.json` for image count, patch count, best epoch, Dice, and IoU
- `history.csv` for per-epoch loss and metric history
- task checkpoint metadata and benchmark history

Current supported production path: 2D U-Net training. 3D U-Net code paths are
reserved for future expansion and should be treated as experimental until the
full UI workflow is validated.

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

Every successful training run creates a new numbered checkpoint. Previous
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
`src/napari_training_assistant/unet.py`. The project-level training runner lives
in `src/napari_training_assistant/unet_backend/project_runner.py` and connects
the active Model Task to patch dataset creation, PyTorch training, run-output
files, checkpoint registration, and benchmark history.

SAM3.1 multiplex behavior is intentionally delegated to `napari-sam3-assistant`
for prompt collection, adapter behavior, video-session semantics, and napari
layer writing. `napari-training-assistant` uses SAM3 as an annotation source and
keeps U-Net training as the persistent trainable-model workflow.
