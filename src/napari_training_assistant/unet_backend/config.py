from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunConfig:
    mode_2d_or_3d: str
    task_type: str               # "binary" or "multiclass"
    model_name: str              # "unet2d" or "unet3d"
    in_channels: int
    out_channels: int
    patch_xy: int
    patch_z: int | None
    overlap_percent: int
    include_empty_mask: bool
    batch_size: int
    epochs: int
    learning_rate: float
    val_mode: str                # "split" or "kfold"
    val_split: float
    k_folds: int
    use_gpu: bool
    image_dir: str
    mask_dir: str
    output_dir: str

    def to_dict(self) -> dict:
        return asdict(self)