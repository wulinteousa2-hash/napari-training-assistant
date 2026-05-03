from __future__ import annotations

from qtpy.QtCore import QObject, Signal

from napari_training_assistant.unet_backend.datasets import PatchDataset
from napari_training_assistant.unet_backend.trainer import TrainConfig, train_model


class TrainWorker(QObject):
    progress = Signal(int, int, object)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(
        self,
        image_paths,
        mask_paths,
        train_cfg: TrainConfig,
        ds_kwargs: dict,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.train_cfg = train_cfg
        self.ds_kwargs = ds_kwargs

    def run(self):
        try:
            ds = PatchDataset(
                image_paths=self.image_paths,
                mask_paths=self.mask_paths,
                **self.ds_kwargs,
            )

            def cb(epoch, total, row):
                self.progress.emit(epoch, total, row)

            model, history = train_model(ds, self.train_cfg, progress_cb=cb)
            self.finished.emit(model, history)
        except Exception as e:
            self.error.emit(str(e))
