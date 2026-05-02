from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

from .errors import SAM3ModelError


class SAM3ModelLoader:
    """Load and cache SAM3 models."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.processor: Any | None = None
        self.video_predictor: Any | None = None
        self.loaded_key: tuple[str, str, str, bool, float] | None = None
        self.loaded_video_key: tuple[str, str, bool] | None = None
        self._dtype_hook_handles: list[Any] = []

    def load_2d_model(
        self,
        model_dir: str | Path,
        device: str,
        *,
        confidence_threshold: float = 0.5,
        enable_instance_interactivity: bool = True,
        compile_model: bool = False,
    ) -> tuple[Any, Any]:
        model_path = Path(model_dir).expanduser()

        if not model_path.exists():
            raise SAM3ModelError(f"SAM3 model folder does not exist: {model_path}")

        if not model_path.is_dir():
            raise SAM3ModelError(f"SAM3 model path is not a folder: {model_path}")

        checkpoint_path = self._checkpoint_path(model_path, ("sam3.pt", "model.safetensors"))
        if checkpoint_path is None:
            raise SAM3ModelError(
                "SAM3 2D model folder must contain sam3.pt or model.safetensors."
            )

        key = (
            str(checkpoint_path.resolve()),
            device,
            "2d",
            bool(enable_instance_interactivity),
            float(confidence_threshold),
        )
        if self.loaded_key == key:
            return self.model, self.processor

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except Exception as exc:
            raise SAM3ModelError(
                "The local SAM3 package could not be imported. Install the SAM3 package "
                "in the same Python environment as napari."
            ) from exc

        bpe_path = self._bpe_path(model_path)

        try:
            self._remove_dtype_hooks()
            self.model = build_sam3_image_model(
                checkpoint_path=str(checkpoint_path),
                bpe_path=str(bpe_path) if bpe_path else None,
                load_from_HF=False,
                enable_inst_interactivity=enable_instance_interactivity,
                compile=compile_model,
                device=device,
            )
            self.processor = Sam3Processor(
                self.model,
                device=device,
                confidence_threshold=float(confidence_threshold),
            )
            if device == "cpu":
                self._force_float32(self.model)
                self._install_cpu_float32_hooks(self.model)
        except Exception as exc:
            self._remove_dtype_hooks()
            self.model = None
            self.processor = None
            self.loaded_key = None
            raise SAM3ModelError(f"Could not load SAM3 image model: {exc}") from exc

        self.loaded_key = key

        return self.model, self.processor

    def load_video_model(
        self,
        model_dir: str | Path,
        device: str,
        *,
        compile_model: bool = False,
    ) -> Any:
        if device == "cpu":
            raise SAM3ModelError("SAM3 visual exemplar prompts require the CUDA video predictor.")

        model_path = Path(model_dir).expanduser()

        if not model_path.exists():
            raise SAM3ModelError(f"SAM3 model folder does not exist: {model_path}")

        if not model_path.is_dir():
            raise SAM3ModelError(f"SAM3 model path is not a folder: {model_path}")

        checkpoint_path = self._checkpoint_path(model_path, ("sam3.pt", "model.safetensors"))
        if checkpoint_path is None:
            raise SAM3ModelError(
                "SAM3 exemplar mode needs a SAM3.0 model folder containing sam3.pt or model.safetensors."
            )

        key = (str(checkpoint_path.resolve()), device, bool(compile_model))
        if self.loaded_video_key == key and self.video_predictor is not None:
            return self.video_predictor

        try:
            import torch
            from sam3.model_builder import build_sam3_video_predictor
        except Exception as exc:
            raise SAM3ModelError(
                "The local SAM3 video predictor could not be imported. Install the SAM3 package "
                "in the same Python environment as napari."
            ) from exc

        if not torch.cuda.is_available():
            raise SAM3ModelError("SAM3 visual exemplar prompts require CUDA.")

        parsed_device = torch.device(device)
        if parsed_device.type != "cuda":
            raise SAM3ModelError(f"SAM3 visual exemplar prompts require CUDA, got {device!r}.")
        gpu_id = parsed_device.index
        if gpu_id is None:
            try:
                gpu_id = torch.cuda.current_device()
            except Exception:
                gpu_id = 0

        bpe_path = self._bpe_path(model_path)
        try:
            torch.cuda.set_device(gpu_id)
            self.video_predictor = build_sam3_video_predictor(
                checkpoint_path=str(checkpoint_path),
                bpe_path=str(bpe_path) if bpe_path else None,
                compile=compile_model,
                gpus_to_use=[gpu_id],
            )
        except Exception as exc:
            self.video_predictor = None
            self.loaded_video_key = None
            raise SAM3ModelError(f"Could not load SAM3 video predictor for exemplar mode: {exc}") from exc

        self.loaded_video_key = key
        return self.video_predictor

    def _force_float32(self, model: Any) -> None:
        try:
            model.float()
        except Exception:
            pass

    def _install_cpu_float32_hooks(self, model: Any) -> None:
        try:
            import torch
            from torch import nn
        except Exception:
            return

        if not isinstance(model, nn.Module):
            return

        for module in model.modules():
            if isinstance(module, nn.Linear):
                self._dtype_hook_handles.append(
                    module.register_forward_pre_hook(self._linear_float32_pre_hook)
                )
            elif isinstance(module, nn.MultiheadAttention):
                self._dtype_hook_handles.append(
                    module.register_forward_pre_hook(self._mha_float32_pre_hook)
                )

    def _remove_dtype_hooks(self) -> None:
        for handle in self._dtype_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._dtype_hook_handles = []

    def _linear_float32_pre_hook(self, module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            return inputs
        return (self._cast_float_tensor(inputs[0], module.weight.dtype), *inputs[1:])

    def _mha_float32_pre_hook(self, module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(inputs) < 3:
            return inputs
        dtype = module.in_proj_weight.dtype
        return (
            self._cast_float_tensor(inputs[0], dtype),
            self._cast_float_tensor(inputs[1], dtype),
            self._cast_float_tensor(inputs[2], dtype),
            *inputs[3:],
        )

    def _cast_float_tensor(self, value: Any, dtype: Any) -> Any:
        try:
            import torch
        except Exception:
            return value
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value

    @staticmethod
    def _checkpoint_path(model_path: Path, names: tuple[str, ...]) -> Path | None:
        for name in names:
            candidate = model_path / name
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _bpe_path(model_path: Path) -> Path | None:
        for name in ("bpe_simple_vocab_16e6.txt.gz", "merges.txt.gz"):
            candidate = model_path / name
            if candidate.exists():
                return candidate

        merges = model_path / "merges.txt"
        if not merges.exists():
            return None

        gz_path = model_path / "bpe_simple_vocab_16e6.txt.gz"
        try:
            with open(merges, "rb") as src, gzip.open(gz_path, "wb") as dst:
                dst.write(src.read())
        except Exception:
            return None
        return gz_path
