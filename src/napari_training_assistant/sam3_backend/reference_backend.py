from __future__ import annotations
import os
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from .errors import SAM3InferenceError
from .model_loader import SAM3ModelLoader
from .prompt_converter import convert_box_prompt, convert_point_prompt

class ReferenceSAM3Backend:
    """Thin wrapper around napari-sam3-assistant's SAM3 runtime.

    The training assistant should not reimplement SAM3.1/video semantics.  This
    class keeps model selection, prompt bundle construction, session start,
    prompt insertion, and propagation aligned with napari-sam3-assistant.
    """
    def log(self, message: str) -> None:
        """Send diagnostic messages to the UI if a logger is attached."""
        if self.logger is not None:
            try:
                self.logger(message)
            except Exception:
                pass
    def __init__(self, loader: SAM3ModelLoader | None = None) -> None:
        self.loader = loader or SAM3ModelLoader()
        self.adapter: Any | None = None
        self.adapter_key: tuple[Any, ...] | None = None
        self.video_session: Any | None = None
        self.video_session_key: tuple[Any, ...] | None = None
        self.logger: Callable[[str], None] | None = None
        self.video_predictor_thread_id: int | None = None
        self.frame_dir_key: tuple[Any, ...] | None = None
        self.frame_dir: TemporaryDirectory[str] | None = None

    def preload_multiplex_model(
        self,
        *,
        model_dir: str,
        device: str,
        threshold: float,
        compile_model: bool = False,
        windows_compatibility_mode: bool = False,
    ) -> None:
        t0 = time.perf_counter()
        adapter = self._ensure_adapter(
            model_dir=model_dir,
            model_type="sam3.1",
            device=device,
            threshold=threshold,
            compile_model=compile_model,
        )
        self.log(f"SAM3.1 ensure_adapter finished in {time.perf_counter() - t0:.2f} sec.")

        if getattr(adapter, "video_predictor", None) is None:
            t1 = time.perf_counter()
            self._log_cuda_diagnostics("before adapter.load_video()")
            adapter.load_video()
            self.video_predictor_thread_id = threading.get_ident()
            self._log_cuda_diagnostics("after adapter.load_video()")
            self.log(f"SAM3.1 model load finished in {time.perf_counter() - t1:.2f} sec.")
        else:
            self.log("SAM3.1 video predictor already loaded; reusing cached predictor.")

        t2 = time.perf_counter()
        self.configure_multiplex_predictor(
            adapter,
            windows_compatibility_mode=windows_compatibility_mode,
        )
        self.log(f"SAM3.1 predictor configuration finished in {time.perf_counter() - t2:.2f} sec.")

    def _log_timing(self, label: str, start: float) -> None:
        self.log(f"{label} finished in {time.perf_counter() - start:.2f} sec.")

    def _log_cuda_diagnostics(self, stage: str) -> None:
        try:
            import torch

            available = bool(torch.cuda.is_available())
            if available:
                device_index = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_index)
                capability = torch.cuda.get_device_capability(device_index)
                allocated = int(torch.cuda.memory_allocated(device_index))
                reserved = int(torch.cuda.memory_reserved(device_index))
            else:
                device_index = None
                device_name = ""
                capability = None
                allocated = 0
                reserved = 0
            self.log(
                "SAM3.1 CUDA diagnostics "
                f"({stage}): available={available}, current_device={device_index}, "
                f"name={device_name}, torch_cuda={getattr(torch.version, 'cuda', None)}, "
                f"capability={capability}, allocated={allocated}, reserved={reserved}."
            )
        except Exception as exc:
            self.log(f"SAM3.1 CUDA diagnostics ({stage}) failed: {exc}")

    def log_runtime_diagnostics(self, adapter: Any, *, stage: str) -> None:
        predictor = getattr(adapter, "video_predictor", None)
        attrs = {
            "predictor_type": type(predictor).__name__ if predictor is not None else None,
            "model_type": type(getattr(predictor, "model", None)).__name__ if predictor is not None else None,
            "async_loading_frames": getattr(predictor, "async_loading_frames", None),
            "video_loader_type": getattr(predictor, "video_loader_type", None),
            "thread_id": threading.get_ident(),
            "loaded_thread_id": self.video_predictor_thread_id,
            "env_USE_PERFLIB": os.environ.get("USE_PERFLIB"),
            "env_SAM3_FORCE_SDPA_FALLBACK": os.environ.get("SAM3_FORCE_SDPA_FALLBACK"),
            "env_SAM3_FORCE_SLOW_EDT": os.environ.get("SAM3_FORCE_SLOW_EDT"),
        }
        try:
            import torch

            attrs.update(
                {
                    "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                    "math_sdp": torch.backends.cuda.math_sdp_enabled(),
                    "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                }
            )
        except Exception as exc:
            attrs["torch_backend_error"] = str(exc)
        try:
            from sam3 import perflib

            attrs["perflib_enabled"] = getattr(perflib, "is_enabled", None)
        except Exception as exc:
            attrs["perflib_error"] = str(exc)
        self.log("SAM3.1 runtime diagnostics " f"({stage}): {attrs}")

    def log_session_diagnostics(self, adapter: Any, session: Any, *, stage: str) -> None:
        predictor = getattr(adapter, "video_predictor", None)
        sessions = getattr(predictor, "_all_inference_states", None)
        if not isinstance(sessions, dict) or session is None:
            self.log(f"SAM3.1 session diagnostics ({stage}): no session state dictionary.")
            return
        record = sessions.get(getattr(session, "session_id", None))
        state = record.get("state") if isinstance(record, dict) else None
        if not isinstance(state, dict):
            self.log(f"SAM3.1 session diagnostics ({stage}): no state for session.")
            return
        images = state.get("images")
        first_image = self._first_sequence_item(images)
        feature_cache = state.get("feature_cache")
        cached_features = state.get("cached_features")
        sam2_states = state.get("sam2_inference_states")
        first_sam2_state = self._first_sequence_item(sam2_states)
        details = {
            "session_id": getattr(session, "session_id", None),
            "state_keys": sorted(str(key) for key in state.keys())[:40],
            "num_frames": state.get("num_frames"),
            "image_size": state.get("image_size"),
            "orig_height": state.get("orig_height"),
            "orig_width": state.get("orig_width"),
            "video_height": state.get("video_height"),
            "video_width": state.get("video_width"),
            "device": str(state.get("device")),
            "storage_device": str(state.get("storage_device")),
            "offload_video_to_cpu": state.get("offload_video_to_cpu"),
            "offload_state_to_cpu": state.get("offload_state_to_cpu"),
            "images_type": type(images).__name__ if images is not None else None,
            "images_len": self._safe_len(images),
            "first_image": self._value_summary(first_image),
            "feature_cache_type": type(feature_cache).__name__ if feature_cache is not None else None,
            "feature_cache_len": self._safe_len(feature_cache),
            "feature_cache_keys": self._mapping_keys(feature_cache),
            "cached_features_type": type(cached_features).__name__ if cached_features is not None else None,
            "cached_features_len": self._safe_len(cached_features),
            "cached_features_keys": self._mapping_keys(cached_features),
            "sam2_inference_states_len": self._safe_len(sam2_states),
            "first_sam2_state_keys": self._mapping_keys(first_sam2_state),
            "obj_ids": self._short_value(state.get("obj_ids")),
            "obj_id_to_idx": self._short_value(state.get("obj_id_to_idx")),
            "tracker_metadata_keys": self._mapping_keys(state.get("tracker_metadata")),
        }
        self.log(f"SAM3.1 session diagnostics ({stage}): {details}")

    @staticmethod
    def _safe_len(value: Any) -> int | None:
        try:
            return len(value) if value is not None and hasattr(value, "__len__") else None
        except Exception:
            return None

    @staticmethod
    def _first_sequence_item(value: Any) -> Any:
        try:
            if value is not None and len(value):
                return value[0]
        except Exception:
            return None
        return None

    @staticmethod
    def _mapping_keys(value: Any) -> list[str] | None:
        if not isinstance(value, dict):
            return None
        return sorted(str(key) for key in value.keys())[:30]

    @classmethod
    def _value_summary(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "type": type(value).__name__,
            "shape": tuple(value.shape) if hasattr(value, "shape") else None,
            "device": str(getattr(value, "device", "")),
            "dtype": str(getattr(value, "dtype", "")),
            "len": cls._safe_len(value),
        }

    @staticmethod
    def _short_value(value: Any) -> str | None:
        if value is None:
            return None
        text = repr(value)
        return text if len(text) <= 300 else text[:300] + "..."

    def log_prompt_diagnostics(self, bundle: Any) -> None:
        try:
            from napari_sam3_assistant.core.coordinates import CoordinateMapper

            height = bundle.image.data_shape[bundle.image.spatial_axes[0]]
            width = bundle.image.data_shape[bundle.image.spatial_axes[1]]
            mapper = CoordinateMapper(bundle.image)
            boxes = [
                {
                    "object_id": getattr(box, "object_id", None),
                    "yx": (float(box.y0), float(box.x0), float(box.y1), float(box.x1)),
                    "xyxy": mapper.box_to_xyxy(box),
                    "norm_xywh": mapper.box_to_normalized_xywh(box, (height, width)),
                }
                for box in getattr(bundle, "boxes", [])
            ]
            self.log(
                "SAM3.1 prompt diagnostics: "
                f"frame={bundle.image.frame_index}, image_hw={(height, width)}, boxes={boxes}, "
                f"points={len(getattr(bundle, 'points', []) or [])}."
            )
        except Exception as exc:
            self.log(f"SAM3.1 prompt diagnostics failed: {exc}")

    def configure_multiplex_predictor(
        self,
        adapter: Any,
        *,
        windows_compatibility_mode: bool = False,
    ) -> None:
        predictor = getattr(adapter, "video_predictor", None)
        if predictor is None:
            return
        if (
            sys.platform.startswith("win")
            and windows_compatibility_mode
            and hasattr(predictor, "async_loading_frames")
        ):
            predictor.async_loading_frames = False

    def iter_multiplex(
        self,
        *,
        image: Any,
        image_layer_name: str,
        dims_current_step: tuple[int, ...] | None,
        model_dir: str,
        device: str,
        prompt_data: Any,
        prompt_bundle: Any | None,
        threshold: float,
        compile_model: bool = False,
        direction: str = "both",
        windows_compatibility_mode: bool = False,
    ) -> Iterator[Any]:
        from napari_sam3_assistant.core.coordinates import infer_image_selection

        adapter = self._ensure_adapter(
            model_dir=model_dir,
            model_type="sam3.1",
            device=device,
            threshold=threshold,
            compile_model=compile_model,
        )
        selection = infer_image_selection(
            layer_name=image_layer_name,
            data_shape=self._data_shape(image),
            dims_current_step=dims_current_step,
        )
        if selection.frame_axis is None:
            raise SAM3InferenceError(
                "SAM3.1 multiplex requires a stack or video image layer with a frame axis."
            )

        bundle = prompt_bundle or self._multiplex_bundle_from_prompt_data(
            selection=selection,
            prompt_data=prompt_data,
        )
        if not bundle.has_prompt():
            raise SAM3InferenceError("No 3D multiplex box or point prompt found.")

        try:
            if getattr(adapter, "video_predictor", None) is None:
                self._log_cuda_diagnostics("before adapter.load_video()")
                load_t0 = time.perf_counter()
                adapter.load_video()
                self.video_predictor_thread_id = threading.get_ident()
                self._log_timing("SAM3.1 model load", load_t0)
                self._log_cuda_diagnostics("after adapter.load_video()")
            else:
                self.log("SAM3.1 video predictor already loaded; reusing cached predictor.")
            config_t0 = time.perf_counter()
            self.configure_multiplex_predictor(
                adapter,
                windows_compatibility_mode=windows_compatibility_mode,
            )
            self._log_timing("SAM3.1 predictor configuration", config_t0)
            session_key = self.video_key(
                image=image,
                image_layer_name=image_layer_name,
                model_dir=model_dir,
                device=device,
                threshold=threshold,
                compile_model=compile_model,
            )
            self.log(f"SAM3.1 image source: {self.describe_image_source(image)}")
            session_t0 = time.perf_counter()
            session = adapter.start_video_session(image, bundle)
            self._log_timing("SAM3.1 session start", session_t0)
            self.log(
                "SAM3.1 session ready: "
                f"{session.session_id}; prompt_frame={bundle.image.frame_index or 0}; "
                f"boxes={len(getattr(bundle, 'boxes', []) or [])}; "
                f"points={len(getattr(bundle, 'points', []) or [])}."
            )
            self.video_session_key = session_key
            self.video_session = session
            prompt_t0 = time.perf_counter()
            prompt_result = adapter.add_video_prompt(bundle, session)
            self._log_timing("SAM3.1 prompt insertion", prompt_t0)
            self._mark_video_result(prompt_result, stage="prompt")
            yield prompt_result
            self._log_cuda_diagnostics("before propagation")
            propagation_t0 = time.perf_counter()
            for result in adapter.propagate_video(bundle, session, direction=direction):
                self._mark_video_result(result, stage="propagation")
                yield result
            self._log_timing("SAM3.1 propagation", propagation_t0)
        except Exception as exc:
            raise SAM3InferenceError(f"SAM3.1 multiplex inference failed: {exc}") from exc

    def run_image(
        self,
        *,
        image: np.ndarray,
        image_layer_name: str,
        dims_current_step: tuple[int, ...] | None,
        mode: str,
        model_dir: str,
        device: str,
        prompt_data: Any,
        threshold: float,
        compile_model: bool,
    ) -> Any:
        from napari_sam3_assistant.core.coordinates import infer_image_selection
        from napari_sam3_assistant.core.models import Sam3Task

        adapter = self._ensure_adapter(
            model_dir=model_dir,
            model_type="sam3",
            device=device,
            threshold=threshold,
            compile_model=compile_model,
        )
        selection = infer_image_selection(
            layer_name=image_layer_name,
            data_shape=tuple(int(value) for value in np.asarray(image).shape),
            dims_current_step=dims_current_step,
        )
        task = {
            "2d_box": Sam3Task.SEGMENT_2D,
            "2d_exemplar": Sam3Task.EXEMPLAR,
            "2d_points": Sam3Task.SEGMENT_2D,
            "live_points": Sam3Task.REFINE,
        }.get(mode)
        if task is None:
            raise SAM3InferenceError(
                f"Unsupported SAM3 preview mode for this backend version: {mode}. "
                "Use a SAM3.0 2D box, point, live-point, or exemplar preview."
            )
        bundle = self._image_bundle_from_prompt_data(
            task=task,
            selection=selection,
            mode=mode,
            prompt_data=prompt_data,
        )
        return adapter.run_image(np.asarray(image), bundle)

    def unload(self) -> None:
        if self.adapter is not None:
            try:
                self.adapter.unload()
            except Exception:
                pass
        self.adapter = None
        self.adapter_key = None
        self.video_session = None
        self.video_session_key = None
        self.video_predictor_thread_id = None
        self._clear_frame_dir()

    def video_predictor_loaded_in_current_thread(self, adapter: Any) -> bool:
        if getattr(adapter, "video_predictor", None) is None:
            return False
        return self.video_predictor_thread_id == threading.get_ident()

    def _ensure_adapter(
        self,
        *,
        model_dir: str,
        model_type: str,
        device: str,
        threshold: float,
        compile_model: bool,
    ) -> Any:
        from napari_sam3_assistant.adapters import Sam3Adapter, Sam3AdapterConfig

        model_path = Path(model_dir).expanduser()
        if model_type == "sam3.1":
            checkpoint_path = self.loader._checkpoint_path(model_path, ("sam3.1_multiplex.pt",))
            bpe_path = None
            missing = "SAM3.1 multiplex model folder must contain sam3.1_multiplex.pt."
        else:
            checkpoint_path = self.loader._checkpoint_path(model_path, ("sam3.pt", "model.safetensors"))
            bpe_path = self.loader._bpe_path(model_path)
            missing = "SAM3 2D model folder must contain sam3.pt or model.safetensors."
        if checkpoint_path is None:
            raise SAM3InferenceError(missing)

        if model_type == "sam3.1":
            key = (
                str(checkpoint_path.resolve()),
                model_type,
                device,
                bool(compile_model),
            )
        else:
            key = (
                str(checkpoint_path.resolve()),
                str(bpe_path.resolve()) if bpe_path else "",
                model_type,
                device,
                float(threshold),
                bool(compile_model),
            )
        if self.adapter is None or self.adapter_key != key:
            self.log("SAM3.1 adapter cache miss; creating adapter" if model_type == "sam3.1" else "SAM3 adapter cache miss; creating adapter")
            self.unload()
            self.adapter = Sam3Adapter(
                Sam3AdapterConfig(
                    checkpoint_path=checkpoint_path,
                    bpe_path=bpe_path,
                    device=device,
                    confidence_threshold=float(threshold),
                    compile_model=bool(compile_model),
                    load_from_hf=False,
                )
            )
            self.adapter_key = key
        elif model_type == "sam3.1":
            self.log("SAM3.1 adapter cache hit")
        return self.adapter

    def video_key(
        self,
        *,
        image: Any,
        image_layer_name: str,
        model_dir: str,
        device: str,
        threshold: float,
        compile_model: bool,
    ) -> tuple[Any, ...]:
        shape = self._data_shape(image)
        dtype = getattr(image, "dtype", None)
        return (
            str(Path(model_dir).expanduser().resolve()),
            image_layer_name,
            id(image),
            shape,
            str(dtype) if dtype is not None else "",
            device,
            float(threshold),
            bool(compile_model),
        )

    def start_or_reuse_video_session(
        self,
        *,
        adapter: Any,
        image: Any,
        bundle: Any,
        session_key: tuple[Any, ...],
        windows_compatibility_mode: bool = False,
    ) -> Any:
        if (
            self.video_session is not None
            and self.video_session_key == session_key
            and self._adapter_has_session(adapter, self.video_session)
        ):
            return self.video_session

        if getattr(adapter, "video_predictor", None) is None:
            self._log_cuda_diagnostics("before adapter.load_video()")
            load_t0 = time.perf_counter()
            adapter.load_video()
            self.video_predictor_thread_id = threading.get_ident()
            self._log_timing("SAM3.1 model load", load_t0)
            self._log_cuda_diagnostics("after adapter.load_video()")
        else:
            self.log("SAM3.1 video predictor already loaded; reusing cached predictor.")
        config_t0 = time.perf_counter()
        self.configure_multiplex_predictor(
            adapter,
            windows_compatibility_mode=windows_compatibility_mode,
        )
        self._log_timing("SAM3.1 predictor configuration", config_t0)
        adapter._install_video_backend_compatibility()
        self.log(f"SAM3.1 image source: {self.describe_image_source(image)}")
        session_t0 = time.perf_counter()
        video_dir = self._ensure_frame_directory(adapter, image, bundle, session_key)
        with adapter._inference_context():
            response = adapter.video_predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_dir),
                }
            )
        self._log_timing("SAM3.1 session start", session_t0)

        from napari_sam3_assistant.core.models import Sam3Session

        session = Sam3Session(
            task=bundle.task,
            image=bundle.image,
            session_id=response["session_id"],
            resource_path=video_dir,
        )
        adapter.video_session = session
        self.video_session = session
        self.video_session_key = session_key
        return session

    def start_video_session_from_memory(
        self,
        *,
        adapter: Any,
        image: Any,
        bundle: Any,
    ) -> Any:
        if getattr(adapter, "video_predictor", None) is None:
            self._log_cuda_diagnostics("before adapter.load_video()")
            load_t0 = time.perf_counter()
            adapter.load_video()
            self.video_predictor_thread_id = threading.get_ident()
            self._log_timing("SAM3.1 model load", load_t0)
            self._log_cuda_diagnostics("after adapter.load_video()")
        else:
            self.log("SAM3.1 video predictor already loaded; reusing cached predictor.")

        adapter._install_video_backend_compatibility()
        frame_t0 = time.perf_counter()
        frames = self._stack_as_pil_frames(image, bundle)
        self.log(
            "SAM3.1 memory frame source prepared: "
            f"frames={len(frames)}, first_size={frames[0].size if frames else None}, "
            f"time={time.perf_counter() - frame_t0:.2f} sec."
        )
        with adapter._inference_context():
            response = adapter.video_predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": frames,
                }
            )

        from napari_sam3_assistant.core.models import Sam3Session

        session = Sam3Session(
            task=bundle.task,
            image=bundle.image,
            session_id=response["session_id"],
            resource_path=Path("<memory>"),
        )
        adapter.video_session = session
        self.video_session = session
        self.video_session_key = None
        return session

    @staticmethod
    def _stack_as_pil_frames(image: Any, bundle: Any) -> list[Image.Image]:
        from napari_sam3_assistant.core.coordinates import (
            extract_video_frame_image,
            to_rgb_uint8,
        )

        frame_count = 1 if bundle.image.frame_axis is None else int(
            bundle.image.data_shape[bundle.image.frame_axis]
        )
        frames: list[Image.Image] = []
        for idx in range(frame_count):
            frame_2d = extract_video_frame_image(image, bundle.image, idx)
            rgb = to_rgb_uint8(np.asarray(frame_2d))
            frames.append(Image.fromarray(rgb))
        return frames

    @staticmethod
    def describe_image_source(image: Any) -> str:
        image_type = type(image)
        module = getattr(image_type, "__module__", "")
        qualname = getattr(image_type, "__qualname__", image_type.__name__)
        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", None)
        chunks = getattr(image, "chunks", None)
        flags = []
        if isinstance(image, np.ndarray):
            flags.append("numpy")
            if isinstance(image, np.memmap):
                flags.append("memmap")
        if module.startswith("dask"):
            flags.append("dask")
        if module.startswith("zarr"):
            flags.append("zarr")
        if not flags:
            flags.append("lazy_or_custom" if hasattr(image, "__getitem__") and shape is not None else "unknown")
        return (
            f"type={module}.{qualname}, shape={shape}, dtype={dtype}, "
            f"chunks={chunks}, source={','.join(flags)}"
        )

    def _ensure_frame_directory(
        self,
        adapter: Any,
        image: Any,
        bundle: Any,
        frame_dir_key: tuple[Any, ...],
    ) -> Path:
        if (
            self.frame_dir is not None
            and self.frame_dir_key == frame_dir_key
            and Path(self.frame_dir.name).exists()
        ):
            return Path(self.frame_dir.name)
        self._clear_frame_dir()
        self.frame_dir = TemporaryDirectory(prefix="napari-training-sam3-video-")
        self.frame_dir_key = frame_dir_key
        video_dir = Path(self.frame_dir.name)
        adapter._write_stack_as_jpeg_dir(image, bundle, video_dir)
        return video_dir

    def _clear_frame_dir(self) -> None:
        if self.frame_dir is not None:
            try:
                self.frame_dir.cleanup()
            except Exception:
                pass
        self.frame_dir = None
        self.frame_dir_key = None

    @staticmethod
    def _adapter_has_session(adapter: Any, session: Any) -> bool:
        has_video_session = getattr(adapter, "has_video_session", None)
        if callable(has_video_session):
            return bool(has_video_session(session))
        return bool(getattr(session, "session_id", None))

    def _multiplex_bundle_from_prompt_data(self, *, selection: Any, prompt_data: Any) -> Any:
        from napari_sam3_assistant.core.models import (
            BoxPrompt,
            PointPrompt,
            PromptBundle,
            PromptPolarity,
            Sam3Task,
        )

        bundle = PromptBundle(task=Sam3Task.SEGMENT_3D, image=selection)
        if isinstance(prompt_data, dict):
            point_prompt = convert_point_prompt(prompt_data)
            for (y, x), label in zip(point_prompt.points_yx, point_prompt.labels, strict=False):
                bundle.points.append(
                    PointPrompt(
                        y=float(y),
                        x=float(x),
                        polarity=PromptPolarity.POSITIVE if int(label) == 1 else PromptPolarity.NEGATIVE,
                        frame_index=selection.frame_index,
                        object_id=1,
                    )
                )
        else:
            box_prompt = convert_box_prompt(prompt_data)
            for object_id, (y0, x0, y1, x1) in enumerate(box_prompt.boxes_yxyx, start=1):
                bundle.boxes.append(
                    BoxPrompt(
                        y0=float(y0),
                        x0=float(x0),
                        y1=float(y1),
                        x1=float(x1),
                        polarity=PromptPolarity.POSITIVE,
                        frame_index=selection.frame_index,
                        object_id=object_id,
                    )
                )
        return bundle

    def _image_bundle_from_prompt_data(
        self,
        *,
        task: Any,
        selection: Any,
        mode: str,
        prompt_data: Any,
    ) -> Any:
        from napari_sam3_assistant.core.models import (
            BoxPrompt,
            PointPrompt,
            PromptBundle,
            PromptPolarity,
        )

        bundle = PromptBundle(task=task, image=selection)
        if mode in {"2d_box", "2d_exemplar"}:
            box_prompt = convert_box_prompt(prompt_data)
            for y0, x0, y1, x1 in box_prompt.boxes_yxyx:
                bundle.boxes.append(
                    BoxPrompt(
                        y0=float(y0),
                        x0=float(x0),
                        y1=float(y1),
                        x1=float(x1),
                        polarity=PromptPolarity.POSITIVE,
                        frame_index=selection.frame_index,
                    )
                )
        elif mode in {"2d_points", "live_points"}:
            point_prompt = convert_point_prompt(prompt_data)
            for (y, x), label in zip(point_prompt.points_yx, point_prompt.labels, strict=False):
                bundle.points.append(
                    PointPrompt(
                        y=float(y),
                        x=float(x),
                        polarity=PromptPolarity.POSITIVE if int(label) == 1 else PromptPolarity.NEGATIVE,
                        frame_index=selection.frame_index,
                    )
                )
        return bundle

    @staticmethod
    def _mark_video_result(result: Any, *, stage: str) -> None:
        result.metadata["backend_status"] = "napari_sam3_assistant_sam31_multiplex"
        result.metadata["result_kind"] = "video_frame"
        result.metadata["stage"] = stage

    @staticmethod
    def _data_shape(data: Any) -> tuple[int, ...]:
        shape = getattr(data, "shape", None)
        if shape is None:
            shape = np.asarray(data).shape
        return tuple(int(value) for value in shape)
