from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np
from PIL import Image

from .device import resolve_device
from .errors import SAM3InferenceError
from .model_loader import SAM3ModelLoader
from .prompt_converter import BoxPrompt, PointPrompt
from .reference_backend import ReferenceSAM3Backend


@dataclass
class SAM3PreviewResult:
    labels: np.ndarray
    metadata: dict[str, Any]


class SAM3PreviewEngine:
    """High-level SAM3 preview engine used by the napari widget."""

    def __init__(self) -> None:
        self.loader = SAM3ModelLoader()
        self.reference_backend = ReferenceSAM3Backend(self.loader)

    def run_preview(
        self,
        *,
        image: np.ndarray,
        mode: str,
        model_dir: str,
        device: str,
        prompt_data: Any,
        image_layer_name: str = "image",
        dims_current_step: tuple[int, ...] | None = None,
        threshold: float = 0.5,
        compile_model: bool = False,
        propagation_direction: str = "both",
        windows_compatibility_mode: bool = False,
        progress_callback: Any | None = None,
    ) -> SAM3PreviewResult:
        self._emit(progress_callback, 5, "Resolving SAM3 device.")
        selected_device = resolve_device(device)
        if mode == "3d_multiplex":
            final_result = None
            for result in self.iter_multiplex_preview(
                image=image,
                image_layer_name=image_layer_name,
                dims_current_step=dims_current_step,
                model_dir=model_dir,
                device=selected_device,
                prompt_data=prompt_data,
                threshold=threshold,
                compile_model=compile_model,
                direction=propagation_direction,
                windows_compatibility_mode=windows_compatibility_mode,
                progress_callback=progress_callback,
            ):
                final_result = result
            if final_result is None:
                raise SAM3InferenceError("SAM3.1 multiplex returned no results.")
            return final_result
        try:
            return self._run_reference_adapter_preview(
                image=image,
                image_layer_name=image_layer_name,
                dims_current_step=dims_current_step,
                mode=mode,
                model_dir=model_dir,
                device=selected_device,
                prompt_data=prompt_data,
                threshold=threshold,
                compile_model=compile_model,
                progress_callback=progress_callback,
            )
        except ImportError as exc:
            raise SAM3InferenceError(
                "SAM3 preview requires napari-sam3-assistant so image slicing, "
                "normalization, live points, and exemplar prompts match the "
                "reference plugin. Install or expose napari-sam3-assistant in "
                "this environment before running SAM3 preview."
            ) from exc

    def preload_multiplex_model(
        self,
        *,
        model_dir: str,
        device: str,
        threshold: float = 0.5,
        compile_model: bool = False,
        windows_compatibility_mode: bool = False,
    ) -> None:
        self.reference_backend.preload_multiplex_model(
            model_dir=model_dir,
            device=resolve_device(device),
            threshold=threshold,
            compile_model=compile_model,
            windows_compatibility_mode=windows_compatibility_mode,
        )

    def _ensure_multiplex_adapter(
        self,
        *,
        model_dir: str,
        device: str,
        threshold: float,
        compile_model: bool,
        windows_compatibility_mode: bool = False,
    ) -> Any:
        return self.reference_backend._ensure_adapter(
            model_dir=model_dir,
            model_type="sam3.1",
            device=device,
            threshold=threshold,
            compile_model=compile_model,
        )

    def iter_multiplex_preview(
        self,
        *,
        image: Any,
        image_layer_name: str,
        dims_current_step: tuple[int, ...] | None,
        model_dir: str,
        device: str,
        prompt_data: Any,
        prompt_bundle: Any | None = None,
        threshold: float,
        compile_model: bool,
        direction: str = "both",
        windows_compatibility_mode: bool = False,
        progress_callback: Any | None = None,
    ) -> Iterator[SAM3PreviewResult]:
        selected_device = resolve_device(device)
        yield from self._iter_reference_multiplex_preview(
            image=image,
            image_layer_name=image_layer_name,
            dims_current_step=dims_current_step,
            model_dir=model_dir,
            device=selected_device,
            prompt_data=prompt_data,
            prompt_bundle=prompt_bundle,
            threshold=threshold,
            compile_model=compile_model,
            direction=direction,
            windows_compatibility_mode=windows_compatibility_mode,
            progress_callback=progress_callback,
        )

    def _iter_reference_multiplex_preview(
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
        compile_model: bool,
        direction: str,
        windows_compatibility_mode: bool,
        progress_callback: Any | None,
    ) -> Iterator[SAM3PreviewResult]:
        self._emit(progress_callback, 25, "Loading SAM3.1 multiplex video predictor.")
        try:
            yield from self.reference_backend.iter_multiplex(
                image=image,
                image_layer_name=image_layer_name,
                dims_current_step=dims_current_step,
                model_dir=model_dir,
                device=device,
                prompt_data=prompt_data,
                prompt_bundle=prompt_bundle,
                threshold=threshold,
                compile_model=compile_model,
                direction=direction,
                windows_compatibility_mode=windows_compatibility_mode,
            )
        except ImportError as exc:
            raise SAM3InferenceError(
                "SAM3.1 multiplex preview requires napari-sam3-assistant. "
                "Install or expose napari-sam3-assistant in this environment."
            ) from exc
        self._emit(progress_callback, 95, "SAM3.1 multiplex propagation complete.")

    @staticmethod
    def _data_shape(data: Any) -> tuple[int, ...]:
        shape = getattr(data, "shape", None)
        if shape is None:
            shape = np.asarray(data).shape
        return tuple(int(value) for value in shape)

    def _run_reference_adapter_preview(
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
        progress_callback: Any | None,
    ) -> SAM3PreviewResult:
        self._emit(progress_callback, 20, "Preparing prompts with napari-sam3-assistant adapter.")
        self._emit(progress_callback, 35, "Running napari-sam3-assistant SAM3 adapter.")
        try:
            result = self.reference_backend.run_image(
                image=image,
                image_layer_name=image_layer_name,
                dims_current_step=dims_current_step,
                mode=mode,
                model_dir=model_dir,
                device=device,
                prompt_data=prompt_data,
                threshold=threshold,
                compile_model=compile_model,
            )
        except ImportError as exc:
            raise SAM3InferenceError(
                "SAM3 preview requires napari-sam3-assistant so image slicing, "
                "normalization, live points, and exemplar prompts match the "
                "reference plugin. Install or expose napari-sam3-assistant in "
                "this environment before running SAM3 preview."
            ) from exc
        except Exception as exc:
            raise SAM3InferenceError(f"SAM3 adapter inference failed: {exc}") from exc

        labels = result.labels
        if labels is None and result.masks is not None:
            labels = self._labels_from_masks(result.masks)
        if labels is None:
            raise SAM3InferenceError("SAM3 returned no preview labels.")
        self._emit(progress_callback, 95, "Writing SAM3 preview labels.")
        scores = [] if result.scores is None else [float(v) for v in np.asarray(result.scores).reshape(-1)]
        return SAM3PreviewResult(
            labels=np.asarray(labels),
            metadata={
                "mode": mode,
                "device": device,
                "threshold": threshold,
                "backend_status": "napari_sam3_assistant_adapter",
                "scores": scores,
                "task": mode,
            },
        )

    @staticmethod
    def _prepare_2d_image(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)

        if array.ndim < 2:
            raise SAM3InferenceError(f"Expected at least 2D image data, got shape {array.shape}.")

        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            pass
        elif array.ndim > 2:
            array = array.reshape((-1, *array.shape[-2:]))[0]

        if array.dtype == np.uint8:
            return array

        array = array.astype(np.float32, copy=False)
        low, high = np.percentile(array, [1, 99])

        if high <= low:
            return np.zeros(array.shape, dtype=np.uint8)

        array = np.clip((array - low) / (high - low), 0, 1)
        return (array * 255).astype(np.uint8)

    @staticmethod
    def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            return np.stack([array, array, array], axis=-1)
        if array.ndim == 3 and array.shape[-1] == 1:
            return np.repeat(array, 3, axis=-1)
        if array.ndim == 3 and array.shape[-1] == 3:
            return array
        if array.ndim == 3 and array.shape[-1] == 4:
            return array[..., :3]
        raise SAM3InferenceError(f"Expected grayscale, RGB, or RGBA image data, got {array.shape}.")

    @staticmethod
    def _insert_video_result_labels(labels: np.ndarray, result: Any) -> None:
        frame_index = getattr(result, "frame_index", None)
        if frame_index is None:
            return
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= labels.shape[0]:
            return
        frame_labels = getattr(result, "labels", None)
        if frame_labels is None:
            frame_labels = SAM3PreviewEngine._labels_from_masks(
                getattr(result, "masks", None),
                object_ids=getattr(result, "object_ids", None),
            )
        if frame_labels is None:
            return
        incoming = np.asarray(frame_labels, dtype=labels.dtype)
        if labels[frame_index].any() and not incoming.any():
            return
        labels[frame_index] = incoming

    @staticmethod
    def _preview_result_from_video_frame(
        result: Any,
        *,
        mode: str,
        device: str,
        threshold: float,
        stage: str,
    ) -> SAM3PreviewResult:
        labels = getattr(result, "labels", None)
        if labels is None:
            labels = SAM3PreviewEngine._labels_from_masks(
                getattr(result, "masks", None),
                object_ids=getattr(result, "object_ids", None),
            )
        if labels is None:
            labels = np.zeros((0, 0), dtype=np.uint32)
        object_ids = getattr(result, "object_ids", None)
        return SAM3PreviewResult(
            labels=np.asarray(labels),
            metadata={
                "mode": mode,
                "device": device,
                "threshold": threshold,
                "backend_status": "napari_sam3_assistant_sam31_multiplex",
                "result_kind": "video_frame",
                "stage": stage,
                "frame_index": getattr(result, "frame_index", None),
                "session_id": getattr(result, "session_id", None),
                "object_ids": (
                    []
                    if object_ids is None
                    else [int(value) for value in np.asarray(object_ids).reshape(-1)]
                ),
            },
        )

    @staticmethod
    def _run_box_prompts(
        model: Any,
        state: dict[str, Any],
        prompt: BoxPrompt,
    ) -> tuple[np.ndarray, np.ndarray]:
        masks_by_box = []
        scores_by_box = []
        for box_xyxy in prompt.boxes_xyxy:
            masks, scores, _low_res = model.predict_inst(
                state,
                point_coords=None,
                point_labels=None,
                box=np.asarray([box_xyxy], dtype=np.float32),
                mask_input=None,
                multimask_output=False,
                normalize_coords=True,
            )
            mask = SAM3PreviewEngine._first_mask(masks)
            if mask is not None:
                masks_by_box.append(mask)
                score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
                scores_by_box.append(float(score_array[0]) if score_array.size else 0.0)

        if not masks_by_box:
            shape = tuple(int(v) for v in state.get("image_size", (0, 0)))
            if len(shape) != 2 or min(shape) <= 0:
                raise SAM3InferenceError("SAM3 returned no masks for the box prompt.")
            return np.zeros((0, *shape), dtype=bool), np.zeros((0,), dtype=np.float32)
        return np.stack(masks_by_box, axis=0), np.asarray(scores_by_box, dtype=np.float32)

    @staticmethod
    def _run_video_exemplar_prompts(
        video_predictor: Any,
        rgb: np.ndarray,
        prompt: BoxPrompt,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_hw = rgb.shape[:2]
        masks_by_prompt = []
        scores_by_prompt = []

        with TemporaryDirectory(prefix="napari-training-sam3-exemplar-") as tmpdir:
            frame_path = f"{tmpdir}/000000.jpg"
            Image.fromarray(rgb).save(frame_path, quality=95)

            start = video_predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": tmpdir,
                }
            )
            session_id = start["session_id"]
            for box_yxyx in prompt.boxes_yxyx:
                response = video_predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": "visual",
                        "bounding_boxes": [
                            list(
                                SAM3PreviewEngine._box_yxyx_to_normalized_xywh(
                                    box_yxyx,
                                    image_hw,
                                )
                            )
                        ],
                        "bounding_box_labels": [1],
                        "output_prob_thresh": float(threshold),
                    }
                )
                outputs = response.get("outputs", {})
                masks = SAM3PreviewEngine._to_numpy(outputs.get("out_binary_masks"))
                scores = SAM3PreviewEngine._to_numpy(outputs.get("out_probs"))
                if masks is not None and masks.size:
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    elif masks.ndim == 2:
                        masks = masks[None]
                    masks_by_prompt.append(masks.astype(bool))
                    if scores is not None and scores.size:
                        scores_by_prompt.extend(np.asarray(scores, dtype=np.float32).reshape(-1))
                    else:
                        scores_by_prompt.extend([0.0] * int(masks.shape[0]))

            try:
                video_predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
            except Exception:
                pass

        if not masks_by_prompt:
            raise SAM3InferenceError("SAM3 returned no masks for the visual exemplar prompt.")
        return (
            np.concatenate(masks_by_prompt, axis=0),
            np.asarray(scores_by_prompt, dtype=np.float32),
        )

    @staticmethod
    def _box_yxyx_to_normalized_xywh(
        box_yxyx: np.ndarray,
        image_hw: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        height, width = image_hw
        y0, x0, y1, x1 = [float(value) for value in box_yxyx]
        y0, y1 = sorted((np.clip(y0, 0, height), np.clip(y1, 0, height)))
        x0, x1 = sorted((np.clip(x0, 0, width), np.clip(x1, 0, width)))
        return (
            float(x0 / max(width, 1)),
            float(y0 / max(height, 1)),
            float(max(0.0, x1 - x0) / max(width, 1)),
            float(max(0.0, y1 - y0) / max(height, 1)),
        )

    @staticmethod
    def _run_exemplar_prompts(
        model: Any,
        processor: Any,
        state: dict[str, Any],
        prompt: BoxPrompt,
        image_hw: tuple[int, int],
        device: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        for box_yxyx in prompt.boxes_yxyx:
            normalized_box = SAM3PreviewEngine._box_yxyx_to_normalized_cxcywh(
                box_yxyx,
                image_hw,
            )
            state = SAM3PreviewEngine._add_geometric_prompt(
                model,
                processor,
                state,
                list(normalized_box),
                True,
                device=device,
            )
            SAM3PreviewEngine._normalize_state_tensors(state, device)

        masks = SAM3PreviewEngine._to_numpy(state.get("masks"))
        scores = SAM3PreviewEngine._to_numpy(state.get("scores"))
        if masks is None or masks.size == 0:
            raise SAM3InferenceError("SAM3 returned no masks for the exemplar prompt.")
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if scores is None:
            scores = np.zeros((masks.shape[0],), dtype=np.float32)
        return masks.astype(bool), np.asarray(scores, dtype=np.float32).reshape(-1)

    @staticmethod
    def _box_yxyx_to_normalized_cxcywh(
        box_yxyx: np.ndarray,
        image_hw: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        height, width = image_hw
        y0, x0, y1, x1 = [float(value) for value in box_yxyx]
        y0, y1 = sorted((np.clip(y0, 0, height), np.clip(y1, 0, height)))
        x0, x1 = sorted((np.clip(x0, 0, width), np.clip(x1, 0, width)))
        cx = ((x0 + x1) * 0.5) / max(width, 1)
        cy = ((y0 + y1) * 0.5) / max(height, 1)
        w = max(0.0, x1 - x0) / max(width, 1)
        h = max(0.0, y1 - y0) / max(height, 1)
        return (float(cx), float(cy), float(w), float(h))

    @staticmethod
    def _add_geometric_prompt(
        model: Any,
        processor: Any,
        state: dict[str, Any],
        box: list[float],
        label: bool,
        *,
        device: str,
    ) -> dict[str, Any]:
        if str(device) != "cpu":
            return processor.add_geometric_prompt(box, label, state)

        try:
            import torch
        except Exception as exc:
            raise SAM3InferenceError("PyTorch is required for SAM3 exemplar prompts.") from exc

        if "language_features" not in state["backbone_out"]:
            text_outputs = model.backbone.forward_text(["visual"], device=processor.device)
            state["backbone_out"].update(text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = model._get_dummy_prompt()

        boxes = torch.tensor(
            box,
            device=processor.device,
            dtype=torch.float32,
        ).view(1, 1, 4)
        labels = torch.tensor(
            [label],
            device=processor.device,
            dtype=torch.bool,
        ).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)
        SAM3PreviewEngine._normalize_state_tensors(state, device)
        return processor._forward_grounding(state)

    @staticmethod
    def _run_point_prompts(
        model: Any,
        state: dict[str, Any],
        prompt: PointPrompt,
    ) -> tuple[np.ndarray, np.ndarray]:
        masks, scores, _low_res = model.predict_inst(
            state,
            point_coords=prompt.points_xy.astype(np.float32, copy=False),
            point_labels=prompt.labels.astype(np.int32, copy=False),
            box=None,
            mask_input=None,
            multimask_output=False,
            normalize_coords=True,
        )
        mask = SAM3PreviewEngine._first_mask(masks)
        if mask is None:
            raise SAM3InferenceError("SAM3 returned no masks for the point prompt.")
        return np.asarray([mask], dtype=bool), np.asarray(scores, dtype=np.float32).reshape(-1)

    @staticmethod
    def _first_mask(masks: Any) -> np.ndarray | None:
        mask_array = SAM3PreviewEngine._to_numpy(masks)
        if mask_array is None:
            return None
        if mask_array.size == 0:
            return None
        if mask_array.ndim == 4 and mask_array.shape[1] == 1:
            mask_array = mask_array[:, 0]
        if mask_array.ndim == 3:
            return np.asarray(mask_array[0]).astype(bool)
        if mask_array.ndim == 2:
            return np.asarray(mask_array).astype(bool)
        return None

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
            try:
                import torch

                if value.dtype == torch.bfloat16:
                    value = value.to(dtype=torch.float32)
            except Exception:
                pass
            value = value.cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _normalize_state_tensors(state: dict[str, Any], device: str) -> None:
        try:
            import torch
        except Exception:
            return

        torch_device = torch.device(device)
        dtype = torch.float32 if torch_device.type == "cpu" else None
        SAM3PreviewEngine._convert_tensors(state, device=torch_device, dtype=dtype)

    @staticmethod
    def _convert_tensors(value: Any, *, device: Any, dtype: Any) -> Any:
        try:
            import torch
        except Exception:
            return value

        if isinstance(value, torch.Tensor):
            target_dtype = dtype if value.is_floating_point() and dtype is not None else value.dtype
            if value.device != device or value.dtype != target_dtype:
                return value.to(device=device, dtype=target_dtype)
            return value
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = SAM3PreviewEngine._convert_tensors(item, device=device, dtype=dtype)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = SAM3PreviewEngine._convert_tensors(item, device=device, dtype=dtype)
            return value
        if isinstance(value, tuple):
            return tuple(
                SAM3PreviewEngine._convert_tensors(item, device=device, dtype=dtype)
                for item in value
            )
        if hasattr(value, "__dict__"):
            for key, item in list(vars(value).items()):
                try:
                    setattr(
                        value,
                        key,
                        SAM3PreviewEngine._convert_tensors(item, device=device, dtype=dtype),
                    )
                except Exception:
                    pass
        return value

    @staticmethod
    def _labels_from_masks(
        masks: np.ndarray | None,
        *,
        object_ids: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if masks is None:
            return None
        mask_array = np.asarray(masks)
        if mask_array.ndim == 4 and mask_array.shape[1] == 1:
            mask_array = mask_array[:, 0]
        if mask_array.ndim == 2:
            return mask_array.astype(np.uint16)
        if mask_array.ndim != 3:
            raise SAM3InferenceError(f"Unexpected SAM3 mask shape: {mask_array.shape}")
        labels = np.zeros(mask_array.shape[-2:], dtype=np.uint16)
        ids = None if object_ids is None else np.asarray(object_ids).reshape(-1)
        for label_value, mask in enumerate(mask_array, start=1):
            value = int(ids[label_value - 1]) if ids is not None and label_value - 1 < len(ids) else label_value
            labels[np.asarray(mask).astype(bool)] = value
        return labels

    @staticmethod
    def _emit(progress_callback: Any | None, value: int, message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback.emit((int(value), message))
        except Exception:
            pass

    @staticmethod
    def _inference_context(device: str):
        if str(device).startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            except Exception:
                pass
        return nullcontext()
