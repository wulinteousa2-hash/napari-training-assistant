from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import SAM3PromptError


@dataclass
class BoxPrompt:
    boxes_xyxy: np.ndarray
    boxes_yxyx: np.ndarray


@dataclass
class PointPrompt:
    points_yx: np.ndarray
    points_xy: np.ndarray
    labels: np.ndarray


def convert_box_prompt(prompt_data: Any) -> BoxPrompt:
    """Convert napari Shapes rectangle data into xyxy boxes."""

    shapes = list(prompt_data)

    if not shapes:
        raise SAM3PromptError("No box prompt found. Draw at least one rectangle.")

    boxes_xyxy = []
    boxes_yxyx = []

    for shape in shapes:
        points = np.asarray(shape)

        if points.ndim != 2 or points.shape[0] < 2:
            continue

        y_min = float(np.min(points[:, -2]))
        x_min = float(np.min(points[:, -1]))
        y_max = float(np.max(points[:, -2]))
        x_max = float(np.max(points[:, -1]))

        boxes_xyxy.append([x_min, y_min, x_max, y_max])
        boxes_yxyx.append([y_min, x_min, y_max, x_max])

    if not boxes_xyxy:
        raise SAM3PromptError("Could not convert napari Shapes data into boxes.")

    return BoxPrompt(
        boxes_xyxy=np.asarray(boxes_xyxy, dtype=np.float32),
        boxes_yxyx=np.asarray(boxes_yxyx, dtype=np.float32),
    )


def convert_point_prompt(prompt_data: Any) -> PointPrompt:
    """Convert napari Points data into yx point coordinates."""

    properties = {}
    if isinstance(prompt_data, dict):
        points = np.asarray(prompt_data.get("data"))
        properties = dict(prompt_data.get("properties") or {})
    else:
        points = np.asarray(prompt_data)

    if points.size == 0:
        raise SAM3PromptError("No point prompt found. Add at least one point.")

    if points.ndim != 2 or points.shape[1] < 2:
        raise SAM3PromptError(f"Invalid point data shape: {points.shape}")

    points_yx = points[:, -2:].astype(np.float32)

    labels = _point_labels(properties, points_yx.shape[0])

    points_xy = points_yx[:, ::-1].copy()

    return PointPrompt(points_yx=points_yx, points_xy=points_xy, labels=labels)


def _point_labels(properties: dict[str, Any], count: int) -> np.ndarray:
    values = properties.get("polarity", [])
    labels = np.ones((count,), dtype=np.int64)
    for index, value in enumerate(list(values)[:count]):
        if str(value).strip().lower() in {"negative", "neg", "background", "bg", "0", "false"}:
            labels[index] = 0
    return labels
