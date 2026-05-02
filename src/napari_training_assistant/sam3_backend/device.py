from __future__ import annotations

from .errors import SAM3ModelError


def resolve_device(requested: str) -> str:
    """Return 'cuda' or 'cpu' based on user request and torch availability."""

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except Exception as exc:
        raise SAM3ModelError("PyTorch is required for SAM3 inference.") from exc

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SAM3ModelError("CUDA was selected, but PyTorch cannot access a CUDA GPU.")
        return "cuda"

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    raise SAM3ModelError(f"Unknown device option: {requested}")