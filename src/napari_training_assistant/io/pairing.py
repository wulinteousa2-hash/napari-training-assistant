from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tifffile


IMAGE_EXTS = {".tif", ".tiff"}
PairingMode = Literal["auto", "two_folders", "one_folder", "csv"]
FileRole = Literal["image", "mask", "unknown"]

# Keep the original suffixes, but expand the vocabulary for practical datasets.
MASK_SUFFIXES = (
    "_mask", "_m", "_masks", "_label", "_labels", "_lbl",
    "_seg", "_segmentation", "_annotation", "_annotations",
    "_gt", "_groundtruth", "_ground_truth", "_binary", "_target",
)
IMAGE_SUFFIXES = (
    "_image", "_img", "_raw", "_input", "_source", "_ch0", "_channel0",
)

_MASK_WORDS = {
    "mask", "masks", "m", "label", "labels", "lbl", "seg", "segmentation",
    "annotation", "annotations", "gt", "groundtruth", "ground_truth", "binary", "target", "truth",
}
_IMAGE_WORDS = {"image", "img", "raw", "input", "source", "ch0", "channel0"}
_SEPARATOR_RE = re.compile(r"[\s._\-]+")
_TRAILING_ROLE_RE = re.compile(
    r"(?i)(?:[\s._\-]+)(?:"
    r"mask|masks|m|label|labels|lbl|seg|segmentation|annotation|annotations|"
    r"gt|groundtruth|ground_truth|binary|target|truth|"
    r"image|img|raw|input|source|ch0|channel0|"
    r"1|2"
    r")$"
)


@dataclass
class PairRecord:
    key: str
    image_path: str
    mask_path: str
    confidence: int = 100
    reason: str = "exact filename key match"
    status: str = "ok"
    image_shape: str = ""
    mask_shape: str = ""
    shape_ok: bool | None = None


@dataclass
class PairingReport:
    pairs: list[PairRecord]
    unmatched_images: list[str]
    unmatched_masks: list[str]
    ambiguous: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    mode_used: str = ""

    @property
    def valid_pairs(self) -> list[PairRecord]:
        return [p for p in self.pairs if p.status == "ok"]


@dataclass(frozen=True)
class FileCandidate:
    path: Path
    role: FileRole
    key: str
    confidence: int
    reason: str
    source: str = ""


def _is_tiff(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def _iter_tiff_files(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and _is_tiff(p))


def _tokens(stem: str) -> list[str]:
    return [t for t in _SEPARATOR_RE.split(stem.lower()) if t]


def _strip_known_role_suffix(stem: str) -> str:
    """Remove one trailing role token, including _1/_2, from a stem."""
    out = _TRAILING_ROLE_RE.sub("", stem).strip(" ._-")
    return out or stem


def _normalize_key(stem: str) -> str:
    base = _strip_known_role_suffix(stem)
    base = base.strip().lower()
    base = re.sub(r"[\s._\-]+", "_", base)
    return base.strip("_") or stem.lower()


def classify_file_role(path: str | Path) -> tuple[FileRole, int, str]:
    """Classify a file as image/mask/unknown from its name only."""
    p = Path(path)
    stem = p.stem
    toks = _tokens(stem)
    last = toks[-1] if toks else ""

    # Strong end-token rules are safest.
    if last in _MASK_WORDS:
        return "mask", 95, f"mask token '{last}'"
    if last in _IMAGE_WORDS:
        return "image", 95, f"image token '{last}'"

    # Numeric pair convention requested by user: sample_1=image, sample_2=mask.
    if last == "1":
        return "image", 85, "numeric _1 image rule"
    if last == "2":
        return "mask", 85, "numeric _2 mask rule"

    # Weaker contains-token rules for names like sample-final-mask-v2 are intentionally omitted
    # to avoid false positives with acquisition metadata.
    return "unknown", 40, "no role token detected"


def _candidate_from_path(path: Path, forced_role: FileRole | None = None, source: str = "") -> FileCandidate:
    detected_role, conf, reason = classify_file_role(path)
    role = forced_role or detected_role

    if forced_role is not None:
        # In two-folder mode, the folder gives the role. The name only helps key normalization.
        if detected_role != "unknown" and detected_role != forced_role:
            reason = f"folder says {forced_role}; filename suggested {detected_role}"
            conf = 75
        elif detected_role == forced_role:
            reason = f"folder + filename agree: {forced_role}"
            conf = max(conf, 98)
        else:
            reason = f"folder role: {forced_role}"
            conf = 90

    return FileCandidate(
        path=path,
        role=role,
        key=_normalize_key(path.stem),
        confidence=conf,
        reason=reason,
        source=source,
    )


def _shape_text(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return ""
    return "x".join(str(int(x)) for x in shape)


def _tiff_shape(path: Path) -> tuple[int, ...] | None:
    try:
        with tifffile.TiffFile(str(path)) as tif:
            if not tif.series:
                return None
            return tuple(int(x) for x in tif.series[0].shape)
    except Exception:
        return None


def _spatial_shape_for_image(shape: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    # RGB/RGBA 2D image: Y, X, C
    if len(shape) == 3 and shape[-1] in {3, 4}:
        return shape[:-1]
    # RGB/RGBA 3D stack: Z, Y, X, C
    if len(shape) == 4 and shape[-1] in {3, 4}:
        return shape[:-1]
    return shape


def _validate_pair_shape(image_path: Path, mask_path: Path) -> tuple[bool | None, str, str, str]:
    img_shape = _tiff_shape(image_path)
    msk_shape = _tiff_shape(mask_path)
    img_txt = _shape_text(img_shape)
    msk_txt = _shape_text(msk_shape)

    if img_shape is None or msk_shape is None:
        return None, img_txt, msk_txt, "shape unchecked"

    ok = _spatial_shape_for_image(img_shape) == msk_shape
    return ok, img_txt, msk_txt, "shape ok" if ok else "shape mismatch"


def _score_pair(img: FileCandidate, msk: FileCandidate) -> tuple[int, str]:
    score = min(img.confidence, msk.confidence)
    reasons = [img.reason, msk.reason]

    if img.key == msk.key:
        score += 10
        reasons.append("same normalized key")
    else:
        score -= 50
        reasons.append("different normalized key")

    if "numeric _1" in img.reason and "numeric _2" in msk.reason:
        score += 5
        reasons.append("_1/_2 pair")

    return max(0, min(100, score)), "; ".join(reasons)


def _build_pairs(
    image_candidates: list[FileCandidate],
    mask_candidates: list[FileCandidate],
    *,
    validate_shapes: bool = True,
    min_confidence: int = 50,
    mode_used: str = "",
) -> PairingReport:
    by_img_key: dict[str, list[FileCandidate]] = {}
    by_msk_key: dict[str, list[FileCandidate]] = {}
    for c in image_candidates:
        by_img_key.setdefault(c.key, []).append(c)
    for c in mask_candidates:
        by_msk_key.setdefault(c.key, []).append(c)

    pairs: list[PairRecord] = []
    ambiguous: list[str] = []
    rejected: list[str] = []
    used_images: set[Path] = set()
    used_masks: set[Path] = set()

    for key in sorted(set(by_img_key) & set(by_msk_key)):
        candidates: list[tuple[int, str, FileCandidate, FileCandidate]] = []
        for img in by_img_key[key]:
            for msk in by_msk_key[key]:
                score, reason = _score_pair(img, msk)
                candidates.append((score, reason, img, msk))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            continue

        best_score, reason, img, msk = candidates[0]
        if best_score < min_confidence:
            rejected.append(f"{key}: low confidence ({best_score})")
            continue

        if len(candidates) > 1 and candidates[1][0] >= best_score - 5:
            names = [f"{c[2].path.name} -> {c[3].path.name} ({c[0]})" for c in candidates[:5]]
            ambiguous.append(f"{key}: " + "; ".join(names))
            # Keep the best candidate but mark it as warning, so user can inspect.
            status = "ambiguous"
        else:
            status = "ok"

        shape_ok: bool | None = None
        image_shape = ""
        mask_shape = ""
        if validate_shapes:
            shape_ok, image_shape, mask_shape, shape_reason = _validate_pair_shape(img.path, msk.path)
            reason = f"{reason}; {shape_reason}"
            if shape_ok is False:
                status = "shape_mismatch"
                rejected.append(f"{key}: shape mismatch | {img.path.name} {image_shape} vs {msk.path.name} {mask_shape}")

        pairs.append(PairRecord(
            key=key,
            image_path=str(img.path),
            mask_path=str(msk.path),
            confidence=best_score,
            reason=reason,
            status=status,
            image_shape=image_shape,
            mask_shape=mask_shape,
            shape_ok=shape_ok,
        ))
        used_images.add(img.path)
        used_masks.add(msk.path)

    unmatched_images = [str(c.path) for c in image_candidates if c.path not in used_images]
    unmatched_masks = [str(c.path) for c in mask_candidates if c.path not in used_masks]

    return PairingReport(
        pairs=pairs,
        unmatched_images=unmatched_images,
        unmatched_masks=unmatched_masks,
        ambiguous=ambiguous,
        rejected=rejected,
        mode_used=mode_used,
    )


def pair_image_mask_folders(
    image_dir: str | Path,
    mask_dir: str | Path,
    *,
    validate_shapes: bool = True,
    recursive: bool = False,
) -> PairingReport:
    """Pair separate image and mask folders.

    Folder identity defines role, while filename normalization handles common naming variants:
    sample.tif + sample_mask.tif, sample_1.tif + sample_2.tif, sample_raw.tif + sample_label.tif.
    """
    image_files = _iter_tiff_files(image_dir, recursive=recursive)
    mask_files = _iter_tiff_files(mask_dir, recursive=recursive)

    images = [_candidate_from_path(p, forced_role="image", source="image_dir") for p in image_files]
    masks = [_candidate_from_path(p, forced_role="mask", source="mask_dir") for p in mask_files]

    return _build_pairs(
        images,
        masks,
        validate_shapes=validate_shapes,
        mode_used="two_folders",
    )


def pair_mixed_folder(
    dataset_dir: str | Path,
    *,
    validate_shapes: bool = True,
    recursive: bool = False,
) -> PairingReport:
    """Pair a single folder containing both images and masks."""
    files = _iter_tiff_files(dataset_dir, recursive=recursive)
    candidates = [_candidate_from_path(p, forced_role=None, source="dataset_dir") for p in files]

    images = [c for c in candidates if c.role == "image"]
    masks = [c for c in candidates if c.role == "mask"]
    unknown = [c for c in candidates if c.role == "unknown"]

    # Safe fallback: if a key has exactly one mask and one unknown file, treat the unknown as image.
    masks_by_key: dict[str, list[FileCandidate]] = {}
    unknown_by_key: dict[str, list[FileCandidate]] = {}
    for c in masks:
        masks_by_key.setdefault(c.key, []).append(c)
    for c in unknown:
        unknown_by_key.setdefault(c.key, []).append(c)

    promoted_unknown_paths: set[Path] = set()
    for key, unk_list in unknown_by_key.items():
        if len(unk_list) == 1 and len(masks_by_key.get(key, [])) == 1:
            u = unk_list[0]
            images.append(FileCandidate(
                path=u.path,
                role="image",
                key=u.key,
                confidence=70,
                reason="unknown file promoted to image because matching mask exists",
                source=u.source,
            ))
            promoted_unknown_paths.add(u.path)

    report = _build_pairs(
        images,
        masks,
        validate_shapes=validate_shapes,
        mode_used="one_folder",
    )

    # Preserve truly unknown files in ambiguous list so user sees they were not silently used.
    for c in unknown:
        if c.path not in promoted_unknown_paths:
            report.ambiguous.append(f"unclassified file: {c.path}")

    return report


def pair_auto(
    *,
    dataset_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    validate_shapes: bool = True,
    recursive: bool = False,
) -> PairingReport:
    """Auto mode: prefer one-folder dataset if provided, otherwise use two folders."""
    dataset_dir = Path(dataset_dir) if dataset_dir else None
    image_dir = Path(image_dir) if image_dir else None
    mask_dir = Path(mask_dir) if mask_dir else None

    if dataset_dir and str(dataset_dir).strip() and dataset_dir.is_dir():
        return pair_mixed_folder(dataset_dir, validate_shapes=validate_shapes, recursive=recursive)

    if image_dir and mask_dir:
        return pair_image_mask_folders(image_dir, mask_dir, validate_shapes=validate_shapes, recursive=recursive)

    raise ValueError("Auto pairing requires either a dataset folder or both image and mask folders.")


def pair_from_csv(csv_path: str | Path, *, validate_shapes: bool = True) -> PairingReport:
    """Load explicit pairs from a CSV with columns: key,image_path,mask_path."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"CSV file not found: {csv_path}")

    pairs: list[PairRecord] = []
    rejected: list[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "mask_path"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Pair CSV must contain at least image_path and mask_path columns.")

        for i, row in enumerate(reader, start=2):
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            key = row.get("key") or _normalize_key(image_path.stem)
            status = "ok"
            reason = "manual CSV pair"
            shape_ok: bool | None = None
            image_shape = ""
            mask_shape = ""

            if not image_path.is_file() or not mask_path.is_file():
                status = "missing_file"
                rejected.append(f"CSV row {i}: missing image or mask file")
            elif validate_shapes:
                shape_ok, image_shape, mask_shape, shape_reason = _validate_pair_shape(image_path, mask_path)
                reason = f"{reason}; {shape_reason}"
                if shape_ok is False:
                    status = "shape_mismatch"
                    rejected.append(f"CSV row {i}: shape mismatch")

            pairs.append(PairRecord(
                key=key,
                image_path=str(image_path),
                mask_path=str(mask_path),
                confidence=100,
                reason=reason,
                status=status,
                image_shape=image_shape,
                mask_shape=mask_shape,
                shape_ok=shape_ok,
            ))

    return PairingReport(
        pairs=pairs,
        unmatched_images=[],
        unmatched_masks=[],
        ambiguous=[],
        rejected=rejected,
        mode_used="csv",
    )
