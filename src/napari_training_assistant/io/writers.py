from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import tifffile


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv_rows(path: str | Path, header: list[str], rows: list[list[Any]]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_tiff(path: str | Path, array) -> None:
    path = Path(path)
    tifffile.imwrite(str(path), array)