"""Provenance verification and recording helpers for manuscript analyses."""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def get_sha256(path: Union[str, Path]) -> str:
    """Return the SHA-256 hex digest of a file."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_mtime_iso(path: Union[str, Path]) -> str:
    """Return the ISO-8601 formatted UTC modification time of a file."""
    p = Path(path)
    mtime = p.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def file_metadata_record(path: Union[str, Path], label: str = "") -> Dict[str, Any]:
    """Return dictionary with path, sha256, mtime, and size in bytes."""
    p = Path(path)
    return {
        "label": label or p.name,
        "path": str(p),
        "sha256": get_sha256(p),
        "mtime": get_mtime_iso(p),
        "bytes": p.stat().st_size,
    }


def write_tidy_table(df: pd.DataFrame, path: Union[str, Path], float_format: str = "%.6f") -> None:
    """Write DataFrame to CSV without index and create parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, float_format=float_format)
