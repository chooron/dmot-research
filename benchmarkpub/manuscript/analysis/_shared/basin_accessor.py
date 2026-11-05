#!/usr/bin/env python3
"""Basin-ID-keyed accessors for manuscript analyses (A6).

All accessors resolve rows strictly by basin ID (8-character zero-padded string),
never by positional order. Two basin orderings exist in this repository:

  * SORTED     — `basin_order()` (numerically sorted `data/531sub_id.txt`);
                 the row order of every performance matrix loaded through `_shared`
                 (stats cubes, `load_oob_cube`, `load_oob_kge_matrix`,
                 `load_canonical_seen_kge_matrix`, attributes).
  * FILE_ORDER — the raw `data/531sub_id.txt` order (`canonical_ids()`), which
                 disagrees with SORTED at 439/531 positions.

Any table or array produced in FILE_ORDER must be realigned through these
accessors before being combined with SORTED-order matrices.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_loaders import basin_order, canonical_ids

# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def pad_basin_id(value) -> str:
    """Return the canonical 8-character zero-padded basin ID string."""
    return str(value).zfill(8)


def canonical_basin_ids() -> List[str]:
    """Return the canonical SORTED basin ID list (same order as loaded matrices)."""
    return list(basin_order())


def file_order_basin_ids() -> List[str]:
    """Return the raw `531sub_id.txt` FILE_ORDER basin ID list (unsorted)."""
    return [pad_basin_id(b) for b in canonical_ids()]


def basin_position_map(basins: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Return {basin_id -> row position} for the given order (default: canonical SORTED)."""
    ids = list(basins) if basins is not None else canonical_basin_ids()
    return {pad_basin_id(b): i for i, b in enumerate(ids)}


def basin_order_mismatch_count() -> int:
    """Number of positions where FILE_ORDER disagrees with SORTED (expected 439)."""
    fo = file_order_basin_ids()
    so = canonical_basin_ids()
    return int(sum(1 for a, b in zip(fo, so) if a != b))


# ---------------------------------------------------------------------------
# Alignment accessors
# ---------------------------------------------------------------------------


def align_series_by_id(ids: Sequence, values: Sequence[float],
                       canonical: Optional[Sequence[str]] = None,
                       name: str = "series") -> np.ndarray:
    """Realign a series given in `ids` order onto the canonical SORTED basin order by ID.

    Raises KeyError if any canonical basin is absent from `ids`, and ValueError
    if `ids` contains duplicates.
    """
    canon = list(canonical) if canonical is not None else canonical_basin_ids()
    ids_p = [pad_basin_id(b) for b in ids]
    if len(set(ids_p)) != len(ids_p):
        raise ValueError(f"{name}: duplicate basin ids in source ({len(ids_p)} rows)")
    pos = {b: i for i, b in enumerate(ids_p)}
    missing = [b for b in canon if b not in pos]
    if missing:
        raise KeyError(f"{name}: {len(missing)} canonical basins missing from source, e.g. {missing[:5]}")
    vals = np.asarray(values, dtype=np.float64)
    out = np.empty(len(canon), dtype=np.float64)
    for i, b in enumerate(canon):
        out[i] = vals[pos[b]]
    return out


def align_frame_by_id(frame: pd.DataFrame, id_column: str,
                      value_columns: Sequence[str],
                      canonical: Optional[Sequence[str]] = None,
                      name: str = "frame") -> np.ndarray:
    """Return [n_canonical, n_value_columns] aligned by basin ID onto canonical SORTED order."""
    ids = frame[id_column].tolist()
    cols = [np.asarray(frame[c], dtype=np.float64) for c in value_columns]
    mat = np.column_stack(cols) if cols else np.empty((len(ids), 0))
    return align_series_by_id(ids, mat, canonical=canonical, name=name) if mat.ndim == 1 \
        else _align_matrix_by_id(ids, mat, canonical=canonical, name=name)


def _align_matrix_by_id(ids: Sequence, mat: np.ndarray,
                        canonical: Optional[Sequence[str]] = None,
                        name: str = "matrix") -> np.ndarray:
    canon = list(canonical) if canonical is not None else canonical_basin_ids()
    ids_p = [pad_basin_id(b) for b in ids]
    if len(set(ids_p)) != len(ids_p):
        raise ValueError(f"{name}: duplicate basin ids in source ({len(ids_p)} rows)")
    pos = {b: i for i, b in enumerate(ids_p)}
    missing = [b for b in canon if b not in pos]
    if missing:
        raise KeyError(f"{name}: {len(missing)} canonical basins missing from source, e.g. {missing[:5]}")
    mat = np.asarray(mat, dtype=np.float64)
    return mat[[pos[b] for b in canon]]


def pivot_by_id(frame: pd.DataFrame, id_column: str, column_column: str,
                value_column: str, canonical: Optional[Sequence[str]] = None,
                columns_order: Optional[Sequence[str]] = None,
                name: str = "pivot") -> np.ndarray:
    """Pivot long (basin, column, value) rows into [n_canonical, n_columns] by basin ID."""
    canon = list(canonical) if canonical is not None else canonical_basin_ids()
    piv = frame.pivot(index=id_column, columns=column_column, values=value_column)
    piv.index = [pad_basin_id(b) for b in piv.index]
    cols = list(columns_order) if columns_order is not None else list(piv.columns)
    missing = [b for b in canon if b not in piv.index]
    if missing:
        raise KeyError(f"{name}: {len(missing)} canonical basins missing from pivot, e.g. {missing[:5]}")
    return piv.loc[canon, cols].to_numpy(dtype=np.float64)


def lookup_by_id(values_by_basin: Dict[str, float], basin_id: str,
                 default: Optional[float] = None) -> float:
    """Return the value for `basin_id` from an ID-keyed dict, with optional default."""
    key = pad_basin_id(basin_id)
    if key in values_by_basin:
        return values_by_basin[key]
    if default is None:
        raise KeyError(f"basin id {key} not present")
    return float(default)
