"""Frozen basin-role resolution for canonical dPL OOB runs.

The canonical five-fold assignment is reused from the repository's existing OOB
contract.  A selected outer fold is held out, the next deterministic fold is
used as internal validation, and the remaining folds are optimization basins.
This is an engineering role assignment, not a performance-based split.
"""

from __future__ import annotations
import ast

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from benchmarkparam.scripts.oob.oob_common import (
    NormalizationStats,
    make_fold_assignment,
    validate_fold_contract,
)


@dataclass(frozen=True)
class BasinRoles:
    """The three basin roles used by one canonical run."""

    enabled: bool
    legacy_temporal_only: bool
    split_id: str
    manifest_path: str | None
    manifest_sha256: str | None
    assignment_sha256: str | None
    fold: int | None
    validation_fold: int | None
    optimization_ids: np.ndarray
    internal_validation_ids: np.ndarray
    outer_oob_ids: np.ndarray
    full_universe_ids: np.ndarray
    limited_for_smoke: bool = False

    def __post_init__(self) -> None:
        for name in (
            "optimization_ids",
            "internal_validation_ids",
            "outer_oob_ids",
            "full_universe_ids",
        ):
            values = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if values.size != np.unique(values).size:
                raise ValueError(f"{name} contains duplicate basin IDs")
            object.__setattr__(self, name, values)

        if self.enabled:
            opt = set(map(int, self.optimization_ids))
            internal = set(map(int, self.internal_validation_ids))
            outer = set(map(int, self.outer_oob_ids))
            if opt & internal or opt & outer or internal & outer:
                raise ValueError("optimization, internal validation, and outer OOB basins overlap")
            if opt | internal | outer != set(map(int, self.full_universe_ids)) and not self.limited_for_smoke:
                raise ValueError("basin roles do not cover the configured basin universe")
            if not opt or not internal or not outer:
                raise ValueError("each enabled basin role must contain at least one basin")

    @property
    def optimization_basin_ids(self) -> np.ndarray:
        return self.optimization_ids

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "legacy_temporal_only": self.legacy_temporal_only,
            "split_id": self.split_id,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "assignment_sha256": self.assignment_sha256,
            "fold": self.fold,
            "validation_fold": self.validation_fold,
            "optimization_basin_count": int(self.optimization_ids.size),
            "internal_validation_basin_count": int(self.internal_validation_ids.size),
            "outer_oob_basin_count": int(self.outer_oob_ids.size),
            "optimization_basin_hash": basin_id_hash(self.optimization_ids),
            "internal_validation_basin_hash": basin_id_hash(self.internal_validation_ids),
            "outer_oob_basin_hash": basin_id_hash(self.outer_oob_ids),
            "full_universe_basin_count": int(self.full_universe_ids.size),
            "full_universe_basin_hash": basin_id_hash(self.full_universe_ids),
            "limited_for_smoke": self.limited_for_smoke,
        }


def basin_id_hash(basin_ids: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(sorted(int(value) for value in basin_ids), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment_sha256(rows: Sequence[Mapping[str, int]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["basin_id", "fold"])
    writer.writeheader()
    writer.writerows({"basin_id": int(row["basin_id"]), "fold": int(row["fold"])} for row in rows)
    return hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"basin split manifest must be a mapping: {path}")
    return value


def _take_limit(values: np.ndarray, limit: Any, role: str) -> np.ndarray:
    if limit is None:
        return values
    count = int(limit)
    if count < 1 or count > values.size:
        raise ValueError(f"basin_split smoke limit for {role} must be in 1..{values.size}")
    return values[:count]


def resolve_basin_roles(
    raw_config: Mapping[str, Any] | None,
    all_basin_ids: Sequence[int] | np.ndarray,
    *,
    repo_root: str | Path,
) -> BasinRoles:
    """Resolve a frozen manifest into disjoint role arrays."""
    universe = np.asarray(all_basin_ids, dtype=np.int64).reshape(-1)
    if universe.size == 0 or universe.size != np.unique(universe).size:
        raise ValueError("canonical basin universe must be non-empty and unique")
    raw = dict(raw_config or {})
    if not bool(raw.get("enabled", False)):
        return BasinRoles(
            enabled=False,
            legacy_temporal_only=True,
            split_id="legacy_temporal_only",
            manifest_path=None,
            manifest_sha256=None,
            assignment_sha256=None,
            fold=None,
            validation_fold=None,
            optimization_ids=universe.copy(),
            # Legacy mode intentionally preserves the historical temporal-only
            # semantics: the same basin IDs are used for post-hoc evaluation.
            internal_validation_ids=universe.copy(),
            outer_oob_ids=np.empty(0, dtype=np.int64),
            full_universe_ids=universe.copy(),
        )

    manifest_value = raw.get("manifest")
    if not manifest_value:
        raise ValueError("basin_split.enabled requires basin_split.manifest")
    manifest_path = _resolve_path(manifest_value, Path(repo_root)).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"basin split manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    split_id = str(manifest.get("split_id", manifest_path.stem))

    configured_universe = manifest.get("basin_universe")
    if configured_universe:
        universe_path = _resolve_path(configured_universe, Path(repo_root))
        raw_text = universe_path.read_text(encoding="utf-8").strip()
        try:
            expected_values = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            expected_values = [line.strip() for line in raw_text.splitlines() if line.strip()]
        expected_universe = np.asarray(expected_values, dtype=np.int64)
        if set(map(int, expected_universe)) != set(map(int, universe)):
            raise ValueError("basin split manifest universe does not match runner basin IDs")

    assignment_cfg = manifest.get("assignment", {})
    if not isinstance(assignment_cfg, Mapping):
        raise ValueError("basin split manifest assignment must be a mapping")
    n_folds = int(assignment_cfg.get("n_folds", 5))
    random_state = int(assignment_cfg.get("random_state", 20260902))
    if raw.get("seed") is not None and int(raw["seed"]) != random_state:
        raise ValueError("basin_split.seed does not match the frozen manifest random_state")
    rows = make_fold_assignment(universe, n_folds=n_folds, random_state=random_state)
    expected_assignment_hash = assignment_cfg.get("sha256")
    assignment_hash = _assignment_sha256(rows)
    if expected_assignment_hash and assignment_hash != str(expected_assignment_hash):
        raise ValueError("generated basin assignment does not match frozen manifest hash")
    # Reuse the existing OOB contract checker when the frozen five-fold
    # assignment is selected.  It checks IDs, fold sizes, and disjointness.
    validate_fold_contract(universe, rows, n_folds=n_folds, expected_count=universe.size)

    fold = int(raw.get("fold", manifest.get("fold", 0)))
    policy = manifest.get("role_policy", {})
    if not isinstance(policy, Mapping):
        raise ValueError("basin split role_policy must be a mapping")
    validation_fold = raw.get("validation_fold", policy.get("validation_fold"))
    if validation_fold is None:
        validation_fold = (fold + 1) % n_folds
    validation_fold = int(validation_fold)
    if fold < 0 or fold >= n_folds or validation_fold < 0 or validation_fold >= n_folds:
        raise ValueError("basin_split fold is outside the frozen assignment range")
    if fold == validation_fold:
        raise ValueError("outer OOB fold and internal validation fold must differ")

    assignment_map = {int(row["basin_id"]): int(row["fold"]) for row in rows}
    outer = np.asarray(sorted(basin_id for basin_id in universe if assignment_map[int(basin_id)] == fold), dtype=np.int64)
    internal = np.asarray(sorted(basin_id for basin_id in universe if assignment_map[int(basin_id)] == validation_fold), dtype=np.int64)
    optimization = np.asarray(
        sorted(
            basin_id
            for basin_id in universe
            if assignment_map[int(basin_id)] not in {fold, validation_fold}
        ),
        dtype=np.int64,
    )

    limits = raw.get("smoke_limits", {})
    if limits and not isinstance(limits, Mapping):
        raise ValueError("basin_split.smoke_limits must be a mapping")
    limited = bool(limits)
    optimization = _take_limit(optimization, limits.get("optimization") if limits else None, "optimization")
    internal = _take_limit(internal, limits.get("internal_validation") if limits else None, "internal_validation")
    outer = _take_limit(outer, limits.get("outer_oob") if limits else None, "outer_oob")

    return BasinRoles(
        enabled=True,
        legacy_temporal_only=False,
        split_id=split_id,
        manifest_path=str(manifest_path),
        manifest_sha256=_file_sha256(manifest_path),
        assignment_sha256=assignment_hash,
        fold=fold,
        validation_fold=validation_fold,
        optimization_ids=optimization,
        internal_validation_ids=internal,
        outer_oob_ids=outer,
        full_universe_ids=universe.copy(),
        limited_for_smoke=limited,
    )


class TrainFittedAttributeNormalizer:
    """Fit attribute transforms on optimization basins and apply them elsewhere."""

    def __init__(self, stats: NormalizationStats):
        self.stats = stats

    @classmethod
    def fit(cls, raw_attributes: np.ndarray) -> "TrainFittedAttributeNormalizer":
        return cls(NormalizationStats.fit(raw_attributes, method="zscore", log_transform_skewed=True))

    def transform(self, raw_attributes: np.ndarray, *, device: torch.device | str) -> torch.Tensor:
        return torch.as_tensor(self.stats.transform(raw_attributes), dtype=torch.float64, device=device)

    def metadata(self) -> dict[str, Any]:
        return {
            "scope": "optimization_basins_only",
            "method": self.stats.method,
            "log_transform_skewed": self.stats.log_transform_skewed,
            "skewed_attribute_indices": [11, 23, 34],
        }
