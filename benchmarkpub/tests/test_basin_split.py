from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch
import pytest

from benchmarkpub.basin_split import resolve_basin_roles
from benchmarkpub.specialization import BasinSpecialization, SpecializationConfig


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data/531sub_id.txt").is_file():
            return parent
    return current.parents[3]

REPO_ROOT = _find_repo_root()
MANIFEST = (
    (REPO_ROOT / "project/benchmarkpub/configs/canonical_v2_oob_5fold_split.yaml")
    if (REPO_ROOT / "project/benchmarkpub").exists()
    else (REPO_ROOT / "benchmarkpub/configs/canonical_v2_oob_5fold_split.yaml")
)
IDS = np.asarray(ast.literal_eval((REPO_ROOT / "data/531sub_id.txt").read_text()), dtype=np.int64)

def test_frozen_roles_are_reproducible_disjoint_and_cover_universe() -> None:
    config = {"enabled": True, "manifest": str(MANIFEST), "fold": 0}
    first = resolve_basin_roles(config, IDS, repo_root=REPO_ROOT)
    second = resolve_basin_roles(config, IDS, repo_root=REPO_ROOT)

    assert first.metadata() == second.metadata()
    assert len(first.optimization_ids) == 318
    assert len(first.internal_validation_ids) == 106
    assert len(first.outer_oob_ids) == 107
    assert not set(first.optimization_ids) & set(first.internal_validation_ids)
    assert not set(first.optimization_ids) & set(first.outer_oob_ids)
    assert not set(first.internal_validation_ids) & set(first.outer_oob_ids)
    assert set(np.concatenate((first.optimization_ids, first.internal_validation_ids, first.outer_oob_ids))) == set(IDS)


def test_smoke_limits_remain_disjoint_and_are_explicit() -> None:
    roles = resolve_basin_roles(
        {
            "enabled": True,
            "manifest": str(MANIFEST),
            "fold": 0,
            "smoke_limits": {"optimization": 4, "internal_validation": 2, "outer_oob": 2},
        },
        IDS,
        repo_root=REPO_ROOT,
    )
    assert roles.limited_for_smoke
    assert tuple(map(int, roles.optimization_ids)) == tuple(sorted(roles.optimization_ids))
    assert len(roles.optimization_ids) == 4
    assert len(roles.internal_validation_ids) == 2
    assert len(roles.outer_oob_ids) == 2
    assert set(roles.optimization_ids).isdisjoint(roles.internal_validation_ids)
    assert set(roles.optimization_ids).isdisjoint(roles.outer_oob_ids)


def test_legacy_mode_preserves_temporal_only_roles() -> None:
    roles = resolve_basin_roles({"enabled": False}, IDS, repo_root=REPO_ROOT)
    assert roles.legacy_temporal_only
    assert np.array_equal(roles.optimization_ids, IDS)
    assert np.array_equal(roles.internal_validation_ids, IDS)
    assert roles.outer_oob_ids.size == 0


def test_role_overlap_and_validation_weight_access_fail_fast() -> None:
    with pytest.raises(ValueError, match="must differ"):
        resolve_basin_roles(
            {"enabled": True, "manifest": str(MANIFEST), "fold": 0, "validation_fold": 0},
            IDS,
            repo_root=REPO_ROOT,
        )

    roles = resolve_basin_roles(
        {"enabled": True, "manifest": str(MANIFEST), "fold": 0}, IDS, repo_root=REPO_ROOT
    )
    state = BasinSpecialization(
        roles.optimization_ids.tolist(),
        SpecializationConfig.from_mapping({"enabled": True, "method": "rank_sigmoid"}),
        device="cpu",
    )
    with pytest.raises(ValueError, match="optimization set"):
        state.observe(
            roles.internal_validation_ids[:1].tolist(),
            torch.tensor([0.2]),
            torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="optimization set"):
        state.weights_for(roles.outer_oob_ids[:1].tolist())
