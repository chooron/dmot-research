#!/usr/bin/env python3
"""Create and verify a dated, source-backed data-freeze record."""
from __future__ import annotations
import ast
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
ANALYSIS = REPO / "project/benchmarkpub/manuscript/analysis"
REGISTRY = ANALYSIS / "_results_registry"
FREEZE = ANALYSIS / "_freeze"
DATE = dt.date.today().strftime("%Y%m%d")
RECORD = FREEZE / f"freeze_record_{DATE}.json"

EXTRA_FILES = [
    (REPO / "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_candidate_set_transfer_20260930/tables/top3_top5_set_overlap.csv", "f1_source_result"),
    (REPO / "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_supplemental_rank_extreme_20260930/tables/topk_overlap_summary.csv", "f1_comparison_result"),
    (REPO / "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_supplemental_rank_extreme_20260930/tables/basin_rank_stability.csv", "f1_per_basin_validation"),
    (REPO / "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv", "f1_per_basin_input"),
    (REPO / "project/benchmarkpub/scripts/oob36_prelim/22_candidate_set_transfer.py", "f1_source_script"),
    (REPO / "project/benchmarkpub/scripts/oob36_prelim/18_basin_rank_stability.py", "f1_comparison_script"),
    (REPO / "project/benchmarkpub/scripts/oob36_prelim/pub_common.py", "f1_shared_source_helper"),
    (REPO / "project/benchmark/analysis/chapter4_final_supplement_20260925/tables/f5_seed_per_seed.csv", "t1_source_result"),
    (REPO / "project/benchmark/analysis/chapter4_final_supplement_20260925/scripts/f5_seed_robustness.py", "t1_source_script"),
    (REPO / "project/benchmark/analysis/chapter4_final_supplement_20260925/logs/f5.log", "t1_provenance_log"),
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def file_kind(path: Path) -> str | None:
    parts = path.relative_to(ANALYSIS).parts
    if "__pycache__" in parts:
        return None
    if parts[0] == "_freeze":
        return None
    if parts[0] == "_results_registry":
        return "registry"
    if "declarations" in parts:
        return "declaration"
    if "tables" in parts:
        return "result_table"
    if parts[0] == "_figures" and ("data" in parts or path.name == "figure_panel_map.csv"):
        return "figure_data"
    if "scripts" in parts and path.suffix == ".py":
        return "script"
    return None


def inventory() -> list[dict[str, object]]:
    files = []
    seen = set()
    for path in sorted(ANALYSIS.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        kind = file_kind(path)
        if kind is None:
            continue
        seen.add(path)
        files.append({"path": rel(path), "kind": kind, "sha256": sha256(path), "bytes": path.stat().st_size})
    for path, kind in EXTRA_FILES:
        if path.is_file() and not path.is_symlink() and path not in seen:
            files.append({"path": rel(path), "kind": kind, "sha256": sha256(path), "bytes": path.stat().st_size})
    return sorted(files, key=lambda item: item["path"])


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def seeds_from_ast(path: Path) -> list[dict[str, object]]:
    found = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return found
    nodes = list(ast.walk(tree))
    for node in nodes:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        if not targets or value is None:
            continue
        for target in targets:
            if "seed" not in target.id.lower():
                continue
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(literal, (int, float, str, list, tuple)):
                found.append({"source": rel(path), "variable": target.id, "value": literal})
    return found


def seed_inventory() -> list[dict[str, object]]:
    found = []
    for path in sorted(ANALYSIS.rglob("*.py")):
        found.extend(seeds_from_ast(path))
    for path in sorted((REGISTRY / "declarations").glob("*.csv")):
        try:
            rows = csv_rows(path)
        except UnicodeDecodeError:
            continue
        if not rows:
            continue
        fields = [f for f in rows[0] if "seed" in f.lower()]
        for field in fields:
            values = sorted({r.get(field, "") for r in rows if r.get(field, "") != ""})
            if values:
                found.append({"source": rel(path), "variable": field, "value": values})
    return found


def versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib"]
    result = {name: (importlib.metadata.version(name) if importlib.metadata.packages_distributions() else "unavailable") for name in names}
    for name in list(result):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def audit_summary() -> dict[str, str]:
    path = REGISTRY / "tables/consistency_audit.csv"
    return {r["check_id"]: r["status"] for r in csv_rows(path)}


def registry_count() -> int:
    return len(csv_rows(REGISTRY / "tables/results_claim_registry.csv"))


def record() -> dict[str, object]:
    files = inventory()
    return {
        "freeze_date": DATE,
        "repository_relative_paths": True,
        "scope": [
            "project/benchmarkpub/manuscript/analysis/_results_registry/**",
            "project/benchmarkpub/manuscript/analysis/**/tables/**",
            "project/benchmarkpub/manuscript/analysis/**/scripts/*.py",
            "project/benchmarkpub/manuscript/analysis/_figures/data/**",
            "project/benchmarkpub/manuscript/analysis/_figures/figure_panel_map.csv",
            "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_candidate_set_transfer_20260930/tables/top3_top5_set_overlap.csv",
            "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_supplemental_rank_extreme_20260930/tables/topk_overlap_summary.csv",
            "project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930/pub_next_stage_20260930/tables/pub_same_pipeline_model_basin.csv",
            "project/benchmarkpub/scripts/oob36_prelim/{18_basin_rank_stability.py,22_candidate_set_transfer.py,pub_common.py}",
            "project/benchmark/analysis/chapter4_final_supplement_20260925/{tables/f5_seed_per_seed.csv,scripts/f5_seed_robustness.py,logs/f5.log}",
        ],
        "files": files,
        "file_count": len(files),
        "registry_entry_count": registry_count(),
        "audit_results": audit_summary(),
        "python": {"version": sys.version, "implementation": platform.python_implementation(), "executable": str(Path(sys.executable).resolve())},
        "key_package_versions": versions(),
        "random_seeds": seed_inventory(),
        "hash_verification_command": f".venv/bin/python -c \"import hashlib,json,pathlib; r=json.load(open('{rel(RECORD)}')); bad=[x['path'] for x in r['files'] if hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()!=x['sha256'] or pathlib.Path(x['path']).stat().st_size!=x['bytes']]; print('hash_mismatches='+str(len(bad))); raise SystemExit(bool(bad))\"",
        "freeze_rule": "After this freeze, any numerical change requires a new versioned freeze directory and written reason; do not modify frozen files in place.",
    }


def verify(data: dict[str, object]) -> list[str]:
    bad = []
    for item in data["files"]:
        path = REPO / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            bad.append(item["path"])
    return bad


def main() -> None:
    FREEZE.mkdir(parents=True, exist_ok=True)
    data = record()
    RECORD.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    bad = verify(data)
    if bad:
        raise SystemExit("freeze hash verification failed: " + ", ".join(bad[:10]))
    frozen_dirs = sorted({item["path"].rsplit("/", 1)[0] for item in data["files"]})
    frozen_md = FREEZE / "FROZEN.md"
    frozen_md.write_text(
        "# Data freeze\n\n"
        f"- Freeze date: {DATE}\n"
        f"- Freeze record: `{rel(RECORD)}`\n"
        f"- Files recorded: {data['file_count']}\n"
        f"- Registry entries: {data['registry_entry_count']}\n\n"
        "## Frozen scope\n\n" + "\n".join(f"- `{d}/`" for d in frozen_dirs) + "\n\n"
        "## Verification\n\n"
        f"Run the following command from the repository root:\n\n```bash\n{data['hash_verification_command']}\n```\n\n"
        "## Rule\n\n"
        "After this freeze, any numerical change requires a new versioned freeze directory and a written reason; frozen files must not be modified in place.\n",
        encoding="utf-8",
    )
    print(json.dumps({"record": rel(RECORD), "frozen_md": rel(frozen_md), "file_count": data["file_count"], "registry_entry_count": data["registry_entry_count"], "hash_mismatches": len(bad)}, sort_keys=True))


if __name__ == "__main__":
    main()
