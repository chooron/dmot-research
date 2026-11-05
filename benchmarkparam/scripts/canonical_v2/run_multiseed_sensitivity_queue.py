#!/usr/bin/env python3
"""5-Worker Queue Scheduler for JoH 8-Model dPL Multi-Seed Sensitivity (Seeds 123 & 2026).

Enforces:
1. TARGET_MODELS = ['alpine2', 'hbv96', 'xinanjiang', 'newzealand2', 'ihacres', 'us1', 'mopex4', 'hillslope']
2. SEEDS = [123, 2026] (Total 16 training jobs; seed 42 reused from canonical v2).
3. TARGET_CONCURRENCY = 5 (Constant 5 concurrent workers on GPU).
4. Dynamic slot replacement (Fills empty slot immediately upon model completion).
5. Process-level isolation, per-job stdout/stderr logging, .lock protection.
6. Continuous ledger updates to MULTISEED_LEDGER.md.
7. Optional --wait-for-pids: polls until specified upstream processes finish, then starts queue.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
RUNNER_SCRIPT = BENCHMARK_ROOT / "scripts/canonical_v2/run_canonical_v2_model.py"
CANONICAL_V2_RUNS = BENCHMARK_ROOT / "results/dpl_canonical_v2_20260831/runs"

TARGET_MODELS = [
    "alpine2",
    "hbv96",
    "xinanjiang",
    "newzealand2",
    "ihacres",
    "us1",
    "mopex4",
    "hillslope",
]

NEW_SEEDS = [123, 2026]


@dataclass
class JobProcess:
    model: str
    seed: int
    run_name: str
    job_config: dict[str, Any]
    process: subprocess.Popen
    stdout_file: Any
    stderr_file: Any
    start_time: float
    retry_count: int = 0


def build_multiseed_manifest(output_dir: Path) -> dict[str, Any]:
    """Construct the 16-job manifest for 8 models x 2 alternative seeds."""
    jobs = []
    for model in TARGET_MODELS:
        for seed in NEW_SEEDS:
            run_name = f"{model}_seed_{seed}"
            cfg = {
                "model": model,
                "seed": seed,
                "run_name": run_name,
                "epochs": 100,
                "min_epochs": 50,
                "patience": 10,
                "plateau_eps": 0.0001,
                "horizon_days": 1095,
                "warmup_days": 730,
                "scored_days": 365,
                "warmup_grad_mode": "detach",
                "lr": 0.001,
                "weight_decay": 0.0001,
                "clip_norm": 1.0,
                "mapping": "auto",
                "selection_metric": "train_loss",
                "status": "ELIGIBLE",
                "train_period": ["1980-10-01", "1995-09-30"],
                "test_period": ["1995-10-01", "2010-09-30"],
                "test_warmup_days": 365,
            }
            jobs.append(cfg)

    manifest = {
        "benchmark": "JoH 8-Model dPL Multi-Seed Sensitivity",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": TARGET_MODELS,
        "seeds": NEW_SEEDS,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    return manifest


def get_job_status(output_dir: Path, run_name: str) -> str:
    run_dir = output_dir / "runs" / run_name
    if not run_dir.exists():
        return "QUEUED"
    if (run_dir / "DONE").exists():
        return "DONE"
    if (run_dir / "FAILED").exists():
        return "FAILED"
    if (run_dir / ".lock").exists():
        return "RUNNING"
    return "QUEUED"


def init_ledger(output_dir: Path) -> None:
    ledger_path = output_dir / "MULTISEED_LEDGER.md"
    if not ledger_path.exists():
        text = """# JoH 8-Model dPL Multi-Seed Sensitivity Execution Ledger

| Model | Seed | Run Name | Status | Best Ep | Best Train Loss | Test Median KGE | Test Q25 KGE | Runtime (s) | Run Dir |
|---|---|---|---|---|---|---|---|---|---|
"""
        ledger_path.write_text(text)


def append_to_ledger(output_dir: Path, job_config: dict[str, Any], result: dict[str, Any], status: str) -> None:
    ledger_path = output_dir / "MULTISEED_LEDGER.md"
    init_ledger(output_dir)

    model = job_config["model"]
    seed = job_config["seed"]
    run_name = job_config.get("run_name", f"{model}_seed_{seed}")

    best_ep = result.get("best_epoch", "N/A")
    best_loss = f"{result.get('best_train_loss', 0.0):.4f}" if "best_train_loss" in result else "N/A"
    test_med = f"{result.get('test_median_kge', 0.0):.4f}" if "test_median_kge" in result else "N/A"
    test_q25 = f"{result.get('test_q25_kge', 0.0):.4f}" if "test_q25_kge" in result else "N/A"
    runtime_s = f"{result.get('runtime_seconds', 0.0):.1f}" if "runtime_seconds" in result else "N/A"

    row = (
        f"| `{model}` | {seed} | `{run_name}` | **`{status}`** | {best_ep} | "
        f"{best_loss} | **{test_med}** | {test_q25} | {runtime_s} | `runs/{run_name}/` |\n"
    )

    with open(ledger_path, "a") as f:
        f.write(row)


def generate_summary(output_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    rows = []
    runs_dir = output_dir / "runs"

    for j in manifest["jobs"]:
        m = j["model"]
        s = j["seed"]
        rn = j["run_name"]
        run_dir = runs_dir / rn
        status = get_job_status(output_dir, rn)

        test_file = run_dir / "test_evaluation.json"
        runtime_file = run_dir / "runtime.json"

        test_med = np.nan
        test_q25 = np.nan
        test_mean = np.nan
        best_loss = np.nan
        best_ep = np.nan
        runtime_s = np.nan

        if test_file.exists():
            try:
                tv = json.loads(test_file.read_text())
                test_med = float(tv.get("median", np.nan))
                test_q25 = float(tv.get("q25", np.nan))
                test_mean = float(tv.get("mean", np.nan))
            except Exception:
                pass

        if runtime_file.exists():
            try:
                rt = json.loads(runtime_file.read_text())
                best_loss = float(rt.get("best_train_loss", np.nan))
                best_ep = int(rt.get("best_epoch", 1))
                runtime_s = float(rt.get("total_runtime_s", np.nan))
            except Exception:
                pass

        rows.append({
            "model": m,
            "seed": s,
            "run_name": rn,
            "status": status,
            "best_epoch": best_ep,
            "best_train_loss": best_loss,
            "test_median_kge": test_med,
            "test_q25_kge": test_q25,
            "test_mean_kge": test_mean,
            "runtime_seconds": runtime_s,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "MULTISEED_SUMMARY.csv", index=False)
    return df


def link_canonical_seed42(output_dir: Path) -> None:
    """Link canonical seed 42 runs into output_dir/runs/{model}_seed_42 for unified access."""
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for m in TARGET_MODELS:
        src = CANONICAL_V2_RUNS / m
        dst = runs_dir / f"{m}_seed_42"
        if src.exists() and not dst.exists():
            try:
                os.symlink(str(src), str(dst))
                print(f"[Canonical Link] Linked {src} -> {dst}")
            except Exception as exc:
                print(f"[Canonical Link Warning] Could not symlink {src} -> {dst}: {exc}")


def wait_for_upstream_pids(pids: list[int], poll_interval: int = 30) -> None:
    """Block until all specified PIDs have exited."""
    if not pids:
        return
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for upstream PIDs {pids} to finish...")
    while True:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
        if not alive:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All upstream PIDs finished!")
            break
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Still waiting for PIDs: {alive} (next check in {poll_interval}s)")
        time.sleep(poll_interval)


def run_queue(
    output_dir: Path,
    concurrency: int = 5,
    device: str = "cuda",
    poll_interval: int = 10,
    wait_pids: list[int] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Wait for upstream processes if requested
    if wait_pids:
        wait_for_upstream_pids(wait_pids, poll_interval=30)

    # 2. Link canonical seed 42 runs
    link_canonical_seed42(output_dir)

    # 3. Build or load manifest
    manifest_path = output_dir / "multiseed_manifest.yaml"
    if not manifest_path.exists():
        manifest = build_multiseed_manifest(output_dir)
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, indent=2)
    else:
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

    init_ledger(output_dir)

    job_list = manifest["jobs"]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Multi-Seed Queue: {len(job_list)} jobs, concurrency={concurrency}, device={device}")

    active_processes: dict[str, JobProcess] = {}

    def launch_job(job_cfg: dict[str, Any]) -> JobProcess:
        m = job_cfg["model"]
        s = job_cfg["seed"]
        rn = job_cfg["run_name"]
        stdout_path = logs_dir / f"{rn}.stdout.log"
        stderr_path = logs_dir / f"{rn}.stderr.log"

        out_f = open(stdout_path, "a")
        err_f = open(stderr_path, "a")

        cmd = [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--job-config",
            json.dumps(job_cfg),
            "--out",
            str(output_dir),
            "--device",
            device,
        ]

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching {rn} (model={m}, seed={s})...")
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
        return JobProcess(
            model=m,
            seed=s,
            run_name=rn,
            job_config=job_cfg,
            process=proc,
            stdout_file=out_f,
            stderr_file=err_f,
            start_time=time.time(),
        )

    # Main queue loop
    while True:
        # Check active processes
        finished = []
        for rn, jp in active_processes.items():
            ret = jp.process.poll()
            if ret is not None:
                jp.stdout_file.close()
                jp.stderr_file.close()
                runtime_s = time.time() - jp.start_time

                status = get_job_status(output_dir, rn)
                run_dir = output_dir / "runs" / rn
                result = {"runtime_seconds": runtime_s}

                if (run_dir / "runtime.json").exists():
                    try:
                        rt_data = json.loads((run_dir / "runtime.json").read_text())
                        result.update(rt_data)
                    except Exception:
                        pass

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished {rn}: exit_code={ret}, status={status}, runtime={runtime_s:.1f}s")
                append_to_ledger(output_dir, jp.job_config, result, status)
                finished.append(rn)

        for rn in finished:
            del active_processes[rn]

        # Fill available slots
        available_slots = concurrency - len(active_processes)
        if available_slots > 0:
            for job_cfg in job_list:
                rn = job_cfg["run_name"]
                if rn in active_processes:
                    continue
                status = get_job_status(output_dir, rn)
                if status == "DONE":
                    continue
                if status == "FAILED":
                    continue
                if status == "RUNNING":
                    # Lock exists but not tracked by this process: check if actually alive
                    lock_file = output_dir / "runs" / rn / ".lock"
                    if lock_file.exists():
                        continue

                # Slot available -> launch
                jp = launch_job(job_cfg)
                active_processes[rn] = jp
                available_slots -= 1
                if available_slots <= 0:
                    break

        # Check termination condition
        all_completed = True
        for job_cfg in job_list:
            rn = job_cfg["run_name"]
            st = get_job_status(output_dir, rn)
            if st not in ("DONE", "FAILED"):
                all_completed = False
                break

        if all_completed and len(active_processes) == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALL {len(job_list)} JOBS HAVE REACHED TERMINAL STATE!")
            break

        # Periodic summary
        generate_summary(output_dir, manifest)
        time.sleep(poll_interval)

    # Final summary update
    df_sum = generate_summary(output_dir, manifest)
    print("\n=== FINAL SUMMARY ===")
    print(df_sum.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Seed Sensitivity Queue Runner (5 Workers)")
    parser.add_argument("--out", type=str, default=str(BENCHMARK_ROOT / "results/dpl_multiseed_sensitivity_20260914"))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--wait-pids", type=str, default=None, help="Comma-separated PIDs to wait for before starting queue")
    args = parser.parse_args()

    pids = [int(p.strip()) for p in args.wait_pids.split(",") if p.strip()] if args.wait_pids else None
    run_queue(
        output_dir=Path(args.out),
        concurrency=args.concurrency,
        device=args.device,
        poll_interval=args.poll_interval,
        wait_pids=pids,
    )


if __name__ == "__main__":
    main()
