# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "tyro",
#   "tabulate",
#   "robot_descriptions",
#   "jax[cuda12]==0.6.2",
#   "pyroki @ git+https://github.com/chungmin99/pyroki.git@d139cbec229a22d9d1232e0b26fdbc0ac66c90f2",
#   "torch==2.9.1",
#   "nvidia-curobo==0.7.8+pt2.9.1cu126",
# ]
#
# [[tool.uv.index]]
# name = "torch-packages-builder"
# url = "https://jianglongye.github.io/torch_packages_builder"
# explicit = true
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# torch = { index = "pytorch-cu126" }
# nvidia-curobo = { index = "torch-packages-builder" }
# ///
"""Run IK benchmark for all implemented solvers.

Example usage:
    uv run scripts/benchmark/ik/run_benchmark.py --batch_sizes 1,10,100
    uv run scripts/benchmark/ik/run_benchmark.py --only RoboKit
"""

import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


# Prevent JAX (used by Pyroki) from preallocating ~75% of GPU memory and starving torch/warp of cuDNN handles.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
import tyro
import warp as wp
from tabulate import tabulate


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.benchmark.ik.helper import IKBenchmarkHelper


try:
    from scripts.benchmark.ik.pyroki_ik import PyrokiIKBenchmarkHelper
except ImportError:
    print("Pyroki is not available, skipping its helpers.")
    PyrokiIKBenchmarkHelper = None


try:
    from scripts.benchmark.ik.curobo_ik import CuroboIKBenchmarkHelper
except ImportError:
    print("Curobo is not available, skipping CuroboIKBenchmarkHelper.")
    CuroboIKBenchmarkHelper = None

from scripts.benchmark.ik.robokit_ik import (
    RoboKitWarpIKBenchmarkHelper,
)


wp.config.quiet = True
wp.config.verify_cuda = False
wp.init()

torch.manual_seed(42)
np.random.seed(42)


@dataclass
class Config:
    """The paper's IK setting: one batched pose goal per instance, scored against robokit FK."""

    batch_sizes: str = "1,10,100,1000,2000"
    num_runs: int = 5
    save_csv: bool = True
    csv_path: str = str(Path(__file__).parent / "results.csv")
    only: str = ""  # comma-separated helper-name substrings, e.g. "RoboKit"; empty = every installed solver


HELPERS: Dict[str, Optional[type[IKBenchmarkHelper]]] = {
    "PyRoKi": PyrokiIKBenchmarkHelper,
    "cuRobo": CuroboIKBenchmarkHelper,
    "RoboKit": RoboKitWarpIKBenchmarkHelper,
}

COLUMNS = [
    "solver",
    "batch",
    "runs",
    "time_ms",
    "time_ms_std",
    "succ",
    "succ_std",
    "pos_err_mm",
    "pos_err_mm_std",
    "ori_err_rad",
    "ori_err_rad_std",
]


def _run_info() -> str:
    """Where the numbers came from: card, CPU, commit and date. Timings mean nothing without them."""
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    cpu = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        f"gpu={gpu[0].strip() if gpu else 'unknown'}"
        f" | cpu={cpu[0].split(':', 1)[1].strip() if cpu else 'unknown'}"
        f" | commit={commit.stdout.strip() or 'unknown'}"
        f" | date={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


def run_benchmark(cfg: Config):
    """Run every installed solver over each batch size and report mean +/- std over `num_runs` runs."""
    keys = [k.strip() for k in cfg.only.split(",") if k.strip()]
    helpers = {n: h for n, h in HELPERS.items() if h is not None and (not keys or any(k in n for k in keys))}

    run_info = _run_info()
    rows = []
    for batch_size in (int(x) for x in cfg.batch_sizes.split(",")):
        for name, helper_class in helpers.items():
            print(f"  {name} (batch_size={batch_size}, num_runs={cfg.num_runs})...")
            out = helper_class(batch_size=batch_size).evaluate(num_runs=cfg.num_runs)
            # each solver's per-instance joints and errors, for inspect_cases.py
            (Path(__file__).parent / "outputs").mkdir(exist_ok=True)
            np.savez(
                Path(__file__).parent / "outputs" / f"{name}.batch{batch_size}.npz",
                q=out.q,
                position_error_mm=out.position_error_mm,
                orientation_error_rad=out.orientation_error_rad,
            )
            rows.append(
                [
                    name,
                    batch_size,
                    cfg.num_runs,
                    float(np.mean(out.time_ms)),
                    float(np.std(out.time_ms)),
                    float(np.mean(out.success)),
                    float(np.std(out.success)),
                    float(np.nanmean(out.position_mm)),
                    float(np.nanstd(out.position_mm)),
                    float(np.nanmean(out.orientation_rad)),
                    float(np.nanstd(out.orientation_rad)),
                ]
            )

    print(f"\nIK benchmark: Panda, mean +/- std over {cfg.num_runs} runs, errors are 98th percentile")
    print(
        tabulate(
            rows,
            headers=COLUMNS,
            tablefmt="grid",
            floatfmt=("", "", "", ".3f", ".3f", ".2f", ".2f", ".4f", ".4f", ".2e", ".2e"),
        )
    )
    if cfg.save_csv:
        with open(cfg.csv_path, "w", newline="") as handle:
            handle.write(f"# {run_info}\n")
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f"saved {cfg.csv_path}")


if __name__ == "__main__":
    run_benchmark(tyro.cli(Config))
