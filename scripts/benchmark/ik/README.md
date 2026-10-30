# IK Benchmark

Batched Franka Panda IK to one pose goal per instance: RoboKit vs cuRobo vs PyRoKi.

## Data

- Synthetic: random joint configurations within limits, FK'd to target poses; fresh targets each run.
- Batch sizes 1, 10, 100, 1000, 2000.

## Protocol

- Ours: shipped `presets.basic`. cuRobo 0.7.8 (`franka.yml`, collision off). PyRoKi pinned commit.
- 5 runs per solver and batch size; one batched solve is timed, warmup excluded.
- One shared evaluator scores every solver's joints with the same FK and thresholds (5 mm, 0.05 rad).
- Mean ± std over the 5 runs.

## Metrics

- `time_ms`: wall time of one batched solve (ms) ↓
- `succ`: % of instances within both thresholds ↑
- `pos_err_mm` / `ori_err_rad`: 98th-percentile error over successful instances (mm / rad) ↓
- `*_std`: std over runs

## Run

```shell
uv run scripts/benchmark/ik/run_benchmark.py
```

~3 min on one GPU. `inspect_cases.py` shows saved per-instance results.
