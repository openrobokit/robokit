import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.benchmark.motion_plan import benchmark


pytestmark = pytest.mark.torch
pytest.importorskip("scipy")


class TestMotionPlanMetrics:
    @pytest.fixture
    def rows(self) -> list:
        q = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]]
        return [
            {
                "case": f"toy/{i}",
                "split": "toy",
                "reach": True,
                "success": True,
                "success_strict": True,
                "time": value,
                "motion": value,
                "path": value,
                "jerk": value * 10,
                "q": q,
                "dt": 1.0,
                "judge": {"pos_err_mm": value, "rot_err": value / 100},
            }
            for i, value in enumerate((1.0, 3.0))
        ] + [
            {"case": "toy/2", "split": "toy", "reach": False, "success": False, "success_strict": False, "time": 100.0}
        ]

    @pytest.fixture
    def saved_runs(self, tmp_path: Path, rows: list) -> benchmark.Config:
        cfg = benchmark.Config(config="robokit-primitive", runs=2, output_dir=tmp_path, aggregate_only=True)
        shards = tmp_path / "outputs"
        shards.mkdir()
        for k in range(cfg.runs):
            (shards / f"{cfg.config}.run{k}.json").write_text(json.dumps(rows))
            metadata = {
                "seed": cfg.seed + k,
                "config": {"runs": cfg.runs, "sample_per_split": 0},
                "source_sha256": "source",
                "dataset_sha256": "dataset",
                "hardware": "saved hardware",
            }
            (shards / f"{cfg.config}.run{k}.meta.json").write_text(json.dumps(metadata))
        (tmp_path / "results.csv").write_text("# original hardware and commit\nconfig,split,runs,n\n")
        return cfg

    def test_derivatives_match_cubic_motion(self):
        t = np.arange(9, dtype=np.float64) * 0.2
        q = np.stack((t**3, -2 * t**3), axis=-1)
        v, a, j = benchmark.trajectory_derivatives(q, 0.2)
        for actual, expected in ((v, 3 * t**2), (a, 6 * t), (j, np.full_like(t, 6))):
            expected = np.stack((expected, -2 * expected), axis=-1)
            torch.testing.assert_close(torch.as_tensor(actual[2:-2]), torch.as_tensor(expected[2:-2]))

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_dt(self, dt: float):
        with pytest.raises(ValueError, match="positive and finite"):
            benchmark.trajectory_derivatives(np.zeros((5, 2)), dt)

    def test_recover_metrics_and_exclude_failures(self, rows: list):
        result = benchmark.summarize([rows], "overall")
        expected = {
            "time_mean": 104 / 3,
            "time_success_mean": 2.0,
            "time_success_p98": 2.96,
            "motion": 2.0,
            "motion_case_std": 1.0,
            "motion_median": 2.0,
            "motion_p75": 2.5,
            "motion_p95": 2.9,
            "motion_p98": 2.96,
            "path": 2.0,
            "jerk": 20.0,
            "velocity": 7 / 30,  # Includes the stationary joint and held frames.
            "acceleration": 7 / 6,
            "position_error": 2.0,
            "rotation_error": 0.02,
        }
        torch.testing.assert_close(torch.tensor([result[k] for k in expected]), torch.tensor(list(expected.values())))
        assert np.isnan(result["time_native_mean"])

    def test_run_uncertainty_is_separate_from_problem_spread(self, rows: list):
        rows[1]["motion"] = 9.0
        other = copy.deepcopy(rows)
        for row in other[:2]:
            row["motion"] += 2.0
        result = benchmark.summarize([rows, other], "toy")
        torch.testing.assert_close(
            torch.tensor([result[k] for k in ("motion", "motion_case_std", "motion_std", "motion_ci95")]),
            torch.tensor([6.0, 4.0, 1.0, 12.706204736432095]),
        )

    def test_no_successes_have_unavailable_quality_statistics(self, rows: list):
        for row in rows:
            row["success"] = False
        result = benchmark.summarize([rows], "overall")
        assert result["sr"] == 0.0
        assert np.isnan(result["motion"]) and np.isnan(result["velocity"])
        assert np.isnan(result["time_mean_ci95"])
        assert np.isfinite(result["time_mean"])

    def test_native_timing_uses_native_successes(self, rows: list):
        for row, native_time, native_success in zip(rows, (2.0, 8.0, 5.0), (True, False, True)):
            row.update(native_time=native_time, native_success=native_success)
        result = benchmark.summarize([rows], "overall")
        torch.testing.assert_close(
            torch.tensor([result["time_native_mean"], result["time_native_p98"], result["time_success_mean"]]),
            torch.tensor([3.5, 4.94, 2.0]),
        )

    def test_aggregate_only_preserves_provenance_without_running_planners(
        self, saved_runs: benchmark.Config, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(benchmark.tyro, "cli", lambda cls: saved_runs)
        for name in ("load_datasets", "run_config_once", "run_info"):
            monkeypatch.setattr(
                benchmark, name, lambda *args: pytest.fail("Aggregation invoked a planner or hardware probe")
            )
        monkeypatch.setattr(benchmark.wp, "init", lambda: pytest.fail("Aggregation initialized Warp"))
        benchmark.main()
        text = (saved_runs.output_dir / "results.csv").read_text()
        assert text.startswith("# original hardware and commit\n")
        csv_rows = list(csv.DictReader(text.splitlines()[1:]))
        assert {r["split"] for r in csv_rows} == {"toy", "overall"}
        row = next(r for r in csv_rows if r["split"] == "overall")
        assert row["velocity_mean_rad_s"] == "0.233"
        assert row["acceleration_mean_rad_s2"] == "1.167"
        assert row["position_error_mean_mm"] == "2.0000"
        assert "motion_p95_s_ci95" in row

    @pytest.mark.parametrize("key", ["source_sha256", "dataset_sha256"])
    def test_aggregate_only_rejects_mixed_experiments(self, saved_runs: benchmark.Config, key: str):
        path = saved_runs.output_dir / "outputs" / f"{saved_runs.config}.run1.meta.json"
        metadata = json.loads(path.read_text())
        metadata[key] = "different"
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="different source or dataset hashes"):
            benchmark.run_config(saved_runs, saved_runs.config)

    def test_aggregate_only_rejects_missing_runs(self, saved_runs: benchmark.Config):
        (saved_runs.output_dir / "outputs" / f"{saved_runs.config}.run1.json").unlink()
        with pytest.raises(FileNotFoundError, match="Missing saved runs"):
            benchmark.run_config(saved_runs, saved_runs.config)

    def test_aggregate_only_rejects_different_cases(self, saved_runs: benchmark.Config):
        path = saved_runs.output_dir / "outputs" / f"{saved_runs.config}.run1.json"
        rows = json.loads(path.read_text())
        rows[0]["case"] = "different"
        path.write_text(json.dumps(rows))
        with pytest.raises(ValueError, match="different cases or ordering"):
            benchmark.run_config(saved_runs, saved_runs.config)
