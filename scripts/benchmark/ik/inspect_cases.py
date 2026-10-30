# /// script
# dependencies = ["robokit[pt-cu126]", "robot_descriptions"]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# torch = { index = "pytorch-cu126" }
# ///
"""Inspect saved IK results, failures first, without solving.

Displays:
- Target: sampled joint configuration (uncolored) and its FK target frame.
- Solutions: saved joints and errors, with RoboKit blue, cuRobo orange, and PyRoKi purple.

    uv run scripts/benchmark/ik/inspect_cases.py --batch-size 10
"""

import sys
import time
from pathlib import Path

import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.benchmark.ik.helper import IKBenchmarkHelper, load_panda_urdf_fixed
from scripts.benchmark.ik.robokit_ik import RoboKitWarpIKBenchmarkHelper


COLORS = {
    "RoboKit": ("blue", (0.2, 0.4, 0.9, 0.6)),
    "cuRobo": ("orange", (0.95, 0.55, 0.1, 0.6)),
    "PyRoKi": ("purple", (0.6, 0.3, 0.8, 0.6)),
}


def main(batch_size: int = 10) -> None:
    solved = {
        p.name.split(".")[0]: np.load(p) for p in (Path(__file__).parent / "outputs").glob(f"*.batch{batch_size}.npz")
    }
    num_runs = next(iter(solved.values()))["q"].shape[0] if solved else 5  # 5: run_benchmark.py's default
    helper = RoboKitWarpIKBenchmarkHelper(batch_size=batch_size)
    problems = []
    for run in range(num_runs):  # the same calls, in the same order, as `IKBenchmarkHelper.evaluate`
        q = helper.sample_q(run)
        problems.append((q, helper.forward_kinematics(q)))
    reached = {
        name: (s["position_error_mm"] < IKBenchmarkHelper.POSITION_THRESHOLD * 1000)
        & (s["orientation_error_rad"] < IKBenchmarkHelper.ROTATION_THRESHOLD)
        for name, s in solved.items()
    }
    # failures first, then the largest position error
    cases = sorted(
        ((run, i) for run in range(num_runs) for i in range(batch_size)),
        key=lambda c: (
            sum(r[c] for r in reached.values()),
            -max([s["position_error_mm"][c] for s in solved.values()] + [0.0]),
        ),
    )
    server = viser.ViserServer()
    robots = {
        role: ViserUrdf(
            server,
            load_panda_urdf_fixed(),
            root_node_name=f"/{role}",
            mesh_color_override=COLORS[role][1] if role in COLORS else None,
        )
        for role in ("truth", *solved)
    }
    order = [helper.reference_joint_names.index(name) for name in robots["truth"].get_actuated_joint_names()]
    case = server.gui.add_slider("Case", min=0, max=max(len(cases) - 1, 1), step=1, initial_value=0)
    server.gui.add_markdown(
        " · ".join(["truth: uncolored", *(f"{name}: {COLORS[name][0]}" for name in solved), "target: axes"])
    )
    info = server.gui.add_markdown("")

    @case.on_update
    def show(_=None) -> None:
        run, i = cases[min(case.value, len(cases) - 1)]  # a slider needs two stops
        q, targets = problems[run]
        robots["truth"].update_cfg(q[i][order])
        for name, s in solved.items():
            robots[name].update_cfg(s["q"][run, i][order])
        server.scene.add_frame(
            "/target", position=targets[i, :3], wxyz=targets[i, 3:], axes_length=0.1, axes_radius=0.005
        )
        lines = [f"run {run} · instance {i}"]
        lines += [
            f"**{name}** {'pass' if reached[name][run, i] else 'FAIL'}: "
            f"{s['position_error_mm'][run, i]:.3f} mm · {s['orientation_error_rad'][run, i]:.2e} rad"
            for name, s in solved.items()
        ]
        info.content = "\n\n".join(lines)

    show()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
