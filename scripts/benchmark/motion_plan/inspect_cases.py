# /// script
# dependencies = [
#   "robokit",
#   "robometrics @ git+https://github.com/fishbotics/robometrics.git",
# ]
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# ///
"""Replay saved motion plans with the judge's verdict, failures first; no planner runs.

Displays:
- Problem: obstacles, goal frame, and a green robot at the dataset's goal IK.
- Plan: RoboKit blue or cuRobo orange; before results exist, the robot stays at its start.

    uv run scripts/benchmark/motion_plan/inspect_cases.py
"""

import json
import sys
import time
from pathlib import Path

import tyro
import viser
from viser.extras import ViserUrdf


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.benchmark.motion_plan.benchmark import (
    RUN_SHARD_DIR,
    cylinder_to_trimesh,
    load_datasets,
    load_panda_urdf_fixed,
    mesh_to_trimesh,
    obb_to_trimesh,
)


COLORS = {
    "goal": (0.2, 0.8, 0.3, 0.4),
    "robokit-primitive": (0.2, 0.4, 0.9, 0.6),
    "curobo-primitive": (0.95, 0.55, 0.1, 0.6),
}


def main() -> None:
    cases = [(split, p) for split, scenes in load_datasets() for ps in scenes.values() for p in ps]
    rows: dict = {}  # case -> {config: judged row}
    for shard in sorted(RUN_SHARD_DIR.glob("*.run0.json")):
        for row in json.loads(shard.read_text()):
            rows.setdefault(row["case"], {})[shard.name.split(".run")[0]] = row
    if rows:
        cases = [c for c in cases if c[1]["case"] in rows]
        # failures first, then strict failures, then the deepest world penetration
        cases.sort(
            key=lambda c: (
                sum(r["success"] for r in rows[c[1]["case"]].values()),
                sum(r["success_strict"] for r in rows[c[1]["case"]].values()),
                -max(r.get("judge", {}).get("obb_mm", 0.0) for r in rows[c[1]["case"]].values()),
            )
        )
    methods = sorted({m for r in rows.values() for m in r})
    server = viser.ViserServer()
    robots = {  # finger opening per split, as the benchmark locks it
        (role, split): ViserUrdf(
            server,
            load_panda_urdf_fixed(opening),
            root_node_name=f"/{split}/{role}",
            mesh_color_override=COLORS.get(role),
        )
        for role in ("start", "goal", *methods)
        for split, opening in (("motion_benchmaker", 0.04), ("mpinets", 0.025))
    }
    case = server.gui.add_slider("Case", min=0, max=len(cases) - 1, step=1, initial_value=0)
    waypoint = server.gui.add_slider("Waypoint", min=0, max=1, step=1, initial_value=0)
    play = server.gui.add_checkbox("Play", initial_value=True)
    waypoint.visible = play.visible = bool(rows)
    info = server.gui.add_markdown("")
    obstacles = []

    @waypoint.on_update
    def step(_=None) -> None:
        split, problem = cases[case.value]
        for method, row in rows.get(problem["case"], {}).items():
            if "q" in row:
                robots[method, split].update_cfg(row["q"][min(waypoint.value, len(row["q"]) - 1)])

    @case.on_update
    def show(_=None) -> None:
        split, problem = cases[case.value]
        outs = rows.get(problem["case"], {})
        visible = {
            "start": not outs,
            "goal": bool(problem.get("goal_ik")),
            **{m: "q" in outs.get(m, {}) for m in methods},
        }
        for (role, s), robot in robots.items():
            robot.show_visual = s == split and visible[role]
        robots["start", split].update_cfg(problem["start"])
        if visible["goal"]:
            robots["goal", split].update_cfg(problem["goal_ik"][0])
        pose = problem["goal_pose"]
        server.scene.add_frame(
            "/goal_pose", wxyz=pose["quaternion_wxyz"], position=pose["position_xyz"], axes_length=0.1
        )
        while obstacles:
            obstacles.pop().remove()
        shapes = [obb_to_trimesh(o["dims"], o["pose"]) for o in problem["obstacles"].get("cuboid", {}).values()]
        shapes += [
            cylinder_to_trimesh(o["radius"], o["height"], o["pose"])
            for o in problem["obstacles"].get("cylinder", {}).values()
        ]
        shapes += [mesh_to_trimesh(o) for o in problem["obstacles"].get("mesh", {}).values()]
        obstacles.extend(server.scene.add_mesh_trimesh(f"/obstacles/{i}", s) for i, s in enumerate(shapes))
        lines = [f"`{problem['case']}` · {len(shapes)} obstacles · {len(problem.get('goal_ik') or [])} goal IK"]
        for method, r in outs.items():
            j = r.get("judge")
            verdict = "pass" if r["success"] else "FAIL"
            if j is None:
                lines.append(f"**{method}** {verdict}: no plan · {r['time'] * 1e3:.0f} ms")
                continue
            lines.append(
                f"**{method}** {verdict} (strict {'pass' if r['success_strict'] else 'FAIL'}): "
                f"reach {j['pos_err_mm']:.1f} mm / {j['rot_err']:.3f} · world {j['obb_mm']:.1f} mm · "
                f"self {'HIT' if j['self_collision'] else 'ok'} · limits +{j['limit_violation_rad']:.4f} rad · "
                f"dynamics {'ok' if j['dynamics_ok'] else 'BAD'} · {r['time'] * 1e3:.0f} ms"
            )
        info.content = "\n\n".join(lines)
        waypoint.max = max([len(r["q"]) - 1 for r in outs.values() if "q" in r] + [1])
        waypoint.value = 0
        step()

    show()
    while True:
        if play.value and rows:
            waypoint.value = (waypoint.value + 1) % (waypoint.max + 1)
        time.sleep(0.1)


if __name__ == "__main__":
    tyro.cli(main)
