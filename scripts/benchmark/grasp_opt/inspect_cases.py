# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "graspkit[maniskill]==0.1.0",
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "torchvision",
#   "scipy",
#   "tabulate",
#   "tyro",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv]
# override-dependencies = ["robokit[pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# torchvision = { index = "pytorch-cu126" }
# ///
"""Inspect each method's first saved grasp per scene, worst success first, no solving; `--outputs PATH` picks the run.

Displays:
- Scene: target object and the selected method's grasp.
- Scores: planning/lifting success, Q1, and penetration.

    uv run scripts/benchmark/grasp_opt/inspect_cases.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import transforms3d
import trimesh
import tyro
import viser
from graspkit.assets import REVISION
from graspkit.configs.robot.mano import mano_right
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID, OBJECT_COLOR, ROBOT_COLOR
from graspkit.scene.scene_description import SceneDescription
from viser.extras import ViserUrdf

from robokit.assets import fetch


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.benchmark.grasp_opt.run_benchmark import OUTPUTS, SCENES  # noqa: E402


def main(outputs: Path = OUTPUTS, port: int = 8080) -> None:
    solved = {p.stem: dict(np.load(p)) for p in sorted(outputs.glob("*.npz"))}
    if not solved:
        raise SystemExit(f"no grasps in {outputs}; run run_benchmark.py first")
    os.chdir(GRASPKIT_HOME)  # scene files name meshes relative to graspkit's data root
    all_scenes = SceneDescription.load_scene_descriptions_msgpack(
        str(fetch([SCENES], REVISION, HF_REPO_ID, GRASPKIT_HOME) / SCENES)
    )
    scene_ids = list(next(iter(solved.values()))["scene_ids"])
    order = np.argsort(np.mean([s["success"].mean(1) for s in solved.values()], 0), kind="stable")
    server = viser.ViserServer(port=port)
    hand = server.scene.add_frame("/hand", show_axes=False)
    urdf = ViserUrdf(
        server, Path(mano_right.urdf_or_mjcf_path), root_node_name="/hand", mesh_color_override=ROBOT_COLOR
    )
    method = server.gui.add_dropdown("Method", tuple(solved))
    scene = server.gui.add_slider("Scene", min=0, max=len(scene_ids) - 1, step=1, initial_value=0)
    grasp = server.gui.add_slider(
        "Grasp", min=0, max=solved[method.value]["qpos"].shape[1] - 1, step=1, initial_value=0
    )
    info = server.gui.add_markdown("")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:  # hand-scale view of the object, which sits near the origin
        client.camera.position, client.camera.look_at = (0.4, 0.4, 0.4), (0.0, 0.0, 0.05)

    @scene.on_update
    def load(_=None) -> None:
        desc = all_scenes[scene_ids[order[scene.value]]]
        m = desc.meshes[desc.target_actor_id]
        mesh = trimesh.load(m.collision_mesh_path, force="mesh").apply_scale(m.scales)
        server.scene.add_mesh_simple(
            "/object", mesh.vertices, mesh.faces, color=OBJECT_COLOR, position=m.position, wxyz=m.quaternion
        )
        show()

    @method.on_update
    @grasp.on_update
    def show(_=None) -> None:
        s, i, g = solved[method.value], order[scene.value], grasp.value
        q, ok = s["qpos"][i, g], bool(s["success"][i, g])  # base xyz + rxyz euler, then the fingers
        hand.position, hand.wxyz = q[:3], transforms3d.euler.euler2quat(*q[3:6], axes="rxyz")
        urdf.update_cfg(q[6:])
        info.content = "\n\n".join(
            [f"`{scene_ids[i]}`"]
            + [f"**{name}**: {100 * r['success'][i].mean():.0f}% success" for name, r in solved.items()]
            + [
                f"**Grasp {g}**: {'success' if ok else 'failure'} · Q1 {s['q1'][i, g]:.3f} · pen {s['pen_mm'][i, g]:.2f} mm"
            ]
        )

    load()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
