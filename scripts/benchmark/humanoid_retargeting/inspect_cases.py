"""Replay saved SFU humanoid-retargeting results, worst position error first; no retargeter runs.

Displays:
- Human: the source SMPL-X body.
- Robots: saved G1 motion and scores, with RoboKit in blue and GMR in orange.

    uv run python scripts/benchmark/humanoid_retargeting/inspect_cases.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from robokit.assets import fetch  # noqa: E402
from robokit.assets.robots.humanoids import unitree_g1  # noqa: E402
from robokit.helpers.humanoid_retargeting.loaders import load_smplx_file  # noqa: E402
from robokit.robo import Robot  # noqa: E402


COLORS = {"RoboKit": (0.2, 0.4, 0.9, 0.7), "GMR": (0.95, 0.55, 0.1, 0.7)}
METRICS = ("mpjpe_m", "wa_mpjpe_m", "limit_violation_ratio", "time_ms")


def main() -> None:
    solved: dict = {}  # sequence id -> {method: saved npz}
    for path in sorted((Path(__file__).parent / "outputs").glob("*.npz")):
        sequence_id, method = path.name.removesuffix(".npz").split(".")
        solved.setdefault(sequence_id, {})[method] = dict(np.load(path))
    paths = {p.stem: p for p in sorted((fetch(["datasets/sfu/**"]) / "datasets/sfu").rglob("*.npz"))}
    sequences = sorted(  # `run_benchmark.py` names each sequence after its file stem
        (s for s in paths if s in solved),
        key=lambda s: -max(json.loads(str(o["metrics"]))["mpjpe_m"] for o in solved[s].values()),
    ) or sorted(paths)
    methods = sorted({m for s in solved.values() for m in s})
    names = Robot.load(str(unitree_g1.URDF_PATH)).spec.actuated_joint_names
    server = viser.ViserServer()
    bases = {m: server.scene.add_frame(f"/{m}", show_axes=False) for m in methods}
    robots = {
        m: ViserUrdf(server, Path(unitree_g1.URDF_PATH), root_node_name=f"/{m}", mesh_color_override=COLORS.get(m))
        for m in methods
    }
    order = {m: [names.index(n) for n in robot.get_actuated_joint_names()] for m, robot in robots.items()}
    sequence = server.gui.add_slider("Sequence", min=0, max=max(len(sequences) - 1, 1), step=1, initial_value=0)
    frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    play = server.gui.add_checkbox("Play", initial_value=bool(solved))
    info = server.gui.add_markdown("")
    data: dict = {}

    @sequence.on_update
    def load(_=None) -> None:
        sequence_id = sequences[min(sequence.value, len(sequences) - 1)]  # a slider needs two stops
        motion = load_smplx_file(str(paths[sequence_id]))
        source_fps = motion.mocap_frame_rate
        step = int(source_fps // 30) if source_fps > 30 and source_fps % 30 == 0 else 1  # `get_smplx_motion` resamples
        outs = solved.get(sequence_id, {})
        vertices = motion.vertices[::step]
        num_frames = min([len(vertices)] + [len(s["qpos"]) for s in outs.values()])
        data["clip"] = (vertices, motion.faces, outs, num_frames)  # one write: `show` never sees a half-swap
        frame.max = num_frames - 1
        frame.value = 0
        lines = [f"`{sequence_id}` · {len(vertices)} frames at {source_fps / step:.0f} fps"]
        for method, s in outs.items():
            scores = json.loads(str(s["metrics"]))
            lines.append(f"**{method}** " + " · ".join(f"{k} {scores[k]:.4g}" for k in METRICS))
        info.content = "\n\n".join(lines)
        show()

    @frame.on_update
    def show(_=None) -> None:
        vertices, faces, outs, num_frames = data["clip"]
        t = min(frame.value, num_frames - 1)  # the play loop may still hold the previous sequence's frame
        server.scene.add_mesh_simple("/human", vertices[t], faces, color=(0.9, 0.75, 0.6), opacity=0.6)
        for method, robot in robots.items():
            s = outs.get(method)
            robot.show_visual = s is not None
            if s is not None:
                q = s["qpos"][t]  # base xyz + wxyz, then the joints
                bases[method].position, bases[method].wxyz = q[:3], q[3:7]
                robot.update_cfg(q[7:][order[method]])

    load()
    while True:
        if play.value:
            frame.value = (frame.value + 1) % (frame.max + 1)
        time.sleep(1 / 30)


if __name__ == "__main__":
    tyro.cli(main)
