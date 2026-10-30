# /// script
# dependencies = ["robokit", "tqdm"]
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# ///
"""Replay saved DexYCB hand-retargeting results without running solvers.

Displays:
- Human: 21 keypoints, pinch/open labels, and the thumb-index tip gap used to judge them.
- Robot: each method's saved `--hand` motion, fingertip error, and missed pinch/open frames.

    uv run scripts/benchmark/hand_retargeting_online/inspect_cases.py --hand shadow
"""

import pickle
import time
from pathlib import Path

import numpy as np
import tyro
import viser
from run_benchmark import resolve_hand_urdf
from viser.extras import ViserUrdf

from robokit.utils.visualize_utils import ViserHandSkeleton


def main(hand: str = "shadow") -> None:
    bundles = {"dexycb.pkl": pickle.loads((Path(__file__).parent / "outputs" / "dexycb.pkl").read_bytes())}
    solved: dict = {}  # (bundle file, method) -> saved npz
    for path in sorted((Path(__file__).parent / "outputs").glob(f"*.{hand}.*.npz")):
        stem, _, method = path.name.removesuffix(".npz").split(".", 2)
        solved[f"{stem}.pkl", method] = dict(np.load(path))
    methods = sorted({m for _, m in solved})
    server = viser.ViserServer()
    skeleton = ViserHandSkeleton(server)
    for i, m in enumerate(methods):  # each method's hand stands in its own lane beside the keypoints
        server.scene.add_frame(f"/{m}", position=(0.0, 0.25 * (i + 1), 0.0), show_axes=False)
        server.scene.add_label(f"/{m}/label", m, position=(0.0, 0.0, -0.05))
    bases = {m: server.scene.add_frame(f"/{m}/base", show_axes=False) for m in methods}  # a floating solve's base
    hands = {m: ViserUrdf(server, Path(resolve_hand_urdf(hand)), root_node_name=f"/{m}/base") for m in methods}
    bundle = server.gui.add_dropdown("Bundle", tuple(bundles))
    frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    play = server.gui.add_checkbox("Play", initial_value=bool(solved))
    info = server.gui.add_markdown("")
    misses = server.gui.add_markdown("")

    @bundle.on_update
    def load(_=None) -> None:
        b = bundles[bundle.value]
        frame.max, frame.value = len(b["keypoints"]) - 1, 0
        lines = []
        for m in methods:
            s = solved.get((bundle.value, m))
            if s is not None:
                pinch = [f for f, hit in zip(b["pinch_frames"], s["pinch_hits"]) if not hit]
                opened = [f for f, hit in zip(b["open_frames"], s["open_hits"]) if not hit]
                lines.append(f"**{m}** missed pinch {pinch} · missed open {opened}")
        misses.content = "\n\n".join(lines)
        show()

    @frame.on_update
    def show(_=None) -> None:
        b, t = bundles[bundle.value], frame.value
        keypoints = np.asarray(b["keypoints"][t], dtype=np.float32)
        skeleton.update(keypoints)
        label = "pinch" if t in b["pinch_frames"] else "open" if t in b["open_frames"] else "-"
        gap = np.linalg.norm(keypoints[4] - keypoints[8]) * 100
        lines = [f"`DexYCB` · {len(b['keypoints'])} frames · label **{label}** · tip gap {gap:.1f} cm"]
        for m, urdf in hands.items():
            s = solved.get((bundle.value, m))
            urdf.show_visual = s is not None
            if s is not None:
                if len(s["base_pose"]):
                    bases[m].position, bases[m].wxyz = s["base_pose"][t, :3], s["base_pose"][t, 3:]
                names = list(s["joint_names"])
                urdf.update_cfg(s["qpos"][t][[names.index(n) for n in urdf.get_actuated_joint_names()]])
                tips = len(s["pos_err_m"]) // len(s["qpos"])  # saved flat, frame-major
                lines.append(f"**{m}** tip error {s['pos_err_m'][t * tips : (t + 1) * tips].mean() * 100:.1f} cm")
        info.content = "\n\n".join(lines)

    load()
    server.initial_camera.look_at = (0.0, 0.25 * (len(methods) + 1) / 2, 0.0)
    server.initial_camera.position = (0.8, 0.25 * (len(methods) + 1) / 2, 0.4)
    while True:
        if play.value:
            frame.value = (frame.value + 1) % (frame.max + 1)
        time.sleep(1 / 30)


if __name__ == "__main__":
    tyro.cli(main)
