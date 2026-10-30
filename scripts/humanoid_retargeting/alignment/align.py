"""Fit humanoid scales and link offsets interactively with fast closed-form least squares.

Session:
1. Pick a pose; missing robot poses are seeded by IK.
2. Match it with joint sliders, "IK init", or "Mirror".
3. Read the match: target spheres and residual lines colored green (matched) to red
  (off), a per-pose and a per-link error table, and limb-direction warnings that expose a
  mismatched pair (the one error scales/offsets cannot absorb),
4. Save both Python alignment presets.

Usage (from repo root; open the printed URL / SSH-forward the port):

    uv run scripts/humanoid_retargeting/alignment/align.py --robot g1
    uv run scripts/humanoid_retargeting/alignment/align.py --robot gr3 --headless
"""

import argparse
import dataclasses
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import warp as wp
import yourdfpy

from robokit.assets.body_models import get_body_models_dir
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnlineConfig
from robokit.helpers.humanoid_retargeting import presets as presets_pkg
from robokit.helpers.humanoid_retargeting.alignment import (
    AlignmentPose,
    HumanPose,
    PosePairs,
    align_config,
    bone_audit,
    compute_targets,
    emit_alignment_preset,
    emit_preset,
    human_pose,
    ik_seed_pairs,
    measure_height,
    pose_error_mm,
    pose_pairs,
    prepare_pose,
    solve_alignment,
)
from robokit.helpers.humanoid_retargeting.presets import alignment as alignment_presets_pkg
from robokit.helpers.humanoid_retargeting.presets.adam_lite import adam_lite
from robokit.helpers.humanoid_retargeting.presets.alignment.adam_lite import adam_lite as adam_lite_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.bhl import bhl as bhl_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.g1 import g1 as g1_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.gr3 import gr3 as gr3_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.h1 import h1 as h1_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.h1_2 import h1_2 as h1_2_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.k1 import k1 as k1_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.pm01 import pm01 as pm01_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.t1 import t1 as t1_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.talos import talos as talos_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.tienkung import tienkung as tienkung_alignment
from robokit.helpers.humanoid_retargeting.presets.alignment.toddy import toddy as toddy_alignment
from robokit.helpers.humanoid_retargeting.presets.bhl import bhl
from robokit.helpers.humanoid_retargeting.presets.g1 import g1
from robokit.helpers.humanoid_retargeting.presets.gr3 import gr3
from robokit.helpers.humanoid_retargeting.presets.h1 import h1
from robokit.helpers.humanoid_retargeting.presets.h1_2 import h1_2
from robokit.helpers.humanoid_retargeting.presets.k1 import k1
from robokit.helpers.humanoid_retargeting.presets.pm01 import pm01
from robokit.helpers.humanoid_retargeting.presets.t1 import t1
from robokit.helpers.humanoid_retargeting.presets.talos import talos
from robokit.helpers.humanoid_retargeting.presets.tienkung import tienkung
from robokit.helpers.humanoid_retargeting.presets.toddy import toddy
from robokit.robo import Robot


PRESETS = {
    "g1": g1,
    "gr3": gr3,
    "h1_2": h1_2,
    "bhl": bhl,
    "adam_lite": adam_lite,
    "h1": h1,
    "k1": k1,
    "pm01": pm01,
    "t1": t1,
    "talos": talos,
    "tienkung": tienkung,
    "toddy": toddy,
}
ALIGNMENT_PRESETS = {
    "g1": g1_alignment,
    "gr3": gr3_alignment,
    "h1_2": h1_2_alignment,
    "bhl": bhl_alignment,
    "adam_lite": adam_lite_alignment,
    "h1": h1_alignment,
    "k1": k1_alignment,
    "pm01": pm01_alignment,
    "t1": t1_alignment,
    "talos": talos_alignment,
    "tienkung": tienkung_alignment,
    "toddy": toddy_alignment,
}
ALIGNMENT_PRESETS_DIR = Path(alignment_presets_pkg.__file__).resolve().parent
PRESETS_DIR = Path(presets_pkg.__file__).resolve().parent

# camera presets as (forward, lateral, up) offsets from the body center, in body heights
VIEWS = {"Front": (2.1, 0.0, 0.4), "3/4": (1.6, 1.3, 0.55), "Side": (0.0, 2.1, 0.4), "Top": (0.7, 0.0, 2.1)}
ERR_CAP = 0.06  # residual (m) drawn fully red; below it fades to green
# slider folders: robots name joints differently, so group by keyword rather than by index
ARM_KEYS = ("shoulder", "elbow", "wrist", "arm", "hand")
LEG_KEYS = ("hip", "knee", "ankle", "foot", "toe", "thigh", "shank", "leg")


def err_color(err: float) -> Tuple[int, int, int]:
    """Green (matched) to red (``ERR_CAP`` or worse) for a residual in meters."""
    t = min(err / ERR_CAP, 1.0)
    return (int(60 + 170 * t), int(180 - 120 * t), int(80 - 35 * t))


def joint_group(joint: str) -> str:
    """Body-part slider folder for a joint name."""
    low = joint.lower()
    side = "Left " if "left" in low else "Right " if "right" in low else ""
    limb = "arm" if any(k in low for k in ARM_KEYS) else "leg" if any(k in low for k in LEG_KEYS) else "torso"
    return f"{side}{limb}" if side else limb.capitalize()


def print_report(report: List[Tuple[str, float, float]], config: HumanoidRetargeterOnlineConfig) -> None:
    print(f"\n  {'pose':<14s} {'before (mm)':>12s} {'after (mm)':>12s}")
    for label, before, after in report:
        print(f"  {label:<14s} {before:>12.0f} {after:>12.0f}")
    means = np.mean([[b, a] for _, b, a in report], axis=0)
    print(f"  {'mean':<14s} {means[0]:>12.0f} {means[1]:>12.0f}")
    print("\n  scale_table:")
    for name, val in config.scale_table.items():
        print(f"    {name}: {val:.4f}")
    print(f"  human_height_assumption: {config.human_height_assumption}")


def save_all(
    robot_name: str, pairs: PosePairs, config: HumanoidRetargeterOnlineConfig, alignment_preset_path: Path
) -> None:
    alignment_preset_path.write_text(emit_alignment_preset(robot_name, pairs))
    preset_path = PRESETS_DIR / f"{robot_name}_aligned.py"
    preset_path.write_text(emit_preset(robot_name, config))
    print(f"Saved pose pairs: {alignment_preset_path}")
    print(f"Wrote preset:     {preset_path}  (register '{robot_name}_aligned' where presets are listed)")


def edit(
    robot_name: str,
    config: HumanoidRetargeterOnlineConfig,
    robot: Robot,
    pairs: PosePairs,
    body_model_path: str,
    alignment_preset_path: Path,
    port: int,
    device: str,
) -> None:
    """Viser editor: every edit re-grounds the current pose and re-solves the alignment live."""
    import viser
    from viser.extras import ViserUrdf

    pose_names = list(pairs)
    humans: Dict[str, HumanPose] = {n: human_pose(body_model_path, bp) for n, (bp, _) in pairs.items()}
    prepared: Dict[str, AlignmentPose] = {n: prepare_pose(config, robot, humans[n], pairs[n][1], n) for n in pose_names}
    height = measure_height(body_model_path)
    position_weights, _ = config.task("target").weights_for(list(config.link_mapping))
    tracked = [rl for rl, weight in zip(config.link_mapping, position_weights) if weight > 0]
    human_of = {rl: m.human_joint for rl, m in config.link_mapping.items()}
    joint_names = list(robot.spec.actuated_joint_names)
    limits = robot.spec.actuated_joint_limits
    mirrors = {j: j.replace("left", "right") for j in joint_names if "left" in j}
    mirrors = {src: dst for src, dst in mirrors.items() if dst in joint_names}
    ui = {"cur": pose_names[0], "loading": False, "saved": True}

    # camera basis from the body itself: lateral points to its left, forward = lateral x up
    ref = humans[pose_names[0]].positions
    lateral = ref["left_shoulder"] - ref["right_shoulder"]
    lateral = lateral / np.linalg.norm(lateral)
    forward = np.cross(lateral, (0.0, 0.0, 1.0))
    center = np.array([0.0, 0.0, 0.55 * height])

    # --- panel: persistent header + save, then tabs (poses / joints / view / fit) ---
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.add_grid("/ground", width=4, height=4)
    header = server.gui.add_markdown("")
    quality = server.gui.add_progress_bar(100.0, color="green")
    save_button = server.gui.add_button("Save poses + preset", color="green", icon=viser.Icon.DEVICE_FLOPPY)
    tabs = server.gui.add_tab_group()

    with tabs.add_tab("Poses", viser.Icon.WALK):
        dropdown = server.gui.add_dropdown("Pose", options=tuple(pose_names), initial_value=pose_names[0])
        prev_button = server.gui.add_button("Prev", icon=viser.Icon.ARROW_LEFT)
        next_button = server.gui.add_button("Next", icon=viser.Icon.ARROW_RIGHT)
        pose_table = server.gui.add_markdown("")
        audit_text = server.gui.add_markdown("")

    with tabs.add_tab("Joints", viser.Icon.ADJUSTMENTS):
        ik_button = server.gui.add_button("IK init this pose", icon=viser.Icon.WAND)
        ik_all_button = server.gui.add_button("IK init every pose")
        mirror_group = server.gui.add_button_group("Mirror", ("L to R", "R to L"))
        zero_button = server.gui.add_button("Zero all joints")
        sliders: Dict[str, viser.GuiInputHandle] = {}
        groups: Dict[str, List[str]] = {}
        for j in joint_names:
            groups.setdefault(joint_group(j), []).append(j)
        for group in sorted(groups, key=lambda g: (not g.endswith("arm"), g)):
            with server.gui.add_folder(group, expand_by_default=group.endswith("arm")):
                for j in groups[group]:
                    i = joint_names.index(j)
                    sliders[j] = server.gui.add_slider(
                        j.replace("_joint", ""),
                        min=float(limits[i, 0]),
                        max=float(limits[i, 1]),
                        step=1e-3,
                        initial_value=0.0,
                    )

    with tabs.add_tab("View", viser.Icon.EYE):
        view_group = server.gui.add_button_group("Camera", tuple(VIEWS))
        aligned_toggle = server.gui.add_checkbox("Aligned targets (off = raw preset)", initial_value=True)
        show_human = server.gui.add_checkbox("Human mesh", initial_value=True)
        show_robot = server.gui.add_checkbox("Robot", initial_value=True)
        xray = server.gui.add_checkbox("X-ray robot (see targets inside)", initial_value=False)
        show_targets = server.gui.add_checkbox("Targets + residuals", initial_value=True)
        server.gui.add_markdown(f"Residual color: green 0 mm to red {ERR_CAP * 1000:.0f} mm.")

    with tabs.add_tab("Fit", viser.Icon.CHART_BAR):
        link_table = server.gui.add_markdown("")
        scale_text = server.gui.add_markdown("")

    # --- scene: human mesh, robot, per-link targets (colored by residual) and residual lines ---
    human_mesh = server.scene.add_mesh_simple(
        "/human", np.zeros((3, 3), np.float32), np.array([[0, 1, 2]]), color=(90, 200, 255), opacity=0.4
    )
    base_frame = server.scene.add_frame("/robot", show_axes=False)
    # Show solid and translucent bodies; target residuals inside meshes need the x-ray view.
    urdf = yourdfpy.URDF.load(config.urdf_path)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_ghost = ViserUrdf(server, urdf, root_node_name="/robot", mesh_color_override=(0.62, 0.66, 0.72, 0.35))
    urdf_ghost.show_visual = False
    viser_joint_order = list(urdf_vis.get_actuated_joint_limits())
    spheres = [server.scene.add_icosphere(f"/tgt/{rl}", radius=0.022, color=(228, 74, 51)) for rl in tracked]
    dots = [server.scene.add_icosphere(f"/link/{rl}", radius=0.012, color=(70, 70, 80)) for rl in tracked]
    err_lines = server.scene.add_line_segments(
        "/err",
        points=np.zeros((len(tracked), 2, 3), np.float32),
        colors=np.zeros((len(tracked), 2, 3), np.uint8),
        line_width=4.0,
    )
    worst_label = server.scene.add_label("/worst", "")

    def set_sliders(values: Dict[str, float]) -> None:
        ui["loading"] = True
        for j, s in sliders.items():
            s.value = float(np.clip(values.get(j, 0.0), s.min, s.max))
        ui["loading"] = False

    def refresh(saved: bool = False) -> None:
        """Commit sliders -> re-ground this pose -> re-solve the alignment -> redraw + report."""
        cur = ui["cur"]
        ui["saved"] = saved
        pairs[cur] = (pairs[cur][0], {j: round(float(s.value), 4) for j, s in sliders.items() if abs(s.value) > 1e-4})
        prepared[cur] = prepare_pose(config, robot, humans[cur], pairs[cur][1], cur)
        aligned = dataclasses.replace(
            solve_alignment(config, list(prepared.values())), human_height_assumption=round(height, 4)
        )
        pose = prepared[cur]
        shown = aligned if aligned_toggle.value else config
        targets = compute_targets(shown, pose.human_positions, pose.human_quaternions)
        errs = {rl: float(np.linalg.norm(targets[rl] - pose.robot_positions[rl])) for rl in tracked}
        worst = max(errs, key=lambda rl: errs[rl])

        with server.atomic():
            human_mesh.vertices, human_mesh.faces = humans[cur].vertices, humans[cur].faces
            base_frame.position, base_frame.wxyz = pose.robot_base[:3], pose.robot_base[3:]
            cfg = np.array([sliders[j].value for j in viser_joint_order], dtype=np.float32)
            urdf_vis.update_cfg(cfg)
            urdf_ghost.update_cfg(cfg)
            pts = np.zeros((len(tracked), 2, 3), np.float32)
            cols = np.zeros((len(tracked), 2, 3), np.uint8)
            for i, rl in enumerate(tracked):
                spheres[i].position, spheres[i].color = tuple(targets[rl]), err_color(errs[rl])
                dots[i].position = tuple(pose.robot_positions[rl])
                pts[i, 0], pts[i, 1] = targets[rl], pose.robot_positions[rl]
                cols[i, 0] = cols[i, 1] = err_color(errs[rl])
            err_lines.points, err_lines.colors = pts, cols
            worst_label.position = tuple((targets[worst] + pose.robot_positions[worst]) / 2 + [0, 0, 0.06])
            worst_label.text = f"{human_of[worst]}  {errs[worst] * 1000:.0f} mm"

        # --- panel text: header, per-pose table, bone audit, per-link table, fitted scales ---
        mean_mm = pose_error_mm(aligned, pose)
        quality.value = float(np.clip(100.0 * (1.0 - mean_mm / 60.0), 0.0, 100.0))
        quality.color = "green" if quality.value > 66 else "yellow" if quality.value > 33 else "red"
        header.content = (
            f"### {robot_name} · {cur}\n\n**{mean_mm:.0f} mm** mean residual · "
            f"{'saved' if saved else 'unsaved changes'} · `{alignment_preset_path.name}`"
        )
        flagged = {n: [lab for lab, a, f in bone_audit(p, config) if a > f + 15.0] for n, p in prepared.items()}
        pose_table.content = "| pose | before | after (mm) |\n|---|---|---|\n" + "".join(
            f"| {'**' + n + '**' if n == cur else n}{' ⚠' if flagged[n] else ''} "
            f"| {pose_error_mm(config, p):.0f} | {pose_error_mm(aligned, p):.0f} |\n"
            for n, p in prepared.items()
        )
        audit_text.content = (
            "**Limb directions off - fix this pair before trusting the fit:** " + ", ".join(flagged[cur])
            if flagged[cur]
            else "Limb directions OK - human and robot are in the same configuration."
        )
        link_table.content = "| link | human joint | residual |\n|---|---|---|\n" + "".join(
            f"| {rl} | {human_of[rl]} | {errs[rl] * 1000:.0f} mm |\n"
            for rl in sorted(errs, key=lambda rl: -errs[rl])[:8]
        )
        scale_text.content = (
            f"**human_height_assumption** {aligned.human_height_assumption} m\n\n"
            "| human joint | scale |\n|---|---|\n"
            + "".join(f"| {k} | {v:.3f} |\n" for k, v in aligned.scale_table.items())
        )

    def select(name: str) -> None:
        ui["cur"] = name
        dropdown.value = name
        set_sliders(pairs[name][1])
        alignment_preset_path.write_text(emit_alignment_preset(robot_name, pairs))
        refresh(saved=True)

    def set_view(name: str, *clients: "viser.ClientHandle") -> None:
        """Frame the body from a named viewpoint; pass a new client explicitly since `get_clients()` stalls its own."""
        f, lat, up = VIEWS[name]
        eye = center + (forward * f + lateral * lat + np.array([0.0, 0.0, up])) * (0.62 * height)
        for client in clients or tuple(server.get_clients().values()):
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.position = tuple(eye)
            client.camera.look_at = tuple(center)

    def set_robot_style() -> None:
        urdf_vis.show_visual = show_robot.value and not xray.value
        urdf_ghost.show_visual = show_robot.value and xray.value

    def notify(title: str, body: str, color: str = "green") -> None:
        for client in server.get_clients().values():
            client.add_notification(title, body, auto_close_seconds=4.0, color=color)

    dropdown.on_update(lambda _e: None if ui["loading"] or dropdown.value == ui["cur"] else select(dropdown.value))
    prev_button.on_click(lambda _e: select(pose_names[(pose_names.index(ui["cur"]) - 1) % len(pose_names)]))
    next_button.on_click(lambda _e: select(pose_names[(pose_names.index(ui["cur"]) + 1) % len(pose_names)]))
    zero_button.on_click(lambda _e: (set_sliders({}), refresh()))
    view_group.on_click(lambda _e: set_view(view_group.value))
    aligned_toggle.on_update(lambda _e: refresh(ui["saved"]))
    show_human.on_update(lambda _e: setattr(human_mesh, "visible", show_human.value))
    show_robot.on_update(lambda _e: set_robot_style())
    xray.on_update(lambda _e: set_robot_style())
    server.on_client_connect(lambda c: set_view("3/4", c))
    for s in sliders.values():
        s.on_update(lambda _e: None if ui["loading"] else refresh())

    @show_targets.on_update
    def _(_event) -> None:
        for handle in [*spheres, *dots, err_lines, worst_label]:
            handle.visible = show_targets.value

    @mirror_group.on_click
    def _(_event) -> None:
        """Copy one side's joints to the other; sagittal joints keep sign, roll/yaw flip."""
        to_apply = mirrors if mirror_group.value == "L to R" else {d: s for s, d in mirrors.items()}
        values = {j: float(s.value) for j, s in sliders.items()}
        for src, dst in to_apply.items():
            values[dst] = -values[src] if ("roll" in dst or "yaw" in dst) else values[src]
        set_sliders(values)
        refresh()

    @ik_button.on_click
    def _(_event) -> None:
        seeded = ik_seed_pairs(config, robot, body_model_path, {ui["cur"]: (pairs[ui["cur"]][0], {})}, device=device)
        set_sliders(seeded[ui["cur"]][1])
        refresh()

    @ik_all_button.on_click
    def _(_event) -> None:
        notify("IK init", f"solving {len(pose_names)} poses...", color="blue")
        seeded = ik_seed_pairs(
            config, robot, body_model_path, {n: (pairs[n][0], {}) for n in pose_names}, device=device
        )
        pairs.update(seeded)
        for n in pose_names:
            prepared[n] = prepare_pose(config, robot, humans[n], pairs[n][1], n)
        set_sliders(pairs[ui["cur"]][1])
        refresh()
        notify("IK init", "all poses seeded - review and correct each one")

    @save_button.on_click
    def _(_event) -> None:
        aligned = dataclasses.replace(
            solve_alignment(config, list(prepared.values())), human_height_assumption=round(height, 4)
        )
        print_report([(n, pose_error_mm(config, p), pose_error_mm(aligned, p)) for n, p in prepared.items()], aligned)
        save_all(robot_name, pairs, aligned, alignment_preset_path)
        notify("Saved", f"{alignment_preset_path.name} + {robot_name}_aligned.py")
        refresh(saved=True)

    select(pose_names[0])
    # viser falls back to another port when `port` is taken, so report the one it actually uses
    print(f"\nViser running - open http://127.0.0.1:{server.get_port()}  (SSH-forward the port if remote)\n")
    server.sleep_forever()  # runs until Ctrl+C; edits persist on pose switch and Save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="g1", choices=sorted(PRESETS), help="Robot preset to align")
    parser.add_argument("--headless", action="store_true", help="No browser: IK-seed missing pairs, solve, save")
    parser.add_argument("--body-model-path", default=None, help="SMPL-X body model directory")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    config = PRESETS[args.robot]
    robot = Robot.load(config.urdf_path, load_meshes=True)
    body_model_path = args.body_model_path or str(get_body_models_dir())
    alignment_preset_path = ALIGNMENT_PRESETS_DIR / f"{args.robot}.py"
    pairs = pose_pairs(ALIGNMENT_PRESETS[args.robot])
    unseeded = {n: p for n, p in pairs.items() if not p[1]}
    if unseeded:
        pairs = {**pairs, **ik_seed_pairs(config, robot, body_model_path, unseeded, device=device)}
        alignment_preset_path.write_text(emit_alignment_preset(args.robot, pairs))
        print(f"IK-seeded {len(unseeded)} pose(s) -> {alignment_preset_path}")

    if args.headless:
        for name, (bp, ov) in pairs.items():
            pose = prepare_pose(config, robot, human_pose(body_model_path, bp), ov, name)
            bad = [f"{lab} {ang:.0f}deg" for lab, ang, floor in bone_audit(pose, config) if ang > floor + 15.0]
            if bad:
                print(f"  WARNING {name}: limb direction off ({', '.join(bad)}) - review in the editor")
        aligned, report = align_config(config, pairs, robot, body_model_path)
        print_report(report, aligned)
        save_all(args.robot, pairs, aligned, alignment_preset_path)
    else:
        edit(args.robot, config, robot, pairs, body_model_path, alignment_preset_path, args.port, device)


if __name__ == "__main__":
    main()
