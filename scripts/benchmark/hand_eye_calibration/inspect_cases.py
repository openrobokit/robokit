# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "torch==2.9.1",
#   "nvdiffrast==0.3.3+pt2.9.1cu126",
#   "easyhec",
#   "mani-skill",
#   "scipy",
#   "tabulate",
#   "tyro",
#   "transforms3d",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [[tool.uv.index]]
# name = "torch-packages-builder"
# url = "https://jianglongye.github.io/torch_packages_builder"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# torch = { index = "pytorch-cu126" }
# nvdiffrast = { index = "torch-packages-builder" }
# ///
"""Inspect saved hand-eye benchmark samples without running a solver.

Displays:
- Scene: robot URDF and all cameras in 3D.
- Image: robot rendered from the selected camera, colored to match the camera legend.
- Calibration: ground-truth overlay and each saved solution's overlay and error; accurate overlays cover the robot.

    uv run scripts/benchmark/hand_eye_calibration/inspect_cases.py panda-easy
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from viser.extras import ViserUrdf


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from robokit.helpers.hand_eye_calibration import compute_robot_masks  # noqa: E402
from robokit.robo import Robot  # noqa: E402
from scripts.benchmark.hand_eye_calibration.run_benchmark import CONFIGS, generate_maniskill_data  # noqa: E402


COLORS = {  # name, RGB
    "Ground truth": ("green", (0, 200, 0)),
    "RoboKit": ("blue", (40, 90, 255)),
    "RoboKit-RGBD": ("cyan", (0, 200, 255)),
    "RoboKit-Depth": ("purple", (150, 60, 220)),
    "EasyHEC": ("orange", (255, 140, 0)),
}


def main() -> None:
    cfg = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS))
    data = generate_maniskill_data(
        cfg.env_id,
        cfg.camera_name,
        cfg.samples,
        cfg.seed,
        mask_dilation_px=cfg.mask_dilation_px,
        mask_dropout_frac=cfg.mask_dropout_frac,
        joint_noise_std=cfg.joint_noise_std,
        joint_bias_std=cfg.joint_bias_std,
        intrinsic_noise=cfg.intrinsic_noise,
        depth_noise_std=cfg.depth_noise_std,
        depth_dropout_frac=cfg.depth_dropout_frac,
    )
    preset = next((k for k, v in CONFIGS.items() if v == cfg), "custom")
    solved = {p.stem: np.load(p) for p in sorted((Path(__file__).parent / "outputs" / preset).glob("*.npz"))}
    # the robot as RoboKitHelper.setup models it, rendered under each camera (every preset solves once: [0])
    robot = Robot.load(data["urdf_path"], load_meshes=True)
    names = data["active_joint_names"]
    q = torch.from_numpy(np.float32(data["qpos"][:, [names.index(n) for n in robot.spec.actuated_joint_names]])).cuda()
    T_world_base = torch.from_numpy(np.float32(data["T_world_base"])).cuda()
    mounts = data["camera_mount_poses"]
    T_mount_base = mounts @ T_world_base if mounts is not None else T_world_base.expand(len(q), 4, 4)
    cameras = {"Ground truth": data["gt_extrinsic"]}
    cameras |= {name: torch.from_numpy(s["predicted"][0]).cuda() for name, s in solved.items()}
    rendered = {
        label: compute_robot_masks(robot, q, data["intrinsic"], T, data["H"], data["W"], T_mount_base=T_mount_base)
        .cpu()
        .numpy()
        > 0.5
        for label, T in cameras.items()
    }
    server = viser.ViserServer()
    server.gui.configure_theme(control_width="large")
    slider = server.gui.add_slider("Sample", min=0, max=cfg.samples - 1, step=1, initial_value=0)
    server.gui.add_markdown(
        "\n\n".join(
            [f"**Ground truth**: {COLORS['Ground truth'][0]}"]
            + [
                f"**{name}**: {COLORS[name][0]} · {s['trans_err_cm'][0]:.3f} cm · {s['rot_err_deg'][0]:.3f} deg"
                for name, s in solved.items()
            ]
        )
    )
    camera = server.gui.add_dropdown("Camera", tuple(cameras))
    image = server.gui.add_image(np.zeros((data["H"], data["W"], 3), dtype=np.uint8), label="Robot under this camera")
    # 3D: the robot in its base frame, and every camera where it puts the robot's base
    urdf = ViserUrdf(server, Path(data["urdf_path"]))
    order = [names.index(n) for n in urdf.get_actuated_joint_names()]
    fov = 2 * np.arctan2(data["H"] / 2, float(data["intrinsic"][1, 1]))
    frustums = {
        label: server.scene.add_camera_frustum(
            f"/cameras/{label}", fov=fov, aspect=data["W"] / data["H"], scale=0.1, color=COLORS[label][1]
        )
        for label in cameras
    }

    @slider.on_update
    @camera.on_update
    def show(_=None) -> None:
        i = slider.value
        urdf.update_cfg(data["qpos"][i][order])
        for label, T in cameras.items():
            T_base_camera = np.linalg.inv((T @ T_mount_base[i]).cpu().numpy())
            frustums[label].position = T_base_camera[:3, 3]
            frustums[label].wxyz = viser.transforms.SO3.from_matrix(T_base_camera[:3, :3]).wxyz
        rgb = data["rgb"][i][..., :3].astype(np.float32)
        m = rendered[camera.value][i]
        rgb[m] = 0.5 * rgb[m] + 0.5 * np.array(COLORS[camera.value][1])
        image.image = rgb.astype(np.uint8)

    show()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
