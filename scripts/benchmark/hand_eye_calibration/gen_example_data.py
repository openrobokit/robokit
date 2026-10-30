# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "torch==2.9.1",
#   "nvdiffrast==0.3.3+pt2.9.1cu126",
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
"""Bake ManiSkill HEC datasets: stores `q` (robokit order) so examples FK via `HandEyeCalibrator`, no ManiSkill needed.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/benchmark/hand_eye_calibration/gen_example_data.py
"""

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from robokit.robo import Robot  # noqa: E402
from scripts.benchmark.hand_eye_calibration.run_benchmark import generate_maniskill_data, perturb_pose  # noqa: E402


OUT_DIR = Path.home() / ".robokit" / "hf_repo" / "calibration"

# 20 samples each: examples use the first 10 to solve, the rest for overlays.
SCENARIOS = {
    "maniskill_stackcube": dict(env_id="StackCube-v1", camera_name="human_render_camera"),
    "maniskill_so100_mounted": dict(env_id="SO100GraspCube-v1", camera_name="base_camera"),
}


def main():
    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, kwargs in SCENARIOS.items():
        data = generate_maniskill_data(num_samples=20, seed=0, **kwargs)

        # Copy the robot description next to the npz so mesh resolution is validated against what actually ships.
        urdf_path = Path(data["urdf_path"])
        robot_dir = OUT_DIR / "robots" / urdf_path.parent.name
        if not robot_dir.exists():
            # only the urdf + meshes it references (full ManiSkill robot dir has other variants + hulls, 60+ MB).
            for rel in {urdf_path.name, *re.findall(r'filename="([^"]+)"', urdf_path.read_text())}:
                dst = robot_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(urdf_path.parent / rel, dst)
        robot = Robot.load(str(robot_dir / urdf_path.name), load_meshes=True)

        # sapien qpos -> robokit actuated-joint order
        sapien_names = data["active_joint_names"]
        q = data["qpos"][:, [sapien_names.index(n) for n in robot.spec.actuated_joint_names]].astype(np.float32)

        # FK sanity: T_world_base @ robokit FK must match sapien world link poses.
        T_base_link = robot.forward_kinematics_via_matrix_torch(torch.from_numpy(q).cuda()).cpu().numpy()
        T_world_base = data["T_world_base"]
        shared = [
            (i, data["link_names"].index(n)) for i, n in enumerate(robot.spec.link_names) if n in data["link_names"]
        ]
        fk_world = np.einsum("ij,nljk->nlik", T_world_base, T_base_link[:, [i for i, _ in shared]].astype(np.float64))
        err = np.abs(fk_world - data["link_world_poses"][:, [j for _, j in shared]]).max()
        assert err < 2e-3, f"{name}: robokit FK vs sapien link poses mismatch, max err {err:.2e}"

        gt = data["gt_extrinsic"].cpu().numpy().astype(np.float64)
        arrays = dict(
            rgb=data["rgb"],
            masks=data["masks"].cpu().numpy().astype(np.uint8),
            depths=data["depths"].cpu().numpy(),
            q=q,
            intrinsic=data["intrinsic"].cpu().numpy(),
            urdf=f"calibration/robots/{urdf_path.parent.name}/{urdf_path.name}",
        )
        mount = data["camera_mount_poses"]
        if mount is not None:
            # gt (camera local pose) is already mount-relative; store per-sample T_mount_base.
            arrays["T_mount_base"] = (mount.cpu().numpy().astype(np.float64) @ T_world_base).astype(np.float32)
        else:
            gt = gt @ T_world_base  # world-frame extrinsic -> base-frame
        arrays["gt_extrinsic"] = gt.astype(np.float32)
        initial = perturb_pose(torch.from_numpy(gt).float(), 0.1, 15.0, np.random.default_rng(0))
        arrays["initial_extrinsic"] = initial.cpu().numpy()

        out = OUT_DIR / f"{name}.npz"
        np.savez_compressed(out, **arrays)
        print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, masks {arrays['masks'].shape}, fk err {err:.2e}")


if __name__ == "__main__":
    main()
