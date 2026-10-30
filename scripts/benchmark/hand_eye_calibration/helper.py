"""HEC benchmark helpers: EasyHEC and RoboKit implementations."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HECBenchmarkOutput:
    time_s: float
    trans_err_cm: float
    rot_err_deg: float
    predicted: np.ndarray  # the solved 4x4 extrinsic


def compute_pose_error(predicted: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """Returns (trans_err_cm, rot_err_deg)."""
    t_err = (gt[:3, 3] - predicted[:3, 3]).norm().item() * 100
    R_rel = gt[:3, :3] @ predicted[:3, :3].t()
    r_err = torch.acos(((R_rel.trace() - 1) / 2).clamp(-1, 1)).item() * 180 / np.pi
    return t_err, r_err


class HECBenchmarkHelper(ABC):
    @abstractmethod
    def setup(self, data: dict): ...

    @abstractmethod
    def optimize(
        self,
        initial_extrinsic: torch.Tensor,
        iterations: int,
        lr: float,
        early_stopping_steps: int,
    ) -> HECBenchmarkOutput: ...


class EasyHandEyeCalibrator(HECBenchmarkHelper):
    def setup(self, data: dict):
        self.mesh_paths = data["mesh_paths"]
        self.link_poses = data["link_poses"]
        self.masks = data["masks"]
        self.intrinsic = data["intrinsic"]
        self.gt_extrinsic = data["gt_extrinsic"]
        self.H = data["H"]
        self.W = data["W"]
        self.camera_mount_poses = data.get("camera_mount_poses")
        self.optimize(self.gt_extrinsic.clone(), 10, 3e-3, 200)  # untimed warm solve, as RoboKit gets

    def optimize(self, initial_extrinsic, iterations, lr, early_stopping_steps) -> HECBenchmarkOutput:
        from easyhec.optim.optimize import optimize

        t0 = time.perf_counter()
        predicted = optimize(
            initial_extrinsic_guess=initial_extrinsic,
            camera_intrinsic=self.intrinsic,
            masks=self.masks,
            link_poses_dataset=self.link_poses,
            meshes=self.mesh_paths,
            camera_width=self.W,
            camera_height=self.H,
            camera_mount_poses=self.camera_mount_poses,
            iterations=iterations,
            learning_rate=lr,
            early_stopping_steps=early_stopping_steps,
            verbose=False,  # no progress bar or per-iteration ground-truth metrics inside the timer
        )
        elapsed = time.perf_counter() - t0
        t_err, r_err = compute_pose_error(predicted, self.gt_extrinsic)
        return HECBenchmarkOutput(elapsed, t_err, r_err, predicted.detach().cpu().numpy())


class RoboKitHelper(HECBenchmarkHelper):
    """RoboKit's HandEyeCalibrator preset; `modality` picks residuals: "rgb"=mask, "rgbd"=+depth, "depth"=all."""

    modality = "rgb"
    preset = ""

    def setup(self, data: dict):
        from robokit.robo import Robot

        self.depths = data.get("depths")
        self.masks = data["masks"]
        self.intrinsic = data["intrinsic"]
        self.gt_extrinsic = data["gt_extrinsic"]
        self.H = data["H"]
        self.W = data["W"]
        self.robot = Robot.load(data["urdf_path"], load_meshes=True)
        names = data["active_joint_names"]
        q = data["qpos"][:, [names.index(n) for n in self.robot.spec.actuated_joint_names]]
        self.q = torch.from_numpy(np.float32(q)).cuda()
        # Calibrate in world or mount frame; fold T_world_base and inverse mount into T_mount_base.
        T_world_base = torch.from_numpy(np.float32(data["T_world_base"])).cuda()
        T_mount_world = data.get("camera_mount_poses")
        if T_mount_world is not None:
            self.T_mount_base = T_mount_world @ T_world_base
        else:
            self.T_mount_base = T_world_base.expand(self.q.shape[0], 4, 4)
        from robokit.helpers.hand_eye_calibration import HandEyeCalibrator, presets

        config = getattr(presets, self.preset or self.modality)
        self._hec = HandEyeCalibrator(
            camera_intrinsic=self.intrinsic,
            height=self.H,
            width=self.W,
            robot=self.robot,
            config=config,
        )
        # untimed warm solve: builds tasks + CUDA graphs, pays nvdiffrast JIT
        self._hec.solve(
            torch.eye(4, device=self.q.device),
            target_masks=self.masks if self.modality != "depth" else None,
            target_depths=self.depths if self.modality != "rgb" else None,
            q=self.q,
            T_mount_base=self.T_mount_base,
        )

    def optimize(self, initial_extrinsic, iterations, lr, early_stopping_steps) -> HECBenchmarkOutput:
        t0 = time.perf_counter()
        predicted = self._hec.solve(initial_extrinsic)
        elapsed = time.perf_counter() - t0
        t_err, r_err = compute_pose_error(predicted, self.gt_extrinsic)
        return HECBenchmarkOutput(elapsed, t_err, r_err, predicted.detach().cpu().numpy())


class RoboKitRGBDHelper(RoboKitHelper):
    modality = "rgbd"


class RoboKitDepthHelper(RoboKitHelper):
    modality = "depth"


class RoboKitSingleLevelHelper(RoboKitHelper):
    preset = "rgb_single_level"
