from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
from robot_descriptions import panda_description
from test_waypoints import build_planner  # pytest puts this directory on the path

from robokit.geom import BoxGeom, WarpScene
from robokit.geom.covering_spheres import _compute_covering_spheres
from robokit.helpers.motion_plan import AttachedObject, FixedRelativePose, MotionPlanner, presets
from robokit.robo import Robot
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.utils.assembly_utils import assemble_urdf, merge_collision_spheres
from robokit.xform.numpy.transforms import pose7_to_tf_mat


PANDA_IGNORE = Path(__file__).parents[2] / "fixtures/franka_self_collision_ignore.yml"


pytestmark = pytest.mark.torch
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"


def _translation(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _box_clearance(q_traj: np.ndarray, spheres: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    """Least clearance of the held spheres to an axis-aligned box along the plan, 20 samples a step."""
    weights = np.linspace(0, 1, 20, endpoint=False)[:, None]
    q = np.concatenate([(1 - weights) * a + weights * b for a, b in zip(q_traj[:-1], q_traj[1:])] + [q_traj[-1:]])
    points = np.c_[q, np.zeros(len(q))][:, None] + spheres[None, :, :3] - center
    outside = np.linalg.norm(np.maximum(np.abs(points) - half, 0), axis=-1)
    inside = np.minimum(np.max(np.abs(points) - half, axis=-1), 0)
    return float((outside + inside - spheres[None, :, 3]).min())


class TestAttachedObjects:
    def _planner(self, tmp_path: Path) -> MotionPlanner:
        spheres = tmp_path / "spheres.yaml"
        spheres.write_text("collision_spheres:\n  tool:\n    - center: [0, 0, 0]\n      radius: 0.05\n")
        poses = wp.array([_translation(0.0, 0.4, 0.0)], dtype=wp.mat44, device="cpu")
        scene = WarpScene(1, "cpu").add(BoxGeom(np.full((1, 3), 0.2), np.array([0, 1], np.int32), poses=poses))
        return build_planner(spheres=spheres, scene=scene, collision=1000.0, attached_spheres=32)

    box = AttachedObject("tool", "box", (0.05, 0.15, 0.05), _translation(0.0, 0.15, 0.0))

    def test_held_box_avoids_obstacle(self, tmp_path: Path):
        """The tool clears the obstacle on a straight line; the box it holds would not, so the held plan detours."""
        planner = self._planner(tmp_path)
        anchors = [np.array([-1.0, 0.0], np.float32), np.array([1.0, 0.0], np.float32)]
        free = planner.solve_offline_numpy(anchors).q_traj[0]
        carried = planner.solve_offline_numpy(anchors, attached_objects=[self.box]).q_traj[0]
        spheres = _compute_covering_spheres("box", self.box.size, self.box.T_link_object)
        center, half = np.array([0.0, 0.4, 0.0]), np.full(3, 0.2)
        free_clearance, carried_clearance = (_box_clearance(q, spheres, center, half) for q in (free, carried))
        print(f"held box clearance: {free_clearance:.4f} m ignored, {carried_clearance:.4f} m attached")
        assert free_clearance < -0.05 < 0.0 < carried_clearance
        assert _box_clearance(free, np.array([[0, 0, 0, 0.05]]), center, half) > 0.1  # the tool alone is clear

    def test_inactive_attachment_and_empty_lists_keep_the_plan(self, tmp_path: Path):
        planner = self._planner(tmp_path)
        anchors = [
            np.array([-1.0, 0.0], np.float32),
            np.array([0.0, -0.5], np.float32),
            np.array([1.0, 0.0], np.float32),
        ]
        base = planner.solve_offline_numpy(anchors).q_traj
        empty = planner.solve_offline_numpy(anchors, attached_objects=[], fixed_relative_poses=[]).q_traj
        torch.testing.assert_close(torch.from_numpy(empty), torch.from_numpy(base), rtol=0, atol=0)
        # held on the second segment only, which never nears the obstacle: the first segment plans as before
        late = planner.solve_offline_numpy(anchors, attached_objects=[replace(self.box, segments=(1,))])
        torch.testing.assert_close(torch.from_numpy(late.q_traj), torch.from_numpy(base), rtol=0, atol=2e-3)
        again = planner.solve_offline_numpy(anchors)  # its reserved spheres off again: the plan as first
        torch.testing.assert_close(torch.from_numpy(again.q_traj), torch.from_numpy(base), rtol=0, atol=0)

    def test_goal_ik_and_score_skip_an_attachment_outside_its_segments(self, tmp_path: Path):
        """The box is held on the first segment only: at the pose goal, where it would overlap the obstacle, neither goal
        IK nor the score counts it, so the tool reaches the goal and the plan ranks feasible."""
        planner = self._planner(tmp_path)
        goal = {"tool": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)}
        anchors = [np.array([-1.0, -0.5], np.float32), np.array([1.0, -0.5], np.float32), goal]
        result = planner.solve_offline_numpy(anchors, attached_objects=[replace(self.box, segments=(0,))])
        assert np.abs(result.q_traj[0, -1]).max() < 1e-3
        held_late = planner.solve_offline_numpy(anchors, attached_objects=[replace(self.box, segments=(1,))])  # moved
        assert np.abs(held_late.q_traj[0, -1]).max() > 0.01


class TestRelativeFrames:
    def test_two_arms_hold_their_relative_pose(self, tmp_path: Path):
        """Two Pandas facing each other carry a shared pose: their hands' relative pose holds while both move."""
        T_mount_b = np.diag([-1.0, -1.0, 1.0, 1.0])
        T_mount_b[0, 3] = 1.1
        urdf = assemble_urdf(
            tmp_path / "two_pandas.urdf",
            Path(panda_description.URDF_PATH),
            Path(panda_description.URDF_PATH),
            "panda_link0",
            "panda_link0",
            T_mount_b,
            "b_",
        )
        franka = Path(__file__).parents[2] / "fixtures/franka_collision_spheres.yaml"
        spheres = merge_collision_spheres(tmp_path / "two_pandas.yaml", [(franka, ""), (franka, "b_")])
        robot = Robot.load(str(urdf), load_collision_spheres=True, collision_spheres_path=spheres)
        # the preset's iterations on every device: the relative pose is a soft cost, still ~3 mm off after 60 on CPU
        config = presets.build_offline_pose_goal_config(max_retries=2, use_cuda_graph=DEVICE != "cpu")
        planner = MotionPlanner(
            replace(config, max_fixed_relative_poses=1), robot, ["panda_hand", "b_panda_hand"], device=DEVICE
        )
        start = np.zeros(robot.num_actuated_joints, np.float32)
        for prefix in ("", "b_"):  # both arms at the ready pose, hands 0.49 m apart
            for k, value in enumerate((0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)):
                start[robot.spec.actuated_joint_names.index(f"{prefix}panda_joint{k + 1}")] = value
        hands = [robot.link_names.index(name) for name in ("panda_hand", "b_panda_hand")]
        T_world_start = pose7_to_tf_mat(robot.forward_kinematics_numpy(start[None]))[0]
        goal = {}
        for name, link in zip(("panda_hand", "b_panda_hand"), hands):  # both hands shift alike: T_a_b can hold
            goal[name] = T_world_start[link].copy()
            goal[name][:3, 3] += (0.0, 0.25, 0.15)
        errors = []
        tray = FixedRelativePose("panda_hand", "b_panda_hand")
        for relative in (None, [tray], [replace(tray, at_waypoint=1)]):  # 1: the goal's
            q_traj = planner.solve_offline_numpy([start, goal], fixed_relative_poses=relative).q_traj[0]
            T_world_link = pose7_to_tf_mat(robot.forward_kinematics_numpy(q_traj))
            T_a_b = np.linalg.inv(T_world_link[:, hands[0]]) @ T_world_link[:, hands[1]]
            error = np.linalg.inv(T_a_b[-1 if relative and relative[0].at_waypoint else 0]) @ T_a_b[1:]  # vs reference
            angle = np.arccos(np.clip((np.trace(error[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1))
            errors.append((float(np.linalg.norm(error[:, :3, 3], axis=1).max()), float(angle.max())))
            final = T_world_link[-1, hands[0], :3, 3]
        print(f"relative pose error: {errors[0]} free, {errors[1]} held, {errors[2]} held from the goal (m, rad)")
        assert errors[1][0] < 0.002 and errors[1][1] < 0.02
        assert errors[2][0] < 0.3 * errors[0][0] and errors[2][1] < 0.5 * errors[0][1]  # held from the (free) goal
        assert errors[0][0] > 2 * errors[1][0]
        assert np.linalg.norm(final - goal["panda_hand"][:3, 3]) < 0.005


class TestHeldSelfCollision:
    bar = AttachedObject(
        "panda_hand", "box", (0.25, 0.015, 0.015), _translation(0.0, 0.0, 0.1)
    )  # 0.5 m, at the fingers

    def _panda(self) -> Robot:
        franka = Path(__file__).parents[2] / "fixtures/franka_collision_spheres.yaml"
        return Robot.load(
            panda_description.URDF_PATH,
            load_collision_spheres=True,
            collision_spheres_path=franka,
            self_collision_ignore_path=PANDA_IGNORE,
        )

    def _planner(self, robot: Robot, device: str = DEVICE, **config) -> MotionPlanner:
        config = presets.build_offline_pose_goal_config(use_cuda_graph=device != "cpu", **config)
        return MotionPlanner(replace(config, max_attached_spheres=8), robot, "panda_hand", device=device)

    def test_held_spheres_meet_the_holders_arm_but_its_hand(self):
        """A held bar meets every link but the hand's body, those one joint from it and the bar's ignored links, though
        the hand's own spheres skip links 3-7."""
        planner = self._planner(self._panda(), "cpu", max_iter=1, max_retries=0, num_ik_samples=8)
        robot, own = planner.robot, len(self._panda().spec.collision_sphere_radii)
        link, start = robot.spec.collision_spheres_link_indices, np.zeros(robot.num_actuated_joints, np.float32)
        for pairs in (
            TrajectorySelfCollisionTask(robot, 2)._active_pairs_np,
            SelfCollisionTask(robot)._pair_indices_np,
        ):
            met = {robot.link_names[link[i]] for i, j in pairs if i < own <= j}
            assert sorted(met) == [f"panda_link{k}" for k in "012345"]
        for ignored, expected in [((), "012345"), (("panda_link4",), "01235")]:
            planner.solve_offline_numpy(
                [start, start], attached_objects=[replace(self.bar, collision_ignored_links=ignored)]
            )
            task = planner._self_collision_tasks[0]
            on = planner._pair_active.numpy()[: task.num_pairs] != 0
            met = {robot.link_names[link[i]] for (i, j), m in zip(task._active_pairs_np, on) if m and i < own <= j}
            assert sorted(met) == [f"panda_link{k}" for k in expected]

    def test_held_bar_clears_its_own_forearm(self):
        """Joint 7 sweeps a held bar through the holder's forearm (link 5): holding it, the plan goes around."""
        planner = self._planner(self._panda(), max_retries=4)
        start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 0.88, -0.8, 0.04], np.float32)
        goal = start.copy()
        goal[6] = 2.4
        plans = [planner.solve_offline_numpy([start, goal], attached_objects=x).q_traj[0] for x in (None, [self.bar])]
        robot = planner.robot  # holding the bar, as the last solve left it
        link, radii = robot.spec.collision_spheres_link_indices, robot.spec.collision_sphere_radii
        bar_spheres = np.flatnonzero((link == len(robot.link_names) - 1) & (radii > 0))
        arm_spheres = np.flatnonzero(link == robot.link_names.index("panda_link5"))
        weights, gaps = np.linspace(0, 1, 20, endpoint=False)[:, None], []
        for q_traj in plans:  # the least gap of the bar to link 5 along the plan, 20 samples a step
            q = np.concatenate([(1 - weights) * a + weights * b for a, b in zip(q_traj[:-1], q_traj[1:])])
            T_world_sphere = pose7_to_tf_mat(robot.forward_kinematics_numpy(q))[:, link]
            centers = np.einsum("nsij,sj->nsi", T_world_sphere[..., :3, :3], robot.spec.local_collision_sphere_centers)
            centers += T_world_sphere[..., :3, 3]
            gap = np.linalg.norm(centers[:, bar_spheres, None] - centers[:, None, arm_spheres], axis=-1)
            gaps.append(float((gap - radii[bar_spheres, None] - radii[arm_spheres]).min()))
        print(f"held bar to link 5: {gaps[0]:.4f} m free, {gaps[1]:.4f} m held")
        assert gaps[0] < -0.005 < 0.0 < gaps[1]
