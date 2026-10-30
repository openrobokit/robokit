import numpy as np
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.utils.warp_utils import wp_vec7
from tests.conftest import box_mesh


class TestWarpIntegration:
    """Test RobotState.integrate()."""

    def test_integrate_fixed_base_preserves_world_base(self, panda_robot):
        q = wp.from_numpy(panda_robot.spec.zero_q, dtype=wp.float32)
        state = panda_robot.state(q=q)
        expected = np.array([[0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        wp.copy(state.T_world_base, wp.from_numpy(expected, dtype=wp_vec7))
        out = panda_robot.state(q=wp.clone(state.q))

        state.integrate(wp.zeros((1, panda_robot.num_actuated_joints), dtype=wp.float32), out=out)

        np.testing.assert_array_equal(out.T_world_base.numpy(), expected)

    def test_integrate_floating_base(self, panda_robot):
        robot = panda_robot
        T_world_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np.reshape(1, 7), dtype=wp_vec7)
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q, T_world_base=T_world_base)

        # create velocity with base motion
        velocity_np = np.zeros(6 + robot.num_actuated_joints, dtype=np.float32)
        velocity_np[0] = 0.1  # move base in x
        velocity = wp.from_numpy(velocity_np.reshape(1, -1), dtype=wp.float32)

        new_state = state.integrate(velocity)

        # check base has moved
        new_T_world_base_np = new_state.T_world_base.numpy()[0]
        assert new_T_world_base_np[0] > 0  # x should have increased


class TestWarpAutodiffJacobian:
    """Test consistency between analytic and autodiff Jacobians for Warp terms."""

    def test_frame_task_jacobian(self, panda_robot):
        robot = panda_robot
        q_np = np.random.randn(robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_world_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_world_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")

        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        frame_task = FrameTask(
            robot=robot,
            frame_index=ee_idx,
            T_world_target=T_world_target.reshape((1, 1)),
            position_weight=1.0,
            orientation_weight=1.0,
        )

        analytic_jacobian = frame_task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state}))
        autodiff_jacobian = autodiff_weighted_jacobian(frame_task, VarValues({VarKey.ROBOT: state}))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4, rtol=1e-4)

    def test_collision_task_jacobian(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        mesh = box_mesh(2.0)

        np.random.seed(0)
        q_np = np.random.randn(robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        q = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        state = robot.state(q=q)
        state = robot.forward_kinematics(state)
        state = robot.transform_collision_spheres(state)
        state = robot.compute_motion_subspace(state)

        sphere_indices = list(range(min(6, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        wm = WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.01,
            sphere_indices=sphere_indices,
        )

        task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        analytic_jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state}))
        autodiff_jacobian = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state}))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert analytic_np.shape == autodiff_np.shape
        assert np.allclose(analytic_np, autodiff_np, atol=3e-4, rtol=3e-4)


class TestMultiSeedConsistency:
    """Identical q within each instance must give identical residuals/Jacobians across its K candidates."""

    NUM_CANDIDATES = 4
    ACTUAL_BATCH = 2

    def _make_expanded_state(self, robot, rng):
        """Create un-expanded q and instance-major expanded state."""
        num_joints = robot.num_actuated_joints
        lo = robot.spec.actuated_joint_limits[:, 0]
        hi = robot.spec.actuated_joint_limits[:, 1]
        q_np = rng.uniform(lo, hi, size=(self.ACTUAL_BATCH, num_joints)).astype(np.float32)
        q_expanded = np.repeat(q_np, self.NUM_CANDIDATES, axis=0)
        state = robot.state(q=wp.from_numpy(q_expanded, dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        return q_np, state

    def _assert_candidates_consistent(self, array_np, name: str):
        """Assert all candidates for each instance produce the same value."""
        for i in range(self.ACTUAL_BATCH):
            base = i * self.NUM_CANDIDATES
            ref = array_np[base]
            for s in range(1, self.NUM_CANDIDATES):
                np.testing.assert_allclose(
                    array_np[base + s],
                    ref,
                    atol=1e-5,
                    rtol=1e-5,
                    err_msg=f"{name}: instance {i}, candidate {s} differs from candidate 0",
                )

    def test_position_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(42)
        q_np, state = self._make_expanded_state(robot, rng)

        target_link_idx = robot.link_names.index("panda_link7")
        single_state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(single_state)
        target_pose = single_state.get_T_world_link(target_link_idx)

        task = PositionTask(robot, target_link_idx, target_pose.reshape((target_pose.shape[0], 1)), weight=1.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()

        total_batch = self.ACTUAL_BATCH * self.NUM_CANDIDATES
        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_candidates_consistent(residual, "PositionTask residual")
        self._assert_candidates_consistent(jacobian, "PositionTask jacobian")

    def test_rotation_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(43)
        q_np, state = self._make_expanded_state(robot, rng)

        target_link_idx = robot.link_names.index("panda_link7")
        single_state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(single_state)
        target_pose = single_state.get_T_world_link(target_link_idx)

        task = RotationTask(robot, target_link_idx, target_pose.reshape((target_pose.shape[0], 1)), weight=1.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()

        total_batch = self.ACTUAL_BATCH * self.NUM_CANDIDATES
        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_candidates_consistent(residual, "RotationTask residual")
        self._assert_candidates_consistent(jacobian, "RotationTask jacobian")

    def test_frame_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(44)
        q_np, state = self._make_expanded_state(robot, rng)

        target_link_idx = robot.link_names.index("panda_link7")
        single_state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(single_state)
        target_pose = single_state.get_T_world_link(target_link_idx)

        task = FrameTask(
            robot,
            target_link_idx,
            target_pose.reshape((target_pose.shape[0], 1)),
            position_weight=1.0,
            orientation_weight=1.0,
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()

        total_batch = self.ACTUAL_BATCH * self.NUM_CANDIDATES
        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_candidates_consistent(residual, "FrameTask residual")
        self._assert_candidates_consistent(jacobian, "FrameTask jacobian")

    def test_configuration_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(45)
        q_np, state = self._make_expanded_state(robot, rng)

        total_batch = self.ACTUAL_BATCH * self.NUM_CANDIDATES
        rest_q = robot.spec.midrange_q
        task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=0.1)
        task.set_target(wp.from_numpy(q_np[:, : robot.num_actuated_joints].copy(), dtype=wp.float32))

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()

        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_candidates_consistent(residual, "ConfigurationTask residual")
        self._assert_candidates_consistent(jacobian, "ConfigurationTask jacobian")

    def test_velocity_limit_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(46)
        q_np, state = self._make_expanded_state(robot, rng)

        total_batch = self.ACTUAL_BATCH * self.NUM_CANDIDATES
        prev_q = rng.uniform(
            robot.spec.actuated_joint_limits[:, 0],
            robot.spec.actuated_joint_limits[:, 1],
            size=(self.ACTUAL_BATCH, robot.num_actuated_joints),
        ).astype(np.float32)
        prev_state = robot.state(q=wp.from_numpy(prev_q, dtype=wp.float32))

        task = VelocityLimitTask(robot=robot, dt=0.1, weight=1.0)
        task.set_prev_state(prev_state)

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()

        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_candidates_consistent(residual, "VelocityLimitTask residual")
        self._assert_candidates_consistent(jacobian, "VelocityLimitTask jacobian")

    def test_smoothness_task_multi_seed(self, panda_robot):
        robot = panda_robot
        rng = np.random.default_rng(47)
        q_np, state = self._make_expanded_state(robot, rng)

        total_batch = self.ACTUAL_BATCH * self.NUM_CANDIDATES
        prev_q = rng.uniform(
            robot.spec.actuated_joint_limits[:, 0],
            robot.spec.actuated_joint_limits[:, 1],
            size=(self.ACTUAL_BATCH, robot.num_actuated_joints),
        ).astype(np.float32)
        prev_state = robot.state(q=wp.from_numpy(prev_q, dtype=wp.float32))

        task = SmoothnessTask(robot=robot, weight=1.0)

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}), prev_var=prev_state).numpy()
        jacobian = task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state}), prev_var=prev_state).numpy()

        assert residual.shape[0] == total_batch
        assert jacobian.shape[0] == total_batch
        self._assert_candidates_consistent(residual, "SmoothnessTask residual")
        self._assert_candidates_consistent(jacobian, "SmoothnessTask jacobian")
