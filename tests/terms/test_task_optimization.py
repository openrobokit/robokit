"""Each warp task term reduces cost during optimization; multi-instance + multi-candidate layouts isolate targets."""

import numpy as np
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.utils.warp_utils import wp_vec7
from tests.conftest import box_mesh


# --- fixtures --------------------------------------------------------------
# --- helpers ---------------------------------------------------------------
def _initial_cost(terms, state):
    """Sum of squared residuals across all terms for a single-batch state."""
    cost = 0.0
    for t in terms:
        r = t.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        cost += float(np.sum(r**2))
    return cost


def _solve(terms, state, max_iter=30, lm_lambda=1.0):
    """Run LMOptimizer and return (final_state, final_costs_np)."""
    config = LMOptimizerConfig(max_iter=max_iter, lm_lambda=lm_lambda)
    optimizer = LMOptimizer(terms=terms, config=config)
    final_state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
    return final_state, optimizer.costs.numpy()


# --- test A: optimization effectiveness - each term reduces cost -----------
class TestOptimizationEffectiveness:
    """Verify that each task term actually reduces cost when used in LMOptimizer."""

    def test_position_task(self, panda_robot):
        robot = panda_robot
        ee_idx = robot.link_names.index("panda_hand")
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        target_pose = target_state.get_T_world_link(ee_idx)

        task = PositionTask(robot, ee_idx, target_pose.reshape((target_pose.shape[0], 1)), weight=10.0)
        terms = [task, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        robot.forward_kinematics(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.1

    def test_rotation_task(self, panda_robot):
        robot = panda_robot
        ee_idx = robot.link_names.index("panda_hand")
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        target_pose = target_state.get_T_world_link(ee_idx)

        task = RotationTask(robot, ee_idx, target_pose.reshape((target_pose.shape[0], 1)), weight=5.0)
        terms = [task, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        robot.forward_kinematics(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.1

    def test_frame_task(self, panda_robot):
        robot = panda_robot
        ee_idx = robot.link_names.index("panda_hand")
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        target_pose = target_state.get_T_world_link(ee_idx)

        task = FrameTask(
            robot,
            ee_idx,
            target_pose.reshape((target_pose.shape[0], 1)),
            position_weight=10.0,
            orientation_weight=5.0,
        )
        terms = [task, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        robot.forward_kinematics(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.1

    def test_fixed_frame_tasks(self, panda_robot):
        robot = panda_robot
        hand_idx = robot.link_names.index("panda_hand")
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        target_pose = target_state.get_T_world_link(hand_idx)

        target = target_pose.reshape((target_pose.shape[0], 1))
        tasks = [
            PositionTask(robot, [hand_idx], target, weight=10.0, fixed_target=True),
            RotationTask(robot, [hand_idx], target, weight=5.0, fixed_target=True),
        ]
        terms = [*tasks, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        robot.forward_kinematics(state)
        init_cost = _initial_cost(tasks, state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.1

    def test_com_position_task(self, g1_robot):
        robot = g1_robot
        T_base_np = np.array([[0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)

        target_state = robot.forward_kinematics(
            robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32), T_world_base=T_base)
        )
        # compute CoM at target config
        T_world_link = target_state.T_world_link.numpy()[0]
        masses = robot.spec.link_masses
        local_coms = robot.spec.link_local_com_positions
        from scipy.spatial.transform import Rotation

        com = np.zeros(3)
        for i in range(robot.spec.num_links):
            m = masses[i]
            if m > 0:
                pos = T_world_link[i, :3]
                q_wxyz = T_world_link[i, 3:]
                rot = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                com += m * (pos + rot.apply(local_coms[i]))
        target_com = (com / robot.spec.total_mass).astype(np.float32)

        # shift target slightly so there is a non-trivial residual
        shifted_target = target_com + np.array([0.05, 0.0, 0.0], dtype=np.float32)
        task = ComPositionTask(robot=robot, target_com_position=shifted_target, weight=5.0)
        terms = [task, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32), T_world_base=T_base)
        robot.forward_kinematics(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.001

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32), T_world_base=T_base)
        final_state, final_costs = _solve(terms, state, max_iter=50)
        assert float(final_costs[0]) < init_cost * 0.5

    def test_frame_position_huber_task(self, panda_robot):
        robot = panda_robot
        link_idx = robot.link_names.index("panda_hand")
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        target_pos = target_state.T_world_link.numpy()[0, link_idx, :3]

        task = FrameVectorTask(
            robot=robot,
            origin_link_indices=[-1],
            task_link_indices=[link_idx],
            targets=target_pos.reshape(1, 3),
            huber_delta=0.05,
            weight=10.0,
        )
        task.init_buffers(wp.get_device("cpu"))
        terms = [task, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        robot.forward_kinematics(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.001

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.1

    def test_frame_vector_distance_task(self, panda_robot):
        robot = panda_robot
        origin_idx = robot.link_names.index("panda_link0")
        task_idx = robot.link_names.index("panda_hand")

        # get vector at midrange config
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        all_pos = target_state.T_world_link.numpy()[0, :, :3]
        target_vector = (all_pos[task_idx] - all_pos[origin_idx]).reshape(1, 3)

        task = FrameVectorTask(
            robot=robot,
            origin_link_indices=[origin_idx],
            task_link_indices=[task_idx],
            targets=target_vector,
            huber_delta=0.1,
            huber_on_norm=True,
        )
        task.init_buffers(wp.get_device("cpu"))
        terms = [task, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        robot.forward_kinematics(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.001

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.5

    def test_configuration_task(self, panda_robot):
        robot = panda_robot
        rest_q = robot.spec.midrange_q
        start_q = (robot.spec.zero_q + 0.5).astype(np.float32)

        task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=1.0)
        terms = [task]

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        init_cost = _initial_cost(terms, state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.01

    def test_position_limit(self, panda_robot):
        robot = panda_robot
        # start far outside joint limits
        start_q = (robot.spec.midrange_q + 10.0).astype(np.float32)

        task = PositionLimit(robot=robot, weight=1.0)
        terms = [task]

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        init_cost = _initial_cost(terms, state)
        assert init_cost > 1.0

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.01

    def test_smoothness_task(self, panda_robot):
        robot = panda_robot
        prev_q = robot.spec.zero_q.astype(np.float32)
        start_q = (robot.spec.zero_q + 0.5).astype(np.float32)
        prev_state = robot.state(q=wp.from_numpy(prev_q, dtype=wp.float32))

        task = SmoothnessTask(robot=robot, prev_var=prev_state, weight=1.0)
        terms = [task]

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        init_cost = _initial_cost(terms, state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state)
        assert float(final_costs[0]) < init_cost * 0.01

    def test_velocity_limit_task(self, panda_robot):
        robot = panda_robot
        prev_q = robot.spec.zero_q.astype(np.float32)
        # large step so velocity limits are violated with small dt
        start_q = (robot.spec.zero_q + 2.0).astype(np.float32)
        prev_state = robot.state(q=wp.from_numpy(prev_q, dtype=wp.float32))

        task = VelocityLimitTask(robot=robot, dt=0.01, prev_state_var=prev_state, weight=1.0)
        terms = [task]

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        init_cost = _initial_cost(terms, state)
        assert init_cost > 0.1

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state, max_iter=50)
        assert float(final_costs[0]) < init_cost * 0.5

    def test_base_position_limit(self, panda_robot):
        robot = panda_robot
        # Start with base z below lo; expect the optimizer to push z up into the [lo, hi] band.
        T_base_np = np.array([[0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7)
        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32), T_world_base=T_base)

        task = PositionLimit(robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=100.0)
        terms = [task]

        init_cost = _initial_cost(terms, state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32), T_world_base=T_base)
        final_state, final_costs = _solve(terms, state, max_iter=50)
        final_z = float(final_state.T_world_base.numpy()[0, 2])
        assert final_z >= 0.3 - 1e-3, f"base z={final_z} did not clear lo=0.3"
        assert float(final_costs[0]) < init_cost * 0.05

    def test_collision_task(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

        # tight box so collision spheres are inside or very close to surfaces
        mesh = box_mesh(0.3)
        wm = WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))

        sphere_indices = list(range(min(6, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=5.0,
            margin=0.1,
            sphere_indices=sphere_indices,
        )
        terms = [task, PositionLimit(robot, weight=2.0)]

        # start at midrange_q - spheres closer to box surfaces
        q_init = robot.spec.midrange_q.astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32, device=device))
        robot.forward_kinematics(state)
        state = robot.transform_collision_spheres(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.001, f"Collision task initial cost too low: {init_cost}"

        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32, device=device))
        final_state, final_costs = _solve(terms, state, max_iter=50)
        assert float(final_costs[0]) < init_cost * 0.9

    def test_self_collision_task(self, panda_robot_with_collision):
        robot = panda_robot_with_collision

        task = SelfCollisionTask(
            robot=robot,
            weight=5.0,
            margin=0.5,  # large margin to ensure nonzero residual
        )
        terms = [task, PositionLimit(robot, weight=2.0)]

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        init_cost = _initial_cost([task], state)
        assert init_cost > 0.01

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state, max_iter=50)
        assert float(final_costs[0]) < init_cost * 0.5


# --- test B: multi-instance target isolation - each instance converges to own target
class TestMultiInstanceTargetIsolation:
    """Verify batch_size=2 with different targets per instance converges each to its own target."""

    NUM_CANDIDATES = 4
    BATCH_SIZE = 2

    def _expand_for_candidates(self, array_np):
        """Repeat each row NUM_CANDIDATES times (instance-major)."""
        return np.repeat(array_np, self.NUM_CANDIDATES, axis=0)

    def _check_per_instance_convergence(self, achieved_np, target_a_np, target_b_np, atol=0.02):
        """Check instance 0 candidates near target A, instance 1 candidates near target B."""
        for s in range(self.NUM_CANDIDATES):
            np.testing.assert_allclose(
                achieved_np[0 * self.NUM_CANDIDATES + s],
                target_a_np,
                atol=atol,
                err_msg=f"Instance 0, candidate {s} did not converge to target A",
            )
            np.testing.assert_allclose(
                achieved_np[1 * self.NUM_CANDIDATES + s],
                target_b_np,
                atol=atol,
                err_msg=f"Instance 1, candidate {s} did not converge to target B",
            )

    def test_position_task(self, panda_robot):
        robot = panda_robot
        ee_idx = robot.link_names.index("panda_hand")

        # two different configs → two different ee positions
        q_a = robot.spec.midrange_q.astype(np.float32)
        q_b = (robot.spec.midrange_q + 0.3).astype(np.float32)
        q_b = np.clip(q_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        targets_q = np.stack([q_a, q_b])
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(targets_q, dtype=wp.float32)))
        target_pose = target_state.get_T_world_link(ee_idx)  # [2, 7]
        target_pos_np = target_pose.numpy()[:, :3]  # [2, 3]

        # expand targets for candidates
        expanded_target = wp.from_numpy(self._expand_for_candidates(target_pose.numpy()), dtype=wp_vec7)
        task = PositionTask(robot, ee_idx, expanded_target.reshape((expanded_target.shape[0], 1)), weight=10.0)
        terms = [task, PositionLimit(robot, weight=2.0)]

        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        q_init = np.tile(robot.spec.zero_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32))

        final_state, _ = _solve(terms, state, max_iter=50)
        final_state = robot.forward_kinematics(final_state)
        achieved = final_state.get_T_world_link(ee_idx).numpy()[:, :3]

        self._check_per_instance_convergence(achieved, target_pos_np[0], target_pos_np[1])

    def test_rotation_task(self, panda_robot):
        robot = panda_robot
        ee_idx = robot.link_names.index("panda_hand")

        q_a = robot.spec.midrange_q.astype(np.float32)
        q_b = (robot.spec.midrange_q + 0.4).astype(np.float32)
        q_b = np.clip(q_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        targets_q = np.stack([q_a, q_b])
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(targets_q, dtype=wp.float32)))
        target_pose = target_state.get_T_world_link(ee_idx)
        q_wxyz_world_target_np = target_pose.numpy()[:, 3:]  # [2, 4]

        expanded_target = wp.from_numpy(self._expand_for_candidates(target_pose.numpy()), dtype=wp_vec7)
        task = RotationTask(robot, ee_idx, expanded_target.reshape((expanded_target.shape[0], 1)), weight=5.0)
        terms = [task, PositionLimit(robot, weight=2.0)]

        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        q_init = np.tile(robot.spec.zero_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32))

        final_state, _ = _solve(terms, state, max_iter=50)
        final_state = robot.forward_kinematics(final_state)
        q_wxyz_world_ee = final_state.get_T_world_link(ee_idx).numpy()[:, 3:]  # [8, 4]

        for s in range(self.NUM_CANDIDATES):
            for inst, tgt in enumerate([q_wxyz_world_target_np[0], q_wxyz_world_target_np[1]]):
                dot = abs(float(np.sum(q_wxyz_world_ee[inst * self.NUM_CANDIDATES + s] * tgt)))
                assert dot > 0.99, f"Instance {inst}, candidate {s}: quaternion dot {dot:.4f} < 0.99"

    def test_frame_task(self, panda_robot):
        robot = panda_robot
        ee_idx = robot.link_names.index("panda_hand")

        q_a = robot.spec.midrange_q.astype(np.float32)
        q_b = (robot.spec.midrange_q + 0.2).astype(np.float32)
        q_b = np.clip(q_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        targets_q = np.stack([q_a, q_b])
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(targets_q, dtype=wp.float32)))
        target_pose = target_state.get_T_world_link(ee_idx)
        target_pos_np = target_pose.numpy()[:, :3]

        expanded_target = wp.from_numpy(self._expand_for_candidates(target_pose.numpy()), dtype=wp_vec7)
        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        task = FrameTask(
            robot,
            ee_idx,
            expanded_target.reshape((total_batch, 1)),
            position_weight=10.0,
            orientation_weight=5.0,
        )
        terms = [task, PositionLimit(robot, weight=2.0)]
        # start from midrange (closer to targets) for better convergence
        q_init = np.tile(robot.spec.midrange_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32))

        final_state, _ = _solve(terms, state, max_iter=100)
        final_state = robot.forward_kinematics(final_state)
        achieved = final_state.get_T_world_link(ee_idx).numpy()[:, :3]

        self._check_per_instance_convergence(achieved, target_pos_np[0], target_pos_np[1], atol=0.05)

    def test_fixed_frame_tasks(self, panda_robot):
        robot = panda_robot
        hand_idx = robot.link_names.index("panda_hand")

        q_a = robot.spec.midrange_q.astype(np.float32)
        q_b = (robot.spec.midrange_q + 0.2).astype(np.float32)
        q_b = np.clip(q_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        targets_q = np.stack([q_a, q_b])
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(targets_q, dtype=wp.float32)))
        target_pos_np = target_state.get_T_world_link(hand_idx).numpy()[:, :3]

        # expand each target for every candidate
        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        expanded_np = self._expand_for_candidates(target_state.get_T_world_link(hand_idx).numpy())
        expanded_target = wp.from_numpy(expanded_np, dtype=wp_vec7)

        target = expanded_target.reshape((total_batch, 1))
        tasks = [
            PositionTask(robot, [hand_idx], target, weight=10.0, fixed_target=True),
            RotationTask(robot, [hand_idx], target, weight=5.0, fixed_target=True),
        ]
        terms = [*tasks, PositionLimit(robot, weight=2.0)]

        # start from midrange for better convergence
        q_init = np.tile(robot.spec.midrange_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32))

        final_state, _ = _solve(terms, state, max_iter=100)
        final_state = robot.forward_kinematics(final_state)
        achieved = final_state.get_T_world_link(hand_idx).numpy()[:, :3]

        self._check_per_instance_convergence(achieved, target_pos_np[0], target_pos_np[1], atol=0.05)

    def test_configuration_task(self, panda_robot):
        robot = panda_robot
        # two different rest configs
        rest_a = robot.spec.midrange_q.astype(np.float32)
        rest_b = (robot.spec.midrange_q + 0.3).astype(np.float32)
        rest_b = np.clip(rest_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        rest_q_batch = np.stack([rest_a, rest_b])  # [2, num_joints]

        task = ConfigurationTask(robot=robot, rest_q=rest_a, weight=1.0)
        task.set_target(wp.from_numpy(rest_q_batch, dtype=wp.float32))
        terms = [task]

        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        start_q = np.tile(robot.spec.zero_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))

        final_state, _ = _solve(terms, state, max_iter=50)
        final_q = final_state.q.numpy()

        self._check_per_instance_convergence(final_q, rest_a, rest_b, atol=0.05)

    def test_frame_position_huber_task(self, panda_robot):
        robot = panda_robot
        link_idx = robot.link_names.index("panda_hand")

        q_a = robot.spec.midrange_q.astype(np.float32)
        q_b = (robot.spec.midrange_q + 0.3).astype(np.float32)
        q_b = np.clip(q_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        targets_q = np.stack([q_a, q_b])
        target_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(targets_q, dtype=wp.float32)))
        target_pos = target_state.T_world_link.numpy()[:, link_idx, :3]  # [2, 3]

        target_positions = np.zeros((2, 3, 3), dtype=np.float32)
        target_positions[:, 2] = target_pos

        task = FrameVectorTask(
            robot=robot,
            origin_link_indices=[-1],
            task_link_indices=[link_idx],
            targets=target_positions,
            huber_delta=0.05,
            weight=10.0,
            target_indices=[2],
        )
        task.init_buffers(wp.get_device("cpu"))
        terms = [task, PositionLimit(robot, weight=2.0)]

        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        q_init = np.tile(robot.spec.zero_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32))

        final_state, _ = _solve(terms, state, max_iter=50)
        final_state = robot.forward_kinematics(final_state)
        achieved = final_state.get_T_world_link(link_idx).numpy()[:, :3]

        self._check_per_instance_convergence(achieved, target_pos[0], target_pos[1])

    def test_smoothness_task(self, panda_robot):
        robot = panda_robot
        # two different prev states per instance
        prev_a = robot.spec.midrange_q.astype(np.float32)
        prev_b = (robot.spec.midrange_q + 0.5).astype(np.float32)
        prev_b = np.clip(prev_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        prev_q_batch = np.stack([prev_a, prev_b])  # [2, num_joints]
        prev_state = robot.state(q=wp.from_numpy(prev_q_batch, dtype=wp.float32))

        task = SmoothnessTask(robot=robot, weight=1.0)
        task.set_prev_state(prev_state)
        terms = [task]

        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        # start from zero_q - optimizer should push toward prev states
        start_q = np.tile(robot.spec.zero_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))

        final_state, _ = _solve(terms, state, max_iter=50)
        final_q = final_state.q.numpy()

        # smoothness task minimizes (q - q_prev), so should converge to prev state
        self._check_per_instance_convergence(final_q, prev_a, prev_b, atol=0.05)

    def test_velocity_limit_task(self, panda_robot):
        """VelocityLimitTask with different prev_states should reduce cost independently."""
        robot = panda_robot
        prev_a = robot.spec.midrange_q.astype(np.float32)
        prev_b = (robot.spec.midrange_q + 0.3).astype(np.float32)
        prev_b = np.clip(prev_b, robot.spec.actuated_joint_limits[:, 0], robot.spec.actuated_joint_limits[:, 1])

        prev_q_batch = np.stack([prev_a, prev_b])
        prev_state = robot.state(q=wp.from_numpy(prev_q_batch, dtype=wp.float32))

        task = VelocityLimitTask(robot=robot, dt=0.01, weight=1.0)
        task.set_prev_state(prev_state)
        terms = [task]

        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        # start far from both prev states so velocity limits are violated
        start_q = np.tile((robot.spec.zero_q + 2.0).astype(np.float32), (total_batch, 1))
        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))

        init_cost = _initial_cost(terms, state)

        state = robot.state(q=wp.from_numpy(start_q, dtype=wp.float32))
        final_state, final_costs = _solve(terms, state, max_iter=50)

        # cost should decrease for all instances
        assert float(final_costs.sum()) < init_cost * 0.5

        # candidates within each instance should have similar final q
        final_q = final_state.q.numpy()
        for inst in range(self.BATCH_SIZE):
            base = inst * self.NUM_CANDIDATES
            ref = final_q[base]
            for s in range(1, self.NUM_CANDIDATES):
                np.testing.assert_allclose(
                    final_q[base + s],
                    ref,
                    atol=0.1,
                    err_msg=f"Instance {inst}: candidate {s} differs from candidate 0",
                )

    def test_com_position_task(self, g1_robot):
        robot = g1_robot
        T_base_np = np.array([[0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        # two different CoM targets (batched)
        target_com_a = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        target_com_b = np.array([0.05, 0.0, 0.5], dtype=np.float32)
        target_com_batch = np.stack([target_com_a, target_com_b])  # [2, 3]

        task = ComPositionTask(robot=robot, target_com_position=target_com_batch, weight=5.0)
        terms = [task, PositionLimit(robot, weight=2.0)]

        total_batch = self.BATCH_SIZE * self.NUM_CANDIDATES
        T_base_expanded = wp.from_numpy(np.tile(T_base_np, (total_batch, 1)), dtype=wp_vec7)
        q_init = np.tile(robot.spec.zero_q, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32), T_world_base=T_base_expanded)

        init_cost = _initial_cost(terms, state)

        state = robot.state(q=wp.from_numpy(q_init, dtype=wp.float32), T_world_base=T_base_expanded)
        final_state, final_costs = _solve(terms, state, max_iter=50)

        # cost should decrease
        assert float(final_costs.sum()) < init_cost * 0.5
