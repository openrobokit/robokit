"""Capacity vs live width: an optimizer set up at N must solve any batch <= N as a prefix, never reallocating."""

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.opt.gd_optimizer import GDOptimizer, GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.terms.dense.frame_task import FrameTask
from robokit.utils.warp_utils import wp_vec7


def _device():
    return "cuda:0" if wp.is_cuda_available() else "cpu"


@pytest.fixture(scope="module")
def problem():
    robot = Robot.load(load_robot_description("panda_description"))
    ee = robot.link_names.index("panda_hand")
    rng = np.random.default_rng(0)
    q_goal = (robot.spec.midrange_q + rng.uniform(-0.3, 0.3, (64, robot.num_actuated_joints))).astype(np.float32)
    goal = robot.forward_kinematics(robot.state(q=wp.from_numpy(q_goal, dtype=wp.float32, device=_device())))
    T_world_goal = goal.get_T_world_link(ee).numpy()  # (64, 7) xyz + wxyz
    q0 = np.tile(robot.spec.midrange_q.astype(np.float32), (64, 1))
    return robot, ee, T_world_goal, q0


def _solve(problem, make_optimizer, n: int, optimizer=None):
    """Compare fresh/reused optimizers on rows [:n] with fixed-address capture targets and per-width states."""
    robot, ee, T_world_goal, q0 = problem
    if optimizer is None:
        target = wp.zeros((64, 1), dtype=wp_vec7, device=_device())
        optimizer = make_optimizer([FrameTask(robot, ee, target, 10.0, 5.0)])
        optimizer.states = {}
    # Use capacity buffers with live prefix views; captured addresses remain stable when views are re-sliced.
    if not hasattr(optimizer, "target"):
        optimizer.target = optimizer.terms[0].T_world_target
    optimizer.terms[0].T_world_target = optimizer.target[:n]
    wp.copy(
        optimizer.target, wp.from_numpy(T_world_goal[:n].reshape(n, 1, 7), dtype=wp_vec7, device=_device()), count=n
    )
    if n not in optimizer.states:
        optimizer.states[n] = robot.state(q=wp.zeros((n, q0.shape[1]), dtype=wp.float32, device=_device()))
    state = optimizer.states[n]
    wp.copy(state.q, wp.from_numpy(q0[:n].copy(), dtype=wp.float32, device=_device()))
    state.invalidate()
    values, costs = optimizer.solve(VarValues({VarKey.ROBOT: state}))
    return values.get(VarKey.ROBOT).q.numpy(), costs.numpy(), optimizer


OPTIMIZERS = {
    "lm": lambda terms, g: LMOptimizer(
        terms, LMOptimizerConfig(max_iter=10, cuda_graph_mode="iter" if g else "none"), _device()
    ),
    "lbfgs": lambda terms, g: LBFGSOptimizer(
        terms, device=_device(), config=LBFGSOptimizerConfig(max_iter=10, cuda_graph_mode="full" if g else "none")
    ),
    "gd": lambda terms, g: GDOptimizer(
        terms, device=_device(), config=GDOptimizerConfig(max_iter=10, cuda_graph_mode="iter" if g else "none")
    ),
}


class TestVariedBatch:
    @pytest.mark.parametrize("name", list(OPTIMIZERS))
    @pytest.mark.parametrize("use_cuda_graph", [False, True])
    def test_prefix_batch_matches_fresh_optimizer(self, problem, name, use_cuda_graph):
        if use_cuda_graph and not wp.is_cuda_available():
            pytest.skip("graph capture needs CUDA")
        make_graph = lambda terms: OPTIMIZERS[name](terms, use_cuda_graph)  # noqa: E731

        _, _, big = _solve(problem, make_graph, 64)  # sets capacity
        for n in (16, 4, 64):
            q_prefix, c_prefix, _ = _solve(problem, make_graph, n, optimizer=big)
            q_fresh, c_fresh, _ = _solve(problem, make_graph, n)
            np.testing.assert_array_equal(q_prefix, q_fresh, err_msg=f"{name} n={n}")
            np.testing.assert_array_equal(c_prefix, c_fresh, err_msg=f"{name} n={n}")

    @pytest.mark.parametrize("name", list(OPTIMIZERS))
    def test_growing_past_capacity_raises(self, problem, name):
        make = lambda terms: OPTIMIZERS[name](terms, False)  # noqa: E731
        _, _, small = _solve(problem, make, 4)
        with pytest.raises(ValueError, match="batch"):
            _solve(problem, make, 16, optimizer=small)
