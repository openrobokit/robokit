import numpy as np
import pytest
import warp as wp

from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.optimizer import TILE_THREADS, _jtr_kernel
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms import FrameTask, PositionLimit
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.utils.warp_utils import wp_vec7


def _target_pose() -> wp.array:
    target = np.array([[[0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    return wp.from_numpy(target, dtype=wp_vec7)


@wp.kernel(enable_backward=False, module="unique")
def _poison_cpu_tile_stack(out: wp.array1d(dtype=wp.float32)):
    wp.tile_store(
        out, wp.tile_ones(shape=(4096,), dtype=wp.float32, storage="shared") * wp.nan
    )  # no tile_full on warp < 1.9


requires_cuda = pytest.mark.skipif(
    not wp.is_cuda_available(), reason="Analytic gradient IK convergence is compiler-dependent on CPU"
)


class TestLBFGSOptimizer:
    @requires_cuda
    @pytest.mark.parametrize("history", [0, 2])
    def test_scalar_direction(self, history: int):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(5)
        gradient = rng.normal(size=6).astype(np.float32)
        s = rng.normal(size=(2, 6)).astype(np.float32)
        y = s * np.arange(1, 7, dtype=np.float32)
        # the fused two-loop runs fixed-length loops, so unfilled history slots must carry rho 0
        rho = np.where(np.arange(2) < history, 1.0 / np.sum(s * y, axis=1), 0.0).astype(np.float32)
        expected = gradient.copy()
        alpha = np.zeros(2, dtype=np.float32)
        for i in reversed(range(history)):
            alpha[i] = rho[i] * np.dot(s[i], expected)
            expected -= alpha[i] * y[i]
        # Barzilai-Borwein initial Hessian from the newest pair; h0_scale until there is one
        newest = history - 1
        expected *= np.dot(s[newest], y[newest]) / np.dot(y[newest], y[newest]) if history else 0.5
        for i in range(history):
            expected += (alpha[i] - rho[i] * np.dot(y[i], expected)) * s[i]
        direction = wp.empty((1, 6), dtype=wp.float32, device="cuda:0")
        slope = wp.empty(1, dtype=wp.float32, device="cuda:0")
        trials = wp.empty((2, 6), dtype=wp.float32, device="cuda:0")
        mask = np.array([1, 1, 0, 1, 1, 1], dtype=np.float32)
        wp.launch(
            LBFGSOptimizer._build_kernel(6, 2, 2, 32, 0, 0, 0, 0.0, 0.9, 0.5),
            dim=32,
            inputs=[
                # the kernel records the previous step's pair first: s = 0 records nothing
                wp.zeros((1, 6), dtype=wp.float32, device="cuda:0"),
                wp.array(gradient[None], device="cuda:0"),
                wp.zeros((1, 6), dtype=wp.float32, device="cuda:0"),
                wp.array(np.einsum("id,jd->ij", s, y)[None], device="cuda:0"),  # s_i . y_j
                wp.array(s[None], device="cuda:0"),
                wp.array(y[None], device="cuda:0"),
                wp.array(rho[None], device="cuda:0"),
                wp.array([history], dtype=wp.int32, device="cuda:0"),
                wp.zeros(1, dtype=wp.int32, device="cuda:0"),
                wp.array([0.5, 2.0], dtype=wp.float32, device="cuda:0"),
                wp.array(mask, device="cuda:0"),
                # the pending step is off here
                *[0, 1, None, None],
            ],
            outputs=[direction, slope, trials, None, None, None, None],
            block_dim=32,
            device="cuda:0",
        )
        torch.testing.assert_close(torch.from_numpy(direction.numpy()[0]), torch.from_numpy(-expected))
        torch.testing.assert_close(torch.from_numpy(slope.numpy()), torch.tensor([-np.dot(gradient, expected)]))
        torch.testing.assert_close(
            torch.from_numpy(trials.numpy()),
            torch.from_numpy(np.outer([0.5, 2.0], -expected * mask).astype(np.float32)),
        )

    @requires_cuda
    @pytest.mark.parametrize("threads", [32, 64])  # two elements per thread, or two warps per row
    def test_direction_over_a_wrapping_history(self, threads: int):
        """Recording more pairs than the history keeps: each direction is the two-loop's on the newest pairs."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(3)
        D, M = 40, 3
        A = rng.normal(size=(D, D)) / np.sqrt(D)
        A = A @ A.T + np.eye(D)  # y = A s, so every pair is recorded
        zeros = lambda *shape: wp.zeros(shape, dtype=wp.float32, device="cuda:0")  # noqa: E731
        last_step, gradient, gradient_prev, direction = zeros(1, D), zeros(1, D), zeros(1, D), zeros(1, D)
        history = [zeros(1, M, M), zeros(1, M, D), zeros(1, M, D), zeros(1, M)]
        count, start = wp.zeros(1, dtype=wp.int32, device="cuda:0"), wp.zeros(1, dtype=wp.int32, device="cuda:0")
        pairs, g_prev = [], rng.normal(size=D)
        for _ in range(7):
            step = rng.normal(size=D)
            g = g_prev + A @ step
            last_step.assign(step[None].astype(np.float32))
            gradient.assign(g[None].astype(np.float32))
            gradient_prev.assign(g_prev[None].astype(np.float32))
            wp.launch(
                LBFGSOptimizer._build_kernel(D, M, 1, threads, 0, 0, 0, 0.0, 0.9, 0.5),
                dim=threads,
                inputs=[last_step, gradient, gradient_prev, *history, count, start]
                + [wp.array([1.0], dtype=wp.float32, device="cuda:0"), wp.ones(D, device="cuda:0")]
                + [0, 1, None, None],
                outputs=[direction, zeros(1), zeros(1, D), None, None, None, None],
                block_dim=threads,
                device="cuda:0",
            )
            pairs = (pairs + [(step, g - g_prev)])[-M:]
            q, alpha = g.copy(), []
            for s, y in reversed(pairs):
                alpha.append(s @ q / (s @ y))
                q -= alpha[-1] * y
            q *= pairs[-1][0] @ pairs[-1][1] / (pairs[-1][1] @ pairs[-1][1])  # Barzilai-Borwein
            for (s, y), a in zip(pairs, reversed(alpha)):
                q += (a - y @ q / (s @ y)) * s
            torch.testing.assert_close(torch.from_numpy(direction.numpy()[0]), torch.from_numpy(-q).float())
            g_prev = g

    @requires_cuda
    @pytest.mark.parametrize("case", ["largest_wolfe", "none_takes_smallest"])
    @pytest.mark.parametrize("zero_trials", [False, True])  # leave the trials' costs and gradients, or zero them
    def test_approx_wolfe_selection(self, case: str, zero_trials: bool):
        alphas = np.array([0.1, 0.3, 0.7, 1.0], dtype=np.float32)
        direction = np.array([1.0, -1.0], dtype=np.float32)
        steps = alphas[:, None] * direction
        slope_0 = -2.0  # gradient . direction at the smallest trial
        costs = np.array([10.0, 9.0, 8.0, 7.0], dtype=np.float32)
        slopes = np.array([slope_0, -1.5, -1.9, -0.5], dtype=np.float32)  # curvature needs slope >= 0.9 * slope_0
        if case == "none_takes_smallest":
            costs[1:] = 20.0
        expected = {"largest_wolfe": 3, "none_takes_smallest": 0}[case]
        last_step = wp.empty((1, 2), dtype=wp.float32, device="cuda:0")
        winner = wp.empty(1, dtype=wp.int32, device="cuda:0")
        # the kernel, stopping at the step: slopes from each trial's gradient (g . direction = slope), the choice
        gradients = np.stack([slopes / 2, -slopes / 2], axis=1).astype(np.float32)
        expanded_gradient = wp.array(gradients, device="cuda:0")
        expanded_costs = wp.array(costs, device="cuda:0")
        costs_out = wp.array([10.5], dtype=wp.float32, device="cuda:0")
        gradient = wp.zeros((1, 2), dtype=wp.float32, device="cuda:0")
        zeros = lambda *shape: wp.zeros(shape, dtype=wp.float32, device="cuda:0")  # noqa: E731
        wp.launch(
            LBFGSOptimizer._build_kernel(2, 2, 4, 32, 1, 1, int(zero_trials), 1e-5, 0.9, 1.0),
            dim=32,
            inputs=[
                last_step,
                gradient,
                zeros(1, 2),
                zeros(1, 2, 2),
                zeros(1, 2, 2),
                zeros(1, 2, 2),
                zeros(1, 2),
                wp.zeros(1, dtype=wp.int32, device="cuda:0"),
                wp.zeros(1, dtype=wp.int32, device="cuda:0"),
                wp.array(alphas, device="cuda:0"),
                wp.ones(2, dtype=wp.float32, device="cuda:0"),
                1,
                0,
                expanded_costs,
                expanded_gradient,
            ],
            outputs=[
                zeros(1, 2),
                wp.array([slope_0], dtype=wp.float32, device="cuda:0"),
                wp.array(steps, device="cuda:0"),
                costs_out,
                winner,
                wp.array([1.0e30], dtype=wp.float32, device="cuda:0"),
                wp.zeros(1, dtype=wp.int32, device="cuda:0"),
            ],
            block_dim=32,
            device="cuda:0",
        )
        assert winner.numpy()[0] == expected and costs_out.numpy()[0] == costs[expected]
        np.testing.assert_allclose(last_step.numpy()[0], steps[expected])
        np.testing.assert_allclose(gradient.numpy()[0], gradients[expected])
        # zeroed, the next line search adds its costs and J^T r into them
        np.testing.assert_array_equal(expanded_gradient.numpy() == 0.0, np.full(gradients.shape, zero_trials))
        np.testing.assert_array_equal(expanded_costs.numpy() == 0.0, np.full(costs.shape, zero_trials))

    def test_dense_gradient_ignores_stale_cpu_tile_memory(self):
        # Warp's CPU tiles live on the stack: NaN-poison it so a read of uninitialized tile memory shows up.
        rng = np.random.default_rng(0)
        J, r = rng.standard_normal((1, 6, 6)).astype(np.float32), rng.standard_normal((1, 6)).astype(np.float32)
        g = wp.zeros((1, 6), dtype=wp.float32, device="cpu")
        sink = wp.zeros(4096, dtype=wp.float32, device="cpu")
        wp.launch_tiled(_poison_cpu_tile_stack, dim=[1], inputs=[sink], block_dim=TILE_THREADS, device="cpu")
        wp.launch_tiled(
            _jtr_kernel(6, 6),
            dim=[1],
            inputs=[wp.array(J, device="cpu"), wp.array(r, device="cpu")],
            outputs=[g],
            block_dim=TILE_THREADS,
            device="cpu",
        )
        np.testing.assert_allclose(g.numpy()[0], J[0].T @ r[0], atol=1e-5)

    def test_armijo_rejects_non_decreasing_step(self, ur10_robot):
        robot = ur10_robot
        q_init = robot.spec.zero_q.astype(np.float32).reshape(1, -1)
        optimizer = LBFGSOptimizer(
            terms=[ConfigurationTask(robot=robot, rest_q=robot.spec.midrange_q)],
            config=LBFGSOptimizerConfig(max_iter=3, line_search_alphas=(1.0e6,)),
            device="cpu",
        )
        var = VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(q_init, dtype=wp.float32, device="cpu"))})

        result, costs = optimizer.solve(var)

        np.testing.assert_array_equal(result.get(VarKey.ROBOT).q.numpy(), q_init)
        assert costs.shape == (1,)

    @requires_cuda
    def test_ik_converges(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()

        config = LBFGSOptimizerConfig(max_iter=100, early_stopping_interval=1, cost_tol=1e-6)
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = LBFGSOptimizer(
            terms=[frame_task],
            config=config,
        )

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        state = robot.forward_kinematics(state)
        T_world_link = state.get_T_world_link(frame_index)

        assert np.allclose(T_world_link.numpy(), T_world_target.numpy()[:, 0], atol=1e-2)

    @requires_cuda
    def test_ik_mobile(self, panda_robot):
        """Multi-var IK: optimize both robot q and T_world_base."""
        robot = panda_robot

        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

        T_world_base_np = np.array([0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)

        q_np = robot.spec.midrange_q
        q = wp.from_numpy(q_np.reshape(1, -1), dtype=wp.float32, device=device)
        state = robot.state(q=q, T_world_base=T_world_base)

        T_world_target_np = np.array([0.6, 0.0, 0.55, 0.0, 0.707, 0.0, -0.707], dtype=np.float32).reshape(1, 7)
        T_world_target = wp.from_numpy(T_world_target_np[:, None], dtype=wp_vec7, device=device)
        frame_index = robot.link_names.index("panda_hand")

        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )
        position_limit = PositionLimit(robot=robot, weight=2.0)

        config = LBFGSOptimizerConfig(max_iter=100)
        optimizer = LBFGSOptimizer(
            terms=[frame_task, position_limit],
            device=device,
            config=config,
        )

        state = robot.state(q=q, T_world_base=T_world_base)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        state = robot.forward_kinematics(state)

        T_world_link = state.get_T_world_link(frame_index).numpy()
        T_world_target_np = T_world_target.numpy()[:, 0]
        assert np.allclose(T_world_link[:, :3], T_world_target_np[:, :3], atol=1e-2)
        assert np.all(np.abs(np.sum(T_world_link[:, 3:] * T_world_target_np[:, 3:], axis=1)) > 0.99)

    def test_ik_autodiff_gradient(self, ur10_robot):
        """Test that autodiff gradient mode converges as well as analytic."""
        robot = ur10_robot
        T_world_target = _target_pose()

        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = LBFGSOptimizerConfig(max_iter=100, early_stopping_interval=1, cost_tol=1e-6, gradient_mode="autodiff")
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = LBFGSOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        state = robot.forward_kinematics(state)
        T_world_link = state.get_T_world_link(frame_index)

        assert np.allclose(T_world_link.numpy(), T_world_target.numpy()[:, 0], atol=1e-2)
