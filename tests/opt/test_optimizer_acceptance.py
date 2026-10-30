from typing import Tuple

import numpy as np
import warp as wp

from robokit.opt.lm_optimizer import LMOptimizerConfig, lm_accept_reject_cost_kernel


class TestWarpOptimizerAcceptance:
    @staticmethod
    def _device():
        wp.init()
        return wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

    def _run_dense_update_step(
        self,
        cost_curr: float,
        cost_prop: float,
        pred_red: float,
        rho_min: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        device = self._device()
        costs_curr = wp.from_numpy(np.array([cost_curr], dtype=np.float32), dtype=wp.float32, device=device)
        costs_prop = wp.from_numpy(np.array([cost_prop], dtype=np.float32), dtype=wp.float32, device=device)
        pred_reduction = wp.from_numpy(np.array([pred_red], dtype=np.float32), dtype=wp.float32, device=device)
        lambda_values = wp.from_numpy(np.array([1.0], dtype=np.float32), dtype=wp.float32, device=device)
        accept_mask = wp.zeros((1,), dtype=wp.int32, device=device)

        wp.launch(
            kernel=lm_accept_reject_cost_kernel,
            dim=1,
            inputs=[costs_curr, costs_prop, pred_reduction, rho_min, 1e-8, 2.0, 1e-6, 1e6],
            outputs=[lambda_values, accept_mask],
            device=device,
        )
        wp.synchronize_device(device)
        cost_out = costs_curr.numpy()
        accept = accept_mask.numpy()
        return cost_out, np.array([[0.3]], dtype=np.float32), accept

    def test_dense_rejects_uphill_step(self):
        cost, _delta, accept = self._run_dense_update_step(
            cost_curr=1.0,
            cost_prop=1.1,
            pred_red=-0.05,
        )
        assert accept[0] == 0
        assert np.isclose(cost[0], 1.0, atol=1e-6)

    def test_dense_rejects_negative_predicted_reduction(self):
        cost, _delta, accept = self._run_dense_update_step(
            cost_curr=1.0,
            cost_prop=0.8,
            pred_red=-0.05,
        )
        assert accept[0] == 0
        assert np.isclose(cost[0], 1.0, atol=1e-6)

    def test_dense_accepts_valid_downhill_step(self):
        cost, _delta, accept = self._run_dense_update_step(
            cost_curr=1.0,
            cost_prop=0.8,
            pred_red=0.2,
        )
        assert accept[0] == 1
        assert np.isclose(cost[0], 0.8, atol=1e-5)

    def test_gain_ratio_epsilon_config_defaults(self):
        assert LMOptimizerConfig().gain_ratio_epsilon == 1e-8
