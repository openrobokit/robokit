import numpy as np
import pytest
import warp as wp

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.coupled_position_limit import CoupledPositionLimit
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.utils.warp_utils import wp_vec7


# |q7| <= 2.5 - 0.5 |q6| on the Panda wrist, as four cuts c_i q6 + c_j q7 <= b
_CUTS = [("panda_joint6", "panda_joint7", 0.5 * si, sj, 2.5) for si in (-1.0, 1.0) for sj in (-1.0, 1.0)]
_PLANES = np.array([c[2:] for c in _CUTS])


def _excess(q: np.ndarray) -> np.ndarray:
    return q[..., 5:6] * _PLANES[:, 0] + q[..., 6:7] * _PLANES[:, 1] - _PLANES[:, 2]


class TestCoupledPositionLimit:
    @pytest.mark.parametrize("floating_base", [False, True])
    def test_residual_and_jacobian_match_finite_differences(self, panda_robot, device, floating_base: bool):
        robot = panda_robot
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (8, 1))
        # every quadrant, half inside and half outside the cuts
        q[:, 5] = [1.0, -1.0, 3.0, -3.0] * 2
        q[:, 6] = [-0.5] * 4 + [2.5, -2.5, 2.5, -2.5]
        T_world_base = wp.array(np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (8, 1)), dtype=wp_vec7, device=device)

        def var(q: np.ndarray) -> VarValues:
            q_wp = wp.from_numpy(q.astype(np.float32), dtype=wp.float32, device=device)
            state = robot.state(q=q_wp, T_world_base=T_world_base) if floating_base else robot.state(q=q_wp)
            return VarValues({VarKey.ROBOT: state})

        task = CoupledPositionLimit(_CUTS, robot=robot, weight=2.0)
        residual = task.compute_weighted_residual(var(q)).numpy()
        jacobian = task.compute_weighted_jacobian(var(q)).numpy()
        np.testing.assert_allclose(residual, 2.0 * np.maximum(_excess(q), 0.0), atol=1e-6)
        # one cut per corner, so a point outside violates exactly one
        assert (residual[:4] == 0).all() and ((residual[4:] > 0).sum(axis=1) == 1).all()
        # callers reuse residual buffers without zeroing them, so every row is rewritten
        out = wp.full((8, 6), 7.0, dtype=wp.float32, device=device)
        task.compute_weighted_residual(var(q), out_residual=out, row_offset=2)
        np.testing.assert_array_equal(out.numpy(), np.concatenate([np.full((8, 2), 7.0), residual], axis=1))

        offset = 6 if floating_base else 0
        assert jacobian.shape == (8, 4, offset + robot.num_actuated_joints)
        np.testing.assert_array_equal(jacobian[..., :offset], 0.0)
        eps = 1e-3
        for d in range(robot.num_actuated_joints):
            step = np.zeros_like(q)
            step[:, d] = eps
            plus = task.compute_weighted_residual(var(q + step)).numpy()
            minus = task.compute_weighted_residual(var(q - step)).numpy()
            np.testing.assert_allclose(jacobian[..., offset + d], (plus - minus) / (2 * eps), atol=1e-2)

        with pytest.raises(ValueError):
            CoupledPositionLimit([("panda_joint6", "panda_joint6", 1.0, 1.0, 1.0)])

    def test_ik_moves_a_violating_seed_inside(self, panda_robot, device):
        """A position-only IK seeded on a violating solution: box limits keep it there, the cuts move it."""
        robot = panda_robot
        q_seed = robot.spec.midrange_q.astype(np.float32)[None]
        q_seed[0, 5], q_seed[0, 6] = 3.0, 2.0  # inside the box, 1.0 over the (+, +) cut

        def seed():
            return robot.state(q=wp.from_numpy(q_seed, dtype=wp.float32, device=device))

        ee = robot.spec.link_names.index("panda_hand")
        T_world_target = robot.forward_kinematics(seed()).T_world_link.numpy()[0, ee]

        def solve(*terms) -> np.ndarray:
            config = (
                IKSolverConfig(solver=PopulationSolverConfig(stages=[Stage(50, 1, LMOptimizerConfig())]))
                .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
                .add(PositionLimit(weight=50.0))
            )
            for term in terms:
                config = config.add(term)
            ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device)
            state = ik.solve_numpy(T_world_target, init_state=seed())
            assert ik.compute_metrics_numpy(state, T_world_target, 0.005, np.pi).success.all()
            return state.q.numpy()

        assert _excess(solve()).max() > 0.9
        assert _excess(solve(CoupledPositionLimit(_CUTS, weight=50.0))).max() < 1e-3
