# pyright: reportCallIssue=false
from typing import Any, List, Tuple

import jax
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from jax import lax
from jax import numpy as jnp
from loguru import logger

from scripts.benchmark.ik.helper import IKBenchmarkHelper, load_panda_urdf_fixed


logger.disable("jaxls")


def newton_raphson(f: Any, x: float, iters: int) -> jax.Array:
    def update(value: jax.Array, _: Any) -> Tuple[jax.Array, None]:
        y: jax.Array = value - f(value) / jax.grad(f)(value)
        return y, None

    root, _ = lax.scan(update, jnp.asarray(x), length=iters)
    return root


def roberts_sequence(num_points: int, dim: int, root: jax.Array) -> jax.Array:
    basis: jax.Array = 1 - (1 / root ** (1 + jnp.arange(dim)))
    n: jax.Array = jnp.arange(num_points)
    x: jax.Array = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)
    return x


class PyrokiIKBenchmarkHelper(IKBenchmarkHelper):
    EXPECTED_TENSOR_TYPE = "jax"

    def __init__(self, batch_size: int):
        super().__init__(batch_size=batch_size)
        self.pyroki_robot: pk.Robot = pk.Robot.from_urdf(load_panda_urdf_fixed())
        self.pyroki_target_link_index: jax.Array = jnp.array(self.pyroki_robot.links.names.index(self.target_link_name))
        exp: int = self.pyroki_robot.joints.num_actuated_joints
        self.root: jax.Array = newton_raphson(lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000)
        self.batched_ik: Any = None

        pyroki_all_joints = list(self.pyroki_robot.joints.names)
        pyroki_actuated_joints = [name for name in pyroki_all_joints if name in self.reference_joint_names]
        self._reindex_to_reference = np.array(
            [pyroki_actuated_joints.index(name) for name in self.reference_joint_names]
        )

    def _solve_ik_single(self, target_wxyz: jax.Array, target_position: jax.Array) -> jax.Array:
        num_candidates_init: int = 64
        num_candidates_final: int = 4
        total_steps: int = 16
        init_steps: int = 6

        def solve_one(initial_q: jax.Array, lambda_initial: jax.Array, max_iters: int) -> Tuple[jax.Array, Any]:
            joint_var: Any = self.pyroki_robot.joint_var_cls(0)
            factors: List[Any] = [
                pk.costs.pose_cost_analytic_jac(
                    self.pyroki_robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
                    self.pyroki_target_link_index,
                    pos_weight=10.0,
                    ori_weight=5.0,
                ),
                pk.costs.limit_cost(self.pyroki_robot, joint_var, weight=50.0),
            ]
            sol: Any
            summary: Any
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(initial_q)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(max_iterations=max_iters, early_termination=False),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve: Any = jax.vmap(solve_one, in_axes=(0, 0, None))

        robot: pk.Robot = self.pyroki_robot
        initial_qs: jax.Array = robot.joints.lower_limits + roberts_sequence(
            num_candidates_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        initial_sols, summary = vmapped_solve(initial_qs, jnp.full(initial_qs.shape[:1], 10.0), init_steps)
        best_initial_sols = jnp.argsort(summary.cost_history[jnp.arange(num_candidates_init), -1])[
            :num_candidates_final
        ]

        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(num_candidates_init), -1][best_initial_sols],
            total_steps - init_steps,
        )
        return best_sols[jnp.argmin(summary.cost_history[jnp.arange(num_candidates_final), summary.iterations])]

    def solve_ik(self, target_arrays: Tuple[jax.Array, jax.Array]) -> jax.Array:
        return self.batched_ik(*target_arrays)

    def reindex_joints_to_reference(self, q: np.ndarray) -> np.ndarray:
        return q[:, self._reindex_to_reference]

    def warmup(self):
        self.batched_ik = jax.jit(jax.vmap(self._solve_ik_single))
        dummy_wxyz = jnp.zeros((self.batch_size, 4))
        dummy_wxyz = dummy_wxyz.at[:, 0].set(1.0)
        dummy_position = jnp.zeros((self.batch_size, 3))
        solution = self.batched_ik(dummy_wxyz, dummy_position)
        jax.block_until_ready(solution)
