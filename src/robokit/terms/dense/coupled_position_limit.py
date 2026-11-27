# pyright: reportArgumentType=false
from typing import Optional, Sequence, Tuple

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_device_type


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _coupled_position_limit_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, n_dofs]
    joints: wp.array2d(dtype=wp.int32),  # [cuts, 2] i, j
    planes: wp.array2d(dtype=wp.float32),  # [cuts, 3] c_i, c_j, b
    weight: float,
    row_offset: int,
    col_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim], or null
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs], or null
):
    batch_idx, c = wp.tid()  # pyright: ignore
    i = joints[c, 0]
    j = joints[c, 1]
    c_i = planes[c, 0]
    c_j = planes[c, 1]
    excess = c_i * q[batch_idx, i] + c_j * q[batch_idx, j] - planes[c, 2]
    if out_residual:
        out_residual[batch_idx, row_offset + c] = weight * wp.max(excess, wp.float32(0.0))
    if out_jacobian and excess > wp.float32(0.0):
        out_jacobian[batch_idx, row_offset + c, col_offset + i] = weight * c_i
        out_jacobian[batch_idx, row_offset + c, col_offset + j] = weight * c_j


# --- task -------------------------------------------------------------------
class CoupledPositionLimit(RobotTask, ResidualTask):
    """Penalize joint pairs whose ranges depend on each other, such as two wrist joints that collide.

    Each cut `(joint_i, joint_j, c_i, c_j, b)` keeps `c_i q_i + c_j q_j <= b`, with joint names and
    `q` in rad. Cuts on the same pair intersect, so with the joints' own limits the pair stays in a
    convex polygon, such as a box with some corners cut. A symmetric limit `|q_j| + k |q_i| <= a` is
    four cuts, one per sign pair. One residual row per cut.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey, VarValues
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> q = robot.spec.midrange_q.astype(np.float32)
        >>> q[5], q[6] = 1.0, -1.0
        >>> state = robot.state(q=wp.from_numpy(q[None], dtype=wp.float32))
        >>> # |q7| + 0.5 |q6| <= 1; only the (+q6, -q7) cut is violated: 0.5 * 1 + 1 - 1
        >>> cuts = [("panda_joint6", "panda_joint7", 0.5 * si, sj, 1.0) for si in (-1, 1) for sj in (-1, 1)]
        >>> task = CoupledPositionLimit(cuts, robot=robot)
        >>> task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy().tolist()
        [[0.0, 0.0, 0.5, 0.0]]
        >>> task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()[0, 2, 5:7].tolist()
        [0.5, -1.0]
    """

    def __init__(
        self,
        cuts: Sequence[Tuple[str, str, float, float, float]],
        robot: Optional[Robot] = None,
        weight: float = 1.0,
        name: str = "",
    ):
        self.name = name
        self.cuts = [tuple(c) for c in cuts]
        if any(c[0] == c[1] for c in self.cuts):
            raise ValueError("Each cut must couple two different joints.")
        self.weight = weight
        self.robot: Optional[Robot] = None
        self.device = None
        if robot is not None:
            self.set_robot(robot)

    def set_robot(self, robot: Robot):
        self.robot = robot
        names = robot.spec.actuated_joint_names
        self._joints_np = np.array([[names.index(c[0]), names.index(c[1])] for c in self.cuts], dtype=np.int32)
        self._planes_np = np.array([c[2:] for c in self.cuts], dtype=np.float32)

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self._joints = wp.from_numpy(self._joints_np, dtype=wp.int32, device=device)
        self._planes = wp.from_numpy(self._planes_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return len(self.cuts)

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if self.device is None:
            self.init_buffers(var.q.device)
        if out_residual is None:
            out_residual = wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=self.device)
            row_offset = 0
        wp.launch(
            _coupled_position_limit_kernel,
            dim=(var.batch_size, self.residual_dim),
            inputs=[var.q, self._joints, self._planes, float(self.weight), row_offset, 0],
            outputs=[out_residual, None],
            device=self.device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if self.device is None:
            self.init_buffers(var.q.device)
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (var.batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=self.device
            )
            row_offset, col_offset = 0, 0
        wp.launch(
            _coupled_position_limit_kernel,
            dim=(var.batch_size, self.residual_dim),
            inputs=[
                var.q,
                self._joints,
                self._planes,
                float(self.weight),
                row_offset,
                col_offset + (6 if var.has_floating_base else 0),
            ],
            outputs=[None, out_jacobian],
            device=self.device,
        )
        return out_jacobian
