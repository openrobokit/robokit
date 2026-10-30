import numpy as np
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from robokit.xform.warp.torch_wrappers import matrix_to_quaternion
from scripts.benchmark.ik.helper import IKBenchmarkHelper


class CuroboIKBenchmarkHelper(IKBenchmarkHelper):
    EXPECTED_TENSOR_TYPE = "torch"

    def __init__(self, batch_size: int):
        super().__init__(batch_size=batch_size, target_link_name="panda_hand")  # cuRobo's franka has no tcp frame
        tensor_args = TensorDeviceType()

        robot_file = "franka.yml"
        robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]

        robot_data["kinematics"]["collision_link_names"] = None
        robot_data["kinematics"]["lock_joints"] = {}
        robot_data["kinematics"]["collision_sphere_buffer"] = 0.0

        robot_cfg = RobotConfig.from_dict(robot_data, tensor_args=tensor_args)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_model=None,
            rotation_threshold=self.ROTATION_THRESHOLD,
            position_threshold=self.POSITION_THRESHOLD,
            num_seeds=16,  # PyRoKi's benchmark runs cuRobo with 16 seeds
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
            high_precision=False,
            regularization=False,
        )

        self.ik_solver: IKSolver = IKSolver(ik_config)
        curobo_joint_names = list(self.ik_solver.kinematics.kinematics_config.joint_names)
        self._reindex_to_reference: np.ndarray = np.array(
            [curobo_joint_names.index(name) for name in self.reference_joint_names]
        )

    def solve_ik(self, target_matrix_tensor: torch.Tensor) -> torch.Tensor:
        target_positions = target_matrix_tensor[:, :3, 3].contiguous()
        target_rotations = target_matrix_tensor[:, :3, :3].contiguous()

        q_wxyz_world_target = matrix_to_quaternion(target_rotations)

        goal_pose = Pose(position=target_positions, quaternion=q_wxyz_world_target)

        result = self.ik_solver.solve_batch(goal_pose)

        solution = result.solution
        if solution.dim() == 3:
            solution = solution[:, 0, :]

        return solution

    def reindex_joints_to_reference(self, q: np.ndarray) -> np.ndarray:
        return q[:, self._reindex_to_reference]

    def warmup(self):
        """Capture the CUDA graph at exactly the batch size we time; a short sample would capture the wrong one."""
        q_sample = self.ik_solver.sample_configs(self.batch_size)
        while q_sample.shape[0] < self.batch_size:
            q_sample = torch.cat([q_sample, self.ik_solver.sample_configs(self.batch_size)])[: self.batch_size]
        kin_state = self.ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
        for _ in range(3):
            self.ik_solver.solve_batch(goal)
            torch.cuda.synchronize()
