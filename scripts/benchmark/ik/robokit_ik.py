import warp as wp

from robokit.helpers.ik import IKSolver, presets
from robokit.lie.se3 import se3_identity
from robokit.robo import Robot, RobotState
from scripts.benchmark.ik.helper import IKBenchmarkHelper, load_panda_urdf_fixed


class RoboKitWarpIKBenchmarkHelper(IKBenchmarkHelper):
    EXPECTED_TENSOR_TYPE = "warp"

    def __init__(self, batch_size: int):
        super().__init__(batch_size=batch_size)
        self.out_state_buffer: RobotState
        self.warp_robot = Robot.load(load_panda_urdf_fixed())
        self.warp_helper = IKSolver(
            presets.basic, robot=self.warp_robot, target_links=self.target_link_name, device="cuda"
        )
        self.warp_helper.warmup(batch_size=batch_size)

    def solve_ik(self, T_world_target: wp.array) -> wp.array:
        self.warp_helper.solve(T_world_target, out_state=self.out_state_buffer)
        return wp.clone(self.out_state_buffer.q)

    def warmup(self):
        self.out_state_buffer = self.warp_robot.state(
            q=wp.empty((self.batch_size, self.warp_helper.num_joints), dtype=wp.float32, device="cuda")
        )
        self.solve_ik(se3_identity((self.batch_size, 1), device="cuda"))
        wp.synchronize()
