import pytest
import torch


pytestmark = pytest.mark.torch


class TestMapToFullJointValues:
    def test_map_to_full_joint_values_consistency(self, panda_robot):
        robot = panda_robot

        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_actuated = robot.spec.num_actuated_joints

        torch.manual_seed(42)

        for shape in [(num_actuated,), (4, num_actuated), (2, 10, num_actuated)]:
            actuated_values = torch.randn(shape, device=device, dtype=torch.float32)
            full_values = robot.map_to_full_joint_values_torch(actuated_values)

            assert full_values.shape[-1] == robot.spec.num_joints
