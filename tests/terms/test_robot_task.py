import pytest

from robokit.lie.se3 import se3_identity
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotCollisionState, RobotState
from robokit.terms.composite_task import CompositeTask
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.robot_task import RobotTask


class TestRobotTaskPrecompute:
    def test_composite_score_precomputes_children(self, panda_robot: Robot):
        state = panda_robot.state()
        target = se3_identity(shape=(1, 1), device=state.device)
        score = CompositeTask([PositionTask(panda_robot, 0, target), RotationTask(panda_robot, 0, target)])

        score.precompute(VarValues({VarKey.ROBOT: state}), need_gradient=True)

        assert state.is_fk_computed
        assert state.is_motion_subspace_computed

    def test_gradient_phase_adds_motion_subspace(self, panda_robot: Robot, monkeypatch: pytest.MonkeyPatch):
        state = panda_robot.state()
        task = RobotTask()
        task.robot = panda_robot
        task.var_key = VarKey.ROBOT
        values = VarValues({VarKey.ROBOT: state})
        fk_calls = 0
        motion_calls = 0
        forward_kinematics = panda_robot.forward_kinematics
        compute_motion_subspace = panda_robot.compute_motion_subspace

        def count_fk(state_arg: RobotState) -> RobotState:
            nonlocal fk_calls
            fk_calls += 1
            return forward_kinematics(state_arg)

        def count_motion(state_arg: RobotState) -> RobotState:
            nonlocal motion_calls
            motion_calls += 1
            return compute_motion_subspace(state_arg)

        monkeypatch.setattr(panda_robot, "forward_kinematics", count_fk)
        monkeypatch.setattr(panda_robot, "compute_motion_subspace", count_motion)

        task.precompute(values)
        task.precompute(values)
        assert state.is_fk_computed
        assert not state.is_motion_subspace_computed

        task.precompute(values, need_gradient=True)
        task.precompute(values, need_gradient=True)
        assert state.is_motion_subspace_computed
        assert fk_calls == 2  # the subspace comes from the parallel FK, rerun once its buffer exists
        assert motion_calls == 1

    def test_collision_geometry_is_prepared_once(
        self, panda_robot_with_collision: Robot, monkeypatch: pytest.MonkeyPatch
    ):
        state = panda_robot_with_collision.state()
        task = RobotTask()
        task.robot = panda_robot_with_collision
        task.var_key = VarKey.ROBOT
        task.precompute_collision_geometry = "sphere"
        calls = 0
        transform_collision_spheres = panda_robot_with_collision.transform_collision_spheres

        def count_transform(state_arg: RobotCollisionState) -> RobotCollisionState:
            nonlocal calls
            calls += 1
            return transform_collision_spheres(state_arg)

        monkeypatch.setattr(panda_robot_with_collision, "transform_collision_spheres", count_transform)

        task.precompute(VarValues({VarKey.ROBOT: state}))
        task.precompute(VarValues({VarKey.ROBOT: state}))
        assert state.is_fk_computed
        assert state.is_collision_spheres_computed
        assert calls == 1
