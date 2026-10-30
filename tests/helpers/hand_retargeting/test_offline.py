"""Regression tests for the named offline hand-retargeting pipeline."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.hand_retargeting.config import (
    HandRetargeterOfflineConfig,
    HandSpec,
    contact_config,
    refine_config,
)
from robokit.helpers.hand_retargeting.offline import HandRetargeterOffline
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.utils.ee_coord_utils import MANOPTH_HAND_COORD_SPEC
from robokit.utils.warp_utils import wp_vec7


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_URDF = _FIXTURES / "shadow_hand_right.urdf"
_TARGET_NAMES = ("index_tip", "root", "thumb_tip", "index_middle", "unused")
_SPEC = HandSpec(
    floating_base=True,
    target_names=_TARGET_NAMES,
    target_link_names={
        "root": "palm",
        "thumb_tip": "thtip",
        "index_middle": "ffmiddle",
        "index_tip": "fftip",
    },
    target_chains=(("thumb_tip",), ("index_middle", "index_tip"), ("unused",)),
    root_target_name="root",
    contact_target_names=("thumb_tip", "index_tip"),
    root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
)
_CONFIG = HandRetargeterOfflineConfig(
    target_scale=1.0,
    active_joint_names=("FFJ3", "FFJ2", "FFJ1"),
    locked_prefix_frames=1,
    solver=LMOptimizerConfig(linear_solver="cg", max_iter=1, lm_lambda=0.5, cuda_graph_mode="none"),
)
_CONFIG.task("positions").weight = 2.0
_CONFIG.task("root_position").weight = 1.0
_CONFIG.task("root_rotation").dense_task.weight = 0.2
_CONFIG.task("retargeting").position_weight = 0.5
_CONFIG.task("retargeting").angle_weight = 1.0
_CONFIG.task("velocity").weight = 0.2
_CONFIG.task("acceleration").weight = 0.1
_CONFIG.task("root_smoothness").base_weight[0:3] = 0.2
_CONFIG.task("root_smoothness").base_weight[3:6] = 0.1
_CONFIG.task("position_limit").dense_task.weight = 1.0
_CONFIG.task("rest").dense_task.weight = 0.01
_CONTACT = contact_config(_CONFIG)
_CONTACT.task("contact").weight = 2.0
_REFINE = refine_config(_CONFIG)
_REFINE.task("contact").weight = 2.0
_REFINE.task("anchor").dense_task.weight = 1.0
_REFINE.task("base_anchor").dense_task.base_weight = 1.0


class TestHandRetargeterOfflineRedesign:
    def test_tracking_without_optional_regularizers(self) -> None:
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = replace(
            _CONFIG,
            terms=[term for term in _CONFIG.terms if term.name == "positions"],
            solver=replace(_CONFIG.solver, max_iter=0),
        )
        points = np.zeros((1, 3, len(_TARGET_NAMES), 3), dtype=np.float32)
        result = HandRetargeterOffline(config, robot, _SPEC, device="cpu").solve_numpy(points)
        assert result.shape == (1, 3, 7 + robot.spec.num_actuated_joints)
        assert np.isfinite(result).all()

    def test_padded_contact_ignores_invalid_targets(self) -> None:
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = replace(
            _CONTACT,
            solver=LMOptimizerConfig(linear_solver="cg", cg_max_iter=16, max_iter=3, cuda_graph_mode="none"),
        )
        rng = np.random.default_rng(13)
        points = rng.normal(0.0, 0.03, (2, 5, len(_TARGET_NAMES), 3)).astype(np.float32)
        points[:, :, _TARGET_NAMES.index("root"), 2] += 0.2
        points[0, 3:] = points[0, 2]
        contact_indices = [_TARGET_NAMES.index(name) for name in _SPEC.contact_target_names]
        contacts = points[:, :, contact_indices].copy()
        masks = np.ones((2, 5, 2), dtype=np.bool_)
        masks[0, 3:] = False

        batched = HandRetargeterOffline(config, robot, _SPEC, device="cpu")
        batched.warmup(2, 5)
        batched.set_valid_lengths(np.array([3, 5]))
        expected = batched.solve_with_contact_numpy(points, contacts, masks)
        changed_points = points.copy()
        changed_contacts = contacts.copy()
        changed_points[0, 3:] += 5.0
        changed_contacts[0, 3:] -= 5.0
        actual = batched.solve_with_contact_numpy(changed_points, changed_contacts, masks)

        np.testing.assert_allclose(actual[0, :3], expected[0, :3], atol=1e-5, rtol=0.0)

    def test_single_chunk_uses_init_state(self) -> None:
        """Preserve a supplied state in one non-optimizing public NumPy solve."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = replace(
            _CONFIG,
            solver=LMOptimizerConfig(linear_solver="cg", max_iter=0, lm_lambda=0.5, cuda_graph_mode="none"),
            terms=deepcopy(_CONFIG.terms),
        )
        config.task("root_rotation").dense_task.weight = 0.0
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 3, 1))
        base = np.zeros((1, 3, 7), dtype=np.float32)
        base[:, :, :3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )
        points = np.zeros((3, len(_TARGET_NAMES), 3), dtype=np.float32)

        result = HandRetargeterOffline(config, robot, _SPEC, device="cpu").solve_numpy(points, init_state=init_state)

        np.testing.assert_array_equal(result[:, :7], base[0])
        np.testing.assert_array_equal(result[:, 7:], q[0])

    def test_non_contact_spec_discovers_hand_joints(self) -> None:
        """Derive hand DOFs from mapped links even when contacts are disabled."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = replace(
            _CONFIG,
            solver=LMOptimizerConfig(linear_solver="cg", max_iter=0, lm_lambda=0.5, cuda_graph_mode="none"),
            terms=deepcopy(_CONFIG.terms),
        )
        config.task("root_rotation").dense_task.weight = 0.0
        config.task("velocity").weight = 0.0
        config.task("acceleration").weight = 0.0
        config.task("root_smoothness").base_weight[0:3] = 0.0
        config.task("root_smoothness").base_weight[3:6] = 0.0
        config.task("position_limit").dense_task.weight = 0.0
        config.task("rest").dense_task.weight = 0.0
        retargeter = HandRetargeterOffline(config, robot, replace(_SPEC, contact_target_names=()), device="cpu")
        result = retargeter.solve_numpy(np.zeros((1, 3, len(_TARGET_NAMES), 3), dtype=np.float32))
        indices = [robot.spec.actuated_joint_names.index(name) for name in ("FFJ1", "FFJ2", "FFJ3")]
        expected = np.mean(robot.spec.actuated_joint_limits[indices], axis=1)

        np.testing.assert_allclose(result[0, :, 7:][:, indices], np.tile(expected, (3, 1)))
        assert np.isfinite(result).all()

    def test_named_normal_contact_and_refinement(self) -> None:
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 3, 1))
        T_world_base = np.zeros((1, 3, 7), dtype=np.float32)
        T_world_base[:, :, 3] = 1.0
        state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(T_world_base, dtype=wp_vec7, device="cpu"),
        )
        robot.forward_kinematics(state)
        T_world_link = state.T_world_link.numpy()
        link_index = {name: index for index, name in enumerate(robot.spec.link_names)}
        points = np.zeros((1, 3, len(_TARGET_NAMES), 3), dtype=np.float32)
        for target, link in _SPEC.target_link_names.items():
            points[:, :, _TARGET_NAMES.index(target)] = T_world_link[:, :, link_index[link], :3]
        q_wxyz_world_palm = T_world_link[:, :, link_index["palm"], 3:].copy()
        contact_points = np.stack(
            [
                T_world_link[:, :, link_index["thtip"], :3],
                T_world_link[:, :, link_index["fftip"], :3],
            ],
            axis=2,
        ).astype(np.float32)
        contact_points += np.array([0.002, -0.001, 0.001], dtype=np.float32)
        contact_mask = np.ones((1, 3, 2), dtype=np.bool_)

        normal = HandRetargeterOffline(_CONFIG, robot, _SPEC, device="cpu").solve_numpy(points, q_wxyz_world_palm)
        contact = HandRetargeterOffline(_CONTACT, robot, _SPEC, device="cpu").solve_with_contact_numpy(
            points, contact_points, contact_mask
        )
        refined = HandRetargeterOffline(_REFINE, robot, _SPEC, device="cpu").solve_with_contact_numpy(
            points,
            contact_points,
            contact_mask,
            init_state=state,
        )

        assert normal.shape == contact.shape == refined.shape == (1, 3, 7 + robot.spec.num_actuated_joints)
        assert np.isfinite(normal).all()
        assert np.isfinite(contact).all()
        assert np.isfinite(refined).all()

    def test_fixed_base_keeps_init_state_base(self) -> None:
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        spec = replace(_SPEC, floating_base=False)
        config = replace(_CONFIG, solver=LMOptimizerConfig(linear_solver="cg", max_iter=0, lm_lambda=0.5))
        points = np.zeros((1, 3, len(_TARGET_NAMES), 3), dtype=np.float32)
        q_wxyz_world_palm = np.zeros((1, 3, 4), dtype=np.float32)
        q_wxyz_world_palm[:, :, 0] = 1.0
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 3, 1))
        base = np.zeros((1, 3, 7), dtype=np.float32)
        base[:, :, :3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )

        retargeter = HandRetargeterOffline(config, robot, spec, device="cpu")
        result = retargeter.solve_numpy(points, q_wxyz_world_palm, init_state=init_state)

        np.testing.assert_array_equal(result[:, :, :7], base)

    def test_caller_owned_chunks(self) -> None:
        """Split, stitch, and adjust the global locked prefix in caller code."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        rng = np.random.default_rng(7)
        points = rng.normal(0.0, 0.04, (1, 5, len(_TARGET_NAMES), 3)).astype(np.float32)
        chunk_config = replace(_CONFIG, terms=deepcopy(_CONFIG.terms))
        chunk_config.task("positions").weight = 0.0
        chunk_config.task("root_position").weight = 0.0
        chunk_config.task("root_rotation").dense_task.weight = 0.0
        chunk_config.task("velocity").weight = 0.0
        chunk_config.task("acceleration").weight = 0.0
        chunk_config.task("root_smoothness").base_weight[0:3] = 0.0
        chunk_config.task("root_smoothness").base_weight[3:6] = 0.0
        chunk_config.task("rest").dense_task.weight = 0.0
        chunks = []
        for start, end, keep_start in ((0, 3, 0), (2, 5, 1)):
            helper = HandRetargeterOffline(chunk_config, robot, _SPEC, device="cpu")
            chunk = helper.solve_numpy(points[:, start:end])
            chunks.append(chunk[:, keep_start:])
        result = np.concatenate(chunks, axis=1)
        assert result.shape == (1, 5, 7 + robot.spec.num_actuated_joints)
        assert np.isfinite(result).all()

        chunk_refine = replace(refine_config(chunk_config), optimize_base=False)
        chunk_refine.task("contact").weight = 2.0
        chunk_refine.task("anchor").dense_task.weight = 1.0
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, 5, 1))
        base = np.zeros((1, 5, 7), dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )
        contact_points = rng.normal(0.0, 0.02, (1, 5, 2, 3)).astype(np.float32)
        contact_mask = np.ones((1, 5, 2), dtype=np.bool_)
        local_points = np.full((1, 5, 2, 3), 0.001, dtype=np.float32)
        chunks = []
        for start, end, locked_prefix, keep_start in ((0, 3, 1, 0), (2, 5, 0, 1)):
            chunk_state = robot.state(
                q=init_state.q[:, start:end].contiguous(),
                T_world_base=init_state.T_world_base[:, start:end].contiguous(),
            )
            chunk_helper = HandRetargeterOffline(
                replace(chunk_refine, locked_prefix_frames=locked_prefix),
                robot,
                _SPEC,
                device="cpu",
            )
            chunk = chunk_helper.solve_with_contact_numpy(
                points[:, start:end],
                contact_points[:, start:end],
                contact_mask[:, start:end],
                init_state=chunk_state,
                local_contact_points=local_points[:, start:end],
            )
            chunks.append(chunk[:, keep_start:])
        result = np.concatenate(chunks, axis=1)
        assert result.shape == (1, 5, 7 + robot.spec.num_actuated_joints)
        np.testing.assert_array_equal(result[:, 0, 7:], q[:, 0])


class TestHandRetargeterOfflineRefinement:
    def test_surface_contact_uses_independent_mask(self, panda_robot_with_collision: Robot) -> None:
        num_frames = 3
        robot = panda_robot_with_collision
        sphere_index = int(np.flatnonzero(robot.spec.collision_spheres_link_indices > 0)[0])
        contact_link = robot.spec.link_names[robot.spec.collision_spheres_link_indices[sphere_index]]
        spec = HandSpec(
            floating_base=False,
            target_names=("root", "contact"),
            target_link_names={"root": "panda_link7", "contact": contact_link},
            target_chains=(("root", "contact"),),
            root_target_name="root",
            contact_target_names=("contact",),
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
            contact_sphere_indices=(sphere_index,),
        )
        config = refine_config(
            HandRetargeterOfflineConfig(
                optimize_base=False,
                solver=LMOptimizerConfig(linear_solver="cg", max_iter=0, lm_lambda=0.5, cuda_graph_mode="none"),
            )
        )
        for name in ("velocity", "acceleration", "root_smoothness", "position_limit", "contact", "collision"):
            config.terms = [term for term in config.terms if term.name != name]
        config.task("surface_contact").weight = 10.0
        config.task("anchor").dense_task.weight = 0.0
        config.task("base_anchor").dense_task.base_weight = 0.0
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = 2.0
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        scene_indices = wp.from_numpy(np.zeros((1, num_frames), dtype=np.int32), dtype=wp.int32, device="cpu")
        helper = HandRetargeterOffline(config, robot, spec, scene=scene, device="cpu", scene_indices=scene_indices)
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, num_frames, 1))
        base = np.zeros((1, num_frames, 7), dtype=np.float32)
        base[:, :, 3] = 1.0
        state = robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )
        points = np.zeros((num_frames, 2, 3), dtype=np.float32)
        contacts = np.zeros((num_frames, 1, 3), dtype=np.float32)
        point_mask = np.ones((num_frames, 1), dtype=np.bool_)

        helper.solve_with_contact_numpy(
            points,
            contacts,
            point_mask,
            init_state=state,
            surface_contact_mask=np.zeros_like(point_mask),
        )
        masked_cost = helper._optimizer.costs.numpy()[0]
        helper.solve_with_contact_numpy(
            points,
            contacts,
            point_mask,
            init_state=state,
            surface_contact_mask=point_mask,
        )
        active_cost = helper._optimizer.costs.numpy()[0]

        assert masked_cost == 0.0
        assert active_cost > 0.0

    def test_caller_owned_scene_collision_is_active(self, panda_robot_with_collision: Robot) -> None:
        """Propagate caller scene updates into anchored collision refinement."""
        num_frames = 3
        spec = HandSpec(
            floating_base=False,
            target_names=("root", "contact"),
            target_link_names={"root": "panda_link7", "contact": "panda_hand"},
            target_chains=(("root", "contact"),),
            root_target_name="root",
            contact_target_names=("contact",),
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
        )
        config = refine_config(
            HandRetargeterOfflineConfig(
                optimize_base=False,
                solver=LMOptimizerConfig(linear_solver="cg", max_iter=0, lm_lambda=0.5, cuda_graph_mode="none"),
            )
        )
        config.terms = [term for term in config.terms if term.name == "collision"]
        config.task("collision").weight = 10.0
        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = (10.0, 0.0, 0.0)
        box = BoxGeom(
            np.array([[0.01, 0.01, 0.01]], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            poses=wp.from_numpy(far[None], dtype=wp.mat44, device="cpu"),
        )
        scene = WarpScene(1, "cpu").add(box)
        helper = HandRetargeterOffline(config, panda_robot_with_collision, spec, scene=scene, device="cpu")
        q = np.tile(panda_robot_with_collision.spec.midrange_q.astype(np.float32), (1, num_frames, 1))
        base = np.zeros((1, num_frames, 7), dtype=np.float32)
        base[:, :, 3] = 1.0
        init_state = panda_robot_with_collision.state(
            q=wp.from_numpy(q, dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(base, dtype=wp_vec7, device="cpu"),
        )
        points = np.zeros((num_frames, 2, 3), dtype=np.float32)
        contacts = np.zeros((num_frames, 1, 3), dtype=np.float32)
        mask = np.zeros((num_frames, 1), dtype=np.bool_)

        helper.solve_with_contact_numpy(points, contacts, mask, init_state=init_state)
        far_cost = helper._optimizer.costs.numpy()[0]
        panda_robot_with_collision.forward_kinematics(init_state)
        panda_robot_with_collision.transform_collision_spheres(init_state)
        assert len(helper._collision_sphere_indices)
        sphere_index = helper._collision_sphere_indices[0]
        near = np.eye(4, dtype=np.float32)
        near[:3, 3] = init_state.collision_sphere_centers_world.numpy()[0, 0, sphere_index]
        box.update(poses=wp.from_numpy(near[None], dtype=wp.mat44, device="cpu"))
        helper.solve_with_contact_numpy(points, contacts, mask, init_state=init_state)
        near_cost = helper._optimizer.costs.numpy()[0]

        assert far_cost == 0.0
        assert near_cost > 0.0

    def test_fixed_base_and_start_do_not_move(self, panda_robot: Robot) -> None:
        """Keep inactive bases and locked trajectory prefixes unchanged."""
        num_frames = 3
        q = np.tile(panda_robot.spec.midrange_q.astype(np.float32), (num_frames, 1))
        q[1, 0] += 0.2
        T_world_base = np.array(
            [
                [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.3, 0.9238795, 0.0, 0.0, 0.3826834],
                [0.3, 0.2, 0.1, 0.9659258, 0.0, 0.2588190, 0.0],
            ],
            dtype=np.float32,
        )
        points = np.zeros((num_frames, 2, 3), dtype=np.float32)
        contact_points = np.zeros((num_frames, 1, 3), dtype=np.float32)
        contact_mask = np.array([[False], [False], [True]])
        spec = HandSpec(
            floating_base=False,
            target_names=("root", "contact"),
            target_link_names={"root": "panda_link7", "contact": "panda_hand"},
            target_chains=(("root", "contact"),),
            root_target_name="root",
            contact_target_names=("contact",),
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
        )
        config = refine_config(
            HandRetargeterOfflineConfig(
                optimize_base=False,
                locked_prefix_frames=2,
                solver=LMOptimizerConfig(linear_solver="cg", max_iter=2, lm_lambda=0.5, cuda_graph_mode="none"),
            )
        )
        config.task("velocity").weight = 0.0
        config.task("acceleration").weight = 0.0
        config.task("position_limit").dense_task.weight = 0.0
        init_state = panda_robot.state(
            q=wp.from_numpy(q[None], dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(T_world_base[None], dtype=wp_vec7, device="cpu"),
        )

        refined = HandRetargeterOffline(config, panda_robot, spec, device="cpu").solve_with_contact_numpy(
            points,
            contact_points,
            contact_mask,
            init_state=init_state,
        )

        np.testing.assert_array_equal(refined[:2, 7:], q[:2])
        np.testing.assert_array_equal(refined[:, :7], T_world_base)
