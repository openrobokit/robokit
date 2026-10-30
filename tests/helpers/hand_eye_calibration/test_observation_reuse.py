"""Repeated observations rebuild eager fits without retaining stale inputs."""

from dataclasses import replace

import pytest
import warp as wp

from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from tests.helpers.hand_eye_calibration.test_hec import (
    _H,
    _REQUIRES_CUDA,
    _W,
    _perturbed,
    hec_config,
    hec_module,
    mask_task,
    torch,
)
from tests.helpers.hand_eye_calibration.test_hec import (
    toy_scene as toy_scene,
)


pytestmark = pytest.mark.torch


@_REQUIRES_CUDA
class TestObservationReuse:
    @pytest.mark.parametrize("external_stream", [False, True])
    def test_antialias_precision_and_native_gradients(self, external_stream: bool):
        wp.init()
        caller = torch.cuda.Stream() if external_stream else torch.cuda.current_stream()
        with torch.cuda.stream(caller):
            positions = torch.tensor(
                [[[-0.8, -0.9, 0, 1], [0.1, 0.7, 0, 1], [0.7, -0.4, 0, 1], [0.9, 0.6, 0, 1]]],
                device="cuda",
                requires_grad=True,
            )
            triangles = torch.tensor([[0, 1, 2], [1, 3, 2]], device="cuda", dtype=torch.int32)
            raster, _ = mask_task.dr.rasterize(mask_task._rasterize_context(0), positions, triangles, (64, 64))
            color, _ = mask_task.dr.interpolate(torch.ones((1, 4, 1), device="cuda"), raster, triangles)
            native = mask_task.dr.antialias(color, raster, positions, triangles)
            precise = mask_task._antialias(color, raster, positions, triangles)
            torch.testing.assert_close(precise, native, rtol=0, atol=2e-7)
            weights = torch.linspace(0, 0.001, native.numel(), device="cuda").reshape_as(native)
            before = torch.autograd.grad((native * weights).sum(), positions, retain_graph=True)[0]
            after = torch.autograd.grad((precise * weights).sum(), positions, retain_graph=True)[0]
            torch.testing.assert_close(after, before, rtol=2e-5, atol=1e-6)
            for _ in range(10):
                actual = mask_task._antialias(color, raster, positions, triangles)
                torch.testing.assert_close(actual, precise, rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(caller)

    @pytest.mark.parametrize("mode", ["rgb", "rgbd", "qpos", "keypoints"])
    @pytest.mark.parametrize("external_stream", [False, True])
    def test_rebuilt_and_fresh_solvers_match(self, toy_scene, mode, external_stream):
        scene = toy_scene
        unknown = mode in ("qpos", "keypoints")
        config = hec_config.HandEyeCalibratorConfig(
            levels=[
                hec_config.PyramidLevel(2, 0.8, 20, obs_subset_size=2),
                hec_config.PyramidLevel(1, 0.0, 20, use_depth=mode == "rgbd"),
            ],
            solver=PopulationSolverConfig(
                num_iters=6,
                select_at=[3],
                num_candidates=[4, 1],
                optimizer=LMOptimizerConfig(early_stopping_interval=0),
            ),
            optimize_q=unknown,
            optimize_camera=not unknown,
            max_restarts=0,
            keypoint_warmup_iters=3,
            candidate_metric="objective" if mode == "rgbd" else "iou",
        )
        helper = hec_module.HandEyeCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)
        caller = torch.cuda.Stream() if external_stream else torch.cuda.current_stream()
        sources = []
        for frame in range(2):
            intrinsic = scene["intrinsic"].repeat(4, 1, 1)
            intrinsic[:, :2, :2] *= 1 + 0.04 * frame
            mount = torch.eye(4, device="cuda").repeat(4, 1, 1)
            mount[:, 0, 3] = torch.linspace(-0.03, 0.03, 4, device="cuda") * (1 + frame)
            q = torch.full_like(scene["q"], 0.3 if frame == 0 else -0.4) if unknown else scene["q"] * (1 - 2 * frame)
            poses, vertices, faces = hec_module.build_robot_render_data(scene["robot"], q, mount)
            triangles, transformed = mask_task._precompute_merged_mesh(vertices, faces, poses)
            masks, positions = mask_task._render_batch(
                mask_task._rasterize_context(0),
                scene["gt_extrinsic"],
                intrinsic,
                triangles,
                transformed,
                _H,
                _W,
            )
            depths = positions[..., 2].clamp(min=0) * (masks > 0.5)
            masks[frame] = 0
            depths[frame] = 0
            weights = torch.tensor([1.0, 0.8, 0.6, 0.4], device="cuda").roll(frame)
            observations = dict(
                target_masks=masks,
                target_depths=depths if mode == "rgbd" else None,
                observation_weights=weights,
                T_mount_base=mount,
                q=None if unknown else q,
            )
            if mode == "keypoints":
                tip = scene["robot"].forward_kinematics_via_matrix_torch(q)[:, -1, :3, 3]
                tip = tip + mount[:, :3, 3] + scene["gt_extrinsic"][:3, 3]
                pixels = tip[:, :2] / tip[:, 2:] * intrinsic[:, [0, 1], [0, 1]] + intrinsic[:, :2, 2]
                pixels[frame] = float("nan")
                observations.update(keypoint_links=["tip"], target_keypoints=pixels[:, None])
            sources.append((masks, masks.clone(), depths, depths.clone()))
            helper.camera_intrinsic = intrinsic
            initial = scene["gt_extrinsic"] if unknown else _perturbed(scene["gt_extrinsic"])
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                fitted = helper.solve(initial, **observations)
            torch.cuda.current_stream().wait_stream(caller)
            assert helper._solver.config.cuda_graph_mode == "none"
            assert all(not optimizer._graphs for optimizer in helper._solver._optimizers)
            if helper._keypoint_solver is not None:
                assert helper._keypoint_solver.config.cuda_graph_mode == "none"
            if frame == 0:
                solver = helper._solver
                continue
            assert helper._solver is not solver
            fresh = hec_module.HandEyeCalibrator(config, scene["robot"], intrinsic, _H, _W)
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                reference = fresh.solve(initial, **observations)
            torch.cuda.current_stream().wait_stream(caller)
            torch.testing.assert_close(fitted, reference, atol=1e-5, rtol=1e-5)
            assert helper.last_final_cost == pytest.approx(fresh.last_final_cost, rel=1e-5, abs=1e-6)
            if unknown:
                torch.testing.assert_close(helper.last_q, fresh.last_q, atol=1e-5, rtol=1e-5)
            for mask, mask_before, depth, depth_before in sources:
                torch.testing.assert_close(mask, mask_before, atol=0, rtol=0)
                torch.testing.assert_close(depth, depth_before, atol=0, rtol=0)

    def test_rebuilds_after_observation_or_config_change(self, toy_scene):
        scene = toy_scene
        config = hec_config.HandEyeCalibratorConfig(
            levels=[hec_config.PyramidLevel(1, 0.0, 10)],
            solver=PopulationSolverConfig(num_iters=1, num_candidates=[1]),
            max_restarts=0,
        )
        helper = hec_module.HandEyeCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)
        helper.solve(scene["gt_extrinsic"], target_masks=scene["masks"], q=scene["q"])
        original = helper._solver
        helper.solve(scene["gt_extrinsic"], target_masks=scene["masks"][:2], q=scene["q"][:2])
        assert helper._solver is not original
        original = helper._solver
        helper.config = replace(config, candidate_noise_t=0.1)
        helper.solve(scene["gt_extrinsic"], target_masks=scene["masks"][:2], q=scene["q"][:2])
        assert helper._solver is not original
        original = helper._solver
        helper.solve(scene["gt_extrinsic"], target_depths=scene["depths"][:2], q=scene["q"][:2])
        assert helper._solver is not original
        wp.synchronize()
