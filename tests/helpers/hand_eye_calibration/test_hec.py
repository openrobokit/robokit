"""End-to-end HandEyeCalibrator tests (rgb/rgbd/depth, model offsets) on toy box robots."""

from dataclasses import replace
from typing import Any, Optional

import numpy as np
import pytest
import trimesh
import warp as wp
from scipy.spatial.transform import Rotation

from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.robo import Robot


pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")
mask_task = pytest.importorskip("robokit.helpers.hand_eye_calibration.mask_alignment_task")
hec_module = pytest.importorskip("robokit.helpers.hand_eye_calibration")
hec_config = pytest.importorskip("robokit.helpers.hand_eye_calibration.config")
configuration_task = pytest.importorskip("robokit.terms.dense.configuration_task")
se3_identity = pytest.importorskip("robokit.lie.se3").se3_identity
se3_to_matrix = pytest.importorskip("robokit.lie.se3").se3_to_matrix

_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="nvdiffrast requires CUDA")

_H = _W = 128
_N_OBS = 4
_LM = LMOptimizerConfig(lm_lambda=1e-3, rho_min=1e-3)
_ONE_ITER = PopulationSolverConfig(num_iters=1, optimizer=_LM)
_TWO_LEVELS = PopulationSolverConfig(num_iters=20 + 40, select_at=[20], num_candidates=[4, 1], optimizer=_LM)

_URDF = """<robot name="toy">
  <link name="base">
    <visual><origin xyz="-0.25 0 0"/><geometry><mesh filename="box0.obj"/></geometry></visual>
  </link>
  <link name="arm">
    <visual><geometry><mesh filename="box1.obj"/></geometry></visual>
  </link>
  <joint name="j0" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0.25 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <link name="tip"/>
  <joint name="tip_fixed" type="fixed">
    <parent link="arm"/><child link="tip"/><origin xyz="0 .4 0"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def toy_scene(tmp_path_factory) -> dict:
    """Two-box single-joint robot rendered from a known extrinsic."""
    tmp_dir, device = tmp_path_factory.mktemp("toy_robot"), torch.device("cuda")
    wp.init()
    try:
        trimesh.creation.box(extents=(0.4, 0.25, 0.3)).export(tmp_dir / "box0.obj")
        trimesh.creation.box(extents=(0.15, 0.5, 0.2)).export(tmp_dir / "box1.obj")
        (tmp_dir / "toy.urdf").write_text(_URDF)
        robot = Robot.load(str(tmp_dir / "toy.urdf"), load_meshes=True)

        rng = np.random.default_rng(0)
        q = torch.tensor(rng.uniform(-0.6, 0.6, (_N_OBS, 1)), dtype=torch.float32, device=device)

        gt_extrinsic = torch.eye(4, dtype=torch.float32, device=device)
        gt_extrinsic[2, 3] = 1.5  # boxes ~1.5 m in front of the camera
        intrinsic = torch.tensor(
            [[150.0, 0.0, _W / 2], [0.0, 150.0, _H / 2], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )

        T_world_link, link_vertices, link_faces = hec_module.build_robot_render_data(robot, q)
        merged_faces, v_base = mask_task._precompute_merged_mesh(link_vertices, link_faces, T_world_link)
        face_link_ids = torch.cat(
            [
                torch.full((faces.shape[0],), i + 1, dtype=torch.int32, device=device)
                for i, faces in enumerate(link_faces)
            ]
        )
        glctx = mask_task.dr.RasterizeCudaContext()
        with torch.no_grad():
            masks, positions = mask_task._render_batch(glctx, gt_extrinsic, intrinsic, merged_faces, v_base, _H, _W)
            arm_masks, arm_positions, _ = mask_task._render_batch(
                glctx,
                gt_extrinsic,
                intrinsic,
                merged_faces,
                v_base,
                _H,
                _W,
                face_link_ids=face_link_ids,
                target_link_mask=torch.tensor([False, False, True], device=device),
            )
        depths = positions[..., 2].clamp(min=0.0) * (masks > 0.5)
        arm_depths = arm_positions[..., 2].clamp(min=0.0) * (arm_masks > 0.5)

        return dict(
            asset_directory=tmp_dir,
            robot=robot,
            q=q,
            intrinsic=intrinsic,
            gt_extrinsic=gt_extrinsic,
            masks=masks.contiguous(),
            depths=depths.contiguous(),
            arm_masks=arm_masks.contiguous(),
            arm_depths=arm_depths.contiguous(),
        )
    except (ImportError, RuntimeError) as e:  # nvdiffrast plugin/torch mismatch
        pytest.skip(f"nvdiffrast unavailable: {e}")


def _perturbed(gt: torch.Tensor) -> torch.Tensor:
    T = gt.clone()
    R = torch.from_numpy(Rotation.from_euler("xyz", [0.05, -0.06, 0.04]).as_matrix().astype(np.float32))
    T[:3, :3] = R.to(gt.device) @ T[:3, :3]
    T[:3, 3] += torch.tensor([0.03, -0.02, 0.03], device=gt.device)
    return T


def _pose_error(predicted: torch.Tensor, gt: torch.Tensor) -> tuple:
    t_err = (gt[:3, 3] - predicted[:3, 3]).norm().item()
    R_rel = gt[:3, :3] @ predicted[:3, :3].t()
    r_err = torch.acos(((R_rel.trace() - 1) / 2).clamp(-1, 1)).item() * 180 / np.pi
    return t_err, r_err


def _solve(scene: dict, masks, depths, levels, solver) -> tuple:
    helper = hec_module.HandEyeCalibrator(
        camera_intrinsic=scene["intrinsic"],
        height=_H,
        width=_W,
        robot=scene["robot"],
        config=hec_config.HandEyeCalibratorConfig(levels=levels, solver=solver),
    )
    predicted = helper.solve(
        _perturbed(scene["gt_extrinsic"]),
        target_masks=masks,
        target_depths=depths,
        q=scene["q"],
    )
    assert all(buffer.shape[1] == 0 for buffer in helper._solver._residuals)
    return _pose_error(predicted, scene["gt_extrinsic"])


_OFFSET_URDF = _URDF.replace(
    "</robot>",
    """  <link name="hand">
    <visual><geometry><mesh filename="box2.obj"/></geometry></visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="arm"/><child link="hand"/>
    <origin xyz="0.2 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>""",
)

_OFFSET_LEVELS = [
    hec_config.PyramidLevel(downscale=4, blur_sigma=1.2, patience=5),
    hec_config.PyramidLevel(downscale=1, blur_sigma=0.0, patience=8),
]


def _rest_weight(masks) -> float:
    """Rest weight making a one-sigma offset cost ~2.5e-4 per target-mask pixel."""
    return float(np.sqrt(2.5e-4 * masks.sum().item()))


@pytest.fixture(scope="module")
def offset_scene(tmp_path_factory) -> dict:
    """Three-box two-joint robot whose targets are rendered from a *wrong* model."""
    tmp_dir, device = tmp_path_factory.mktemp("offset_robot"), torch.device("cuda")
    wp.init()
    try:
        trimesh.creation.box(extents=(0.4, 0.25, 0.3)).export(tmp_dir / "box0.obj")
        trimesh.creation.box(extents=(0.15, 0.5, 0.2)).export(tmp_dir / "box1.obj")
        trimesh.creation.box(extents=(0.3, 0.3, 0.35)).export(tmp_dir / "box2.obj")
        (tmp_dir / "toy.urdf").write_text(_OFFSET_URDF)
        robot = Robot.load(str(tmp_dir / "toy.urdf"), load_meshes=True)

        rng = np.random.default_rng(0)
        q = torch.tensor(rng.uniform(-0.8, 0.8, (8, 2)), dtype=torch.float32, device=device)
        # oblique view: looking down the joint-0 axis, a base offset is just a camera roll
        eye = np.array([0.9, -0.9, 0.7])
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        T_base_cam = np.eye(4, dtype=np.float32)
        T_base_cam[:3, :3] = np.stack([right, np.cross(fwd, right), fwd], axis=1)
        T_base_cam[:3, 3] = eye
        gt_extrinsic = torch.from_numpy(np.linalg.inv(T_base_cam).astype(np.float32)).to(device)
        intrinsic = torch.tensor(
            [[150.0, 0.0, _W / 2], [0.0, 150.0, _H / 2], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )

        qpos_offset = torch.tensor([0.03, -0.04], dtype=torch.float32, device=device)
        mount_offset = torch.eye(4, dtype=torch.float32, device=device)
        mount_offset[:3, 3] = torch.tensor([0.012, -0.008, 0.01], device=device)
        mount_offset[:3, :3] = torch.from_numpy(
            Rotation.from_rotvec([0.0, 0.03, -0.02]).as_matrix().astype(np.float32)
        ).to(device)
        common = dict(robot=robot, camera_intrinsic=intrinsic, extrinsic=gt_extrinsic, height=_H, width=_W)
        return dict(
            robot=robot,
            q=q,
            intrinsic=intrinsic,
            gt_extrinsic=gt_extrinsic,
            qpos_offset=qpos_offset,
            mount_offset=mount_offset,
            qpos_masks=hec_module.compute_robot_masks(q=q + qpos_offset, **common),
            mount_masks=hec_module.compute_robot_masks(q=q, mount_link="hand", T_mount_offset=mount_offset, **common),
        )
    except (ImportError, RuntimeError) as e:  # nvdiffrast plugin/torch mismatch
        pytest.skip(f"nvdiffrast unavailable: {e}")


@_REQUIRES_CUDA
class TestHandEyeCalibratorOffsets:
    """Solving the model offset must beat absorbing it into the extrinsic."""

    def _solve(self, offset_scene, masks, mask_task_kwargs=None, terms=()) -> Any:
        config = hec_config.HandEyeCalibratorConfig(levels=_OFFSET_LEVELS, solver=_TWO_LEVELS)
        if mask_task_kwargs is not None:
            mask_term = mask_task.MaskAlignmentOffsetTask(**mask_task_kwargs)
            config = replace(config, terms=[mask_term, *terms])
        helper = hec_module.HandEyeCalibrator(
            camera_intrinsic=offset_scene["intrinsic"], height=_H, width=_W, robot=offset_scene["robot"], config=config
        )
        predicted = helper.solve(_perturbed(offset_scene["gt_extrinsic"]), target_masks=masks, q=offset_scene["q"])
        return helper, predicted

    def test_qpos_offset_recovery(self, offset_scene):
        masks = offset_scene["qpos_masks"]
        camera_only, _ = self._solve(offset_scene, masks)
        rest = configuration_task.ConfigurationTask(
            rest_q=np.zeros(2, dtype=np.float32),
            weight=_rest_weight(masks) / 0.035,
            var_key=mask_task.VarKey.QPOS_OFFSET,
        )
        helper, predicted = self._solve(offset_scene, masks, dict(offset_joints=[0, 1]), [rest])
        solved = wp.to_torch(helper.last_var.get(mask_task.VarKey.QPOS_OFFSET).x)[0]
        err = (solved - offset_scene["qpos_offset"]).abs().max().item()
        t_err, r_err = _pose_error(predicted, offset_scene["gt_extrinsic"])
        assert helper.last_final_cost < 0.7 * camera_only.last_final_cost
        assert err < 0.01, f"qpos offset error {np.rad2deg(err):.2f} deg"
        assert t_err < 0.01 and r_err < 1.0, f"qpos: {t_err * 100:.2f}cm / {r_err:.2f}deg"

    def test_mount_offset_recovery(self, offset_scene):
        masks = offset_scene["mount_masks"]
        camera_only, _ = self._solve(offset_scene, masks)
        scale = _rest_weight(masks)
        rest = configuration_task.ConfigurationTask(
            T_world_base_rest=se3_identity(shape=(1,), device="cuda:0"),
            base_weight=[scale / 0.01] * 3 + [scale / 0.035] * 3,
            include_joints=False,
            var_key=mask_task.VarKey.MOUNT_OFFSET,
        )
        helper, _ = self._solve(offset_scene, masks, dict(offset_link="hand"), [rest])
        solved = torch.as_tensor(se3_to_matrix(helper.last_var.get(mask_task.VarKey.MOUNT_OFFSET).xyz_wxyz).numpy()[0])
        t_err, r_err = _pose_error(solved.cuda(), offset_scene["mount_offset"])
        assert helper.last_final_cost < 0.7 * camera_only.last_final_cost
        assert t_err < 0.005 and r_err < 1.5, f"mount: {t_err * 1000:.2f}mm / {r_err:.2f}deg"

    def test_offset_task_needs_an_offset(self, offset_scene):
        with pytest.raises(ValueError):
            self._solve(offset_scene, offset_scene["mount_masks"], {})


@_REQUIRES_CUDA
class TestHandEyeCalibratorDepth:
    def test_depth_only_convergence(self, toy_scene):
        # masks=None also exercises the valid-depth-pixel observation ranking (obs_subset_size < N)
        levels = [
            hec_config.PyramidLevel(downscale=4, blur_sigma=0.0, patience=5, use_depth=True, obs_subset_size=3),
            hec_config.PyramidLevel(downscale=1, blur_sigma=0.0, patience=8, use_depth=True),
        ]
        t_err, r_err = _solve(toy_scene, None, toy_scene["depths"], levels, _TWO_LEVELS)
        assert t_err < 0.01 and r_err < 1.0, f"depth-only: {t_err * 100:.2f}cm / {r_err:.2f}deg"

    def test_rgbd_convergence(self, toy_scene):
        levels = [
            hec_config.PyramidLevel(downscale=4, blur_sigma=1.2, patience=5),
            hec_config.PyramidLevel(downscale=1, blur_sigma=0.0, patience=8, use_depth=True),
        ]
        t_err, r_err = _solve(toy_scene, toy_scene["masks"], toy_scene["depths"], levels, _TWO_LEVELS)
        assert t_err < 0.01 and r_err < 1.0, f"rgbd: {t_err * 100:.2f}cm / {r_err:.2f}deg"

    def test_target_link_subset(self, toy_scene):
        levels = [hec_config.PyramidLevel(downscale=1, blur_sigma=0.0, patience=1, use_depth=True)]
        helper = hec_module.HandEyeCalibrator(
            config=hec_config.HandEyeCalibratorConfig(levels=levels, solver=_ONE_ITER),
            robot=toy_scene["robot"],
            camera_intrinsic=toy_scene["intrinsic"],
            height=_H,
            width=_W,
            render_link_names=["base", "arm"],
            target_link_names=["arm"],
        )
        helper.solve(
            toy_scene["gt_extrinsic"],
            target_masks=toy_scene["arm_masks"],
            target_depths=toy_scene["arm_depths"],
            q=toy_scene["q"],
        )
        assert helper.last_final_cost < 1e-6

    def test_target_link_subset_per_observation(self, toy_scene):
        levels = [hec_config.PyramidLevel(downscale=1, blur_sigma=0.0, patience=1, use_depth=True)]
        helper = hec_module.HandEyeCalibrator(
            config=hec_config.HandEyeCalibratorConfig(levels=levels, solver=_ONE_ITER),
            robot=toy_scene["robot"],
            camera_intrinsic=toy_scene["intrinsic"],
            height=_H,
            width=_W,
            render_link_names=["base", "arm"],
            target_link_names=[["arm"]] * _N_OBS + [["base", "arm"]] * _N_OBS,
        )
        helper.solve(
            toy_scene["gt_extrinsic"],
            target_masks=torch.cat([toy_scene["arm_masks"], torch.zeros_like(toy_scene["masks"])]),
            target_depths=torch.cat([toy_scene["arm_depths"], torch.zeros_like(toy_scene["depths"])]),
            observation_weights=torch.cat(
                [torch.ones(_N_OBS, device=toy_scene["q"].device), torch.zeros(_N_OBS, device=toy_scene["q"].device)]
            ),
            q=torch.cat([toy_scene["q"], toy_scene["q"]]),
        )
        assert helper.last_final_cost < 1e-6

    def test_target_links_must_be_rendered(self, toy_scene):
        helper = hec_module.HandEyeCalibrator(
            config=hec_config.HandEyeCalibratorConfig(),
            robot=toy_scene["robot"],
            camera_intrinsic=toy_scene["intrinsic"],
            height=_H,
            width=_W,
            render_link_names=["base"],
            target_link_names=["arm"],
        )
        with pytest.raises(ValueError, match="target links are not rendered"):
            helper.solve(toy_scene["gt_extrinsic"], target_masks=toy_scene["arm_masks"], q=toy_scene["q"])

    def test_requires_some_target(self, toy_scene):
        with pytest.raises(ValueError):
            _solve(toy_scene, None, None, hec_config._default_levels(), hec_config._default_solver())

    def test_use_depth_without_depths_raises(self, toy_scene):
        levels = [hec_config.PyramidLevel(downscale=1, blur_sigma=0.0, patience=2, use_depth=True)]
        with pytest.raises(ValueError):
            _solve(toy_scene, toy_scene["masks"], None, levels, PopulationSolverConfig(num_iters=5, optimizer=_LM))


@_REQUIRES_CUDA
class TestHandEyeCalibratorQpos:
    @pytest.mark.parametrize("selection", [None, "global", "observation"])
    def test_raster_chunks_preserve_instances_and_link_selection(self, monkeypatch, selection: Optional[str]):
        vertices = torch.tensor([[[3.1, 3.1, 1.0], [5.8, 3.1, 1.0], [3.1, 5.8, 1.0]]], device="cuda")
        vertices = vertices.repeat(2, 513, 1, 1)
        vertices[1, :, :, 0] += 1
        cameras = torch.eye(4, device="cuda").repeat(2, 1, 1)
        cameras[1, 1, 3] += 1
        intrinsic = torch.eye(3, device="cuda").repeat(513, 1, 1)
        intrinsic[:, 0, 2] = torch.arange(513, device="cuda") % 2
        selected = None
        if selection == "global":
            selected = torch.tensor([False, True], device="cuda")
        elif selection == "observation":
            selected = torch.stack(
                [torch.zeros(513, device="cuda", dtype=torch.bool), torch.arange(513, device="cuda") % 2 == 0], dim=1
            )
        args = (
            mask_task._rasterize_context(0),
            cameras,
            intrinsic,
            torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda"),
            vertices,
            16,
            16,
        )
        kwargs = dict(face_link_ids=torch.ones(1, dtype=torch.int32, device="cuda"), target_link_mask=selected)
        rasterize = mask_task.dr.rasterize
        calls = []

        def record(context, clip, *args, **kwargs):
            calls.append(len(clip))
            return rasterize(context, clip, *args, **kwargs)

        monkeypatch.setattr(mask_task.dr, "rasterize", record)
        actual = mask_task._render_batch(*args, **kwargs)
        assert calls == [1024, 2]
        expected = []
        for candidate in range(2):
            for observation in range(513):
                expected.append(
                    mask_task._render_batch(
                        args[0],
                        cameras[candidate],
                        intrinsic[observation],
                        args[3],
                        vertices[candidate, observation : observation + 1],
                        16,
                        16,
                        face_link_ids=kwargs["face_link_ids"],
                        target_link_mask=selected[observation] if selection == "observation" else selected,
                    )
                )
        for component, result in enumerate(actual):
            torch.testing.assert_close(result, torch.cat([item[component] for item in expected]), rtol=0, atol=0)
        assert actual[0].sum() > 0
        if selection == "observation":
            assert actual[0].reshape(2, 513, 16, 16)[:, 1::2].sum() == 0

    @pytest.mark.parametrize("expand", [False, True])
    def test_boundary_scale_gradient_matches_projected_edge(self, expand: bool):
        rendered = torch.zeros((1, 5, 5), device="cuda")
        rendered[..., :3] = 1
        target = torch.zeros_like(rendered)
        target[..., : 4 if expand else 2] = 1
        positions = torch.zeros((1, 5, 5, 3), device="cuda")
        positions[..., :3, 0] = torch.arange(3, device="cuda") + 0.5
        positions[..., :3, 2] = 1
        normal = wp.zeros((1, 6, 6), dtype=wp.float64, device="cuda")
        gradient = wp.zeros((1, 6, 1), dtype=wp.float64, device="cuda")
        cost = wp.zeros(1, dtype=wp.float64, device="cuda")
        wp.launch(
            mask_task._accumulate_normal_equations_kernel,
            dim=(1, 5, 5),
            inputs=[
                wp.from_torch(rendered),
                wp.from_torch(target),
                wp.ones(1, device="cuda"),
                wp.zeros(1, device="cuda"),
                True,
                True,
                wp.from_torch(positions),
                wp.ones((1, 2), device="cuda"),
                5,
                5,
                1,
                wp.from_torch(torch.eye(6, device="cuda")[None], dtype=mask_task.wp_mat66),
                wp.zeros((1, 1, 1, 1), dtype=mask_task.wp_vec6, device="cuda"),
                wp.zeros((1, 5, 5), dtype=wp.int32, device="cuda"),
                0,
                6,
            ],
            outputs=[normal, gradient, cost],
            device="cuda",
        )

        # Project a plane's edge under camera-x, camera-z and camera-y-rotation perturbations.
        def image_cost(tx: float = 0, tz: float = 0, ry: float = 0) -> torch.Tensor:
            edge = (3 * np.cos(ry) + np.sin(ry) + tx) / (-3 * np.sin(ry) + np.cos(ry) + tz)
            image = (edge - torch.arange(5, dtype=torch.float64)).clamp(0, 1)
            return 0.5 * (image - target[0].cpu().double()).square().sum()

        epsilon = 1e-6
        for column, parameter in [(0, "tx"), (2, "tz"), (4, "ry")]:
            finite = (image_cost(**{parameter: epsilon}) - image_cost(**{parameter: -epsilon})) / (2 * epsilon)
            torch.testing.assert_close(wp.to_torch(gradient)[0, column, 0].cpu(), finite, rtol=0, atol=1e-4)

    def test_pixel_reductions_repeat_exactly(self, toy_scene: dict):
        config = hec_config.HandEyeCalibratorConfig(levels=[hec_config.PyramidLevel(1, 0, 5)], solver=_ONE_ITER)
        helper = hec_module.HandEyeCalibrator(config, toy_scene["robot"], toy_scene["intrinsic"], _H, _W)
        helper._build(toy_scene["masks"], None, None, toy_scene["q"], None)
        camera = toy_scene["gt_extrinsic"].clone()
        camera[0, 3] += 0.03
        variables = mask_task.VarValues(
            {
                mask_task.VarKey.ROBOT: mask_task.SE3Var(
                    mask_task.se3_from_matrix(wp.from_torch(camera[None], dtype=wp.mat44))
                )
            }
        )
        term = helper._solver.terms[0][0]
        term.prepare(variables, False, 0)
        normal = wp.zeros((1, 6, 6), device="cuda")
        gradient, cost = wp.zeros((1, 6, 1), device="cuda"), wp.zeros(1, device="cuda")
        previous = None
        for _ in range(10):
            normal.zero_()
            gradient.zero_()
            cost.zero_()
            term.accumulate_normal_equations(variables, costs=cost, JtJ=normal, Jtr=gradient)
            system = torch.cat([wp.to_torch(buffer).flatten() for buffer in [normal, gradient, cost]]).clone()
            assert float(wp.to_torch(cost)[0]) > 0
            if previous is not None:
                torch.testing.assert_close(system, previous, rtol=0, atol=0)
            previous = system

    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("target_value", [0.0, 0.3])
    @pytest.mark.parametrize("optimize_camera", [False, True])
    def test_union_normalization_matches_loss_gradient(
        self, normalize: bool, target_value: float, optimize_camera: bool
    ):
        wp.init()
        rendered = (torch.arange(5, device="cuda") * 0.2)[None, None].expand(1, 5, 5).contiguous()
        target = torch.full_like(rendered, target_value)
        term = mask_task.MaskAlignmentNormalEqTask(
            target_masks=target,
            target_depths=None,
            observation_weights=torch.ones(1, device="cuda"),
            height=5,
            width=5,
            blur_sigma=0,
        )
        term.N, term.torch_device, term.normalize_by_union = 1, rendered.device, normalize
        term.num_extra_dofs, term._optimize_camera = 1, optimize_camera
        term._alloc_buffers(1, "cuda")
        term._rendered_curr.copy_(rendered)
        term._positions_curr.zero_()
        term._positions_curr[..., 2] = 1
        wp.to_torch(term._wp_adjoint_curr).copy_(torch.eye(6, device="cuda")[None])
        term._wp_focal = wp.from_torch(torch.full((1, 2), 10.0, device="cuda"))
        term._wp_screws = wp.from_torch(
            torch.tensor([[[[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]], device="cuda"), dtype=mask_task.wp_vec6
        )
        term._normalize_mask_loss(False)
        normal, gradient, cost = (
            wp.zeros((1, 7, 7), device="cuda"),
            wp.zeros((1, 7, 1), device="cuda"),
            wp.zeros(1, device="cuda"),
        )
        variables = mask_task.VarValues({mask_task.VarKey.ROBOT: mask_task.SE3Var(se3_identity(1, device="cuda"))})
        term.accumulate_normal_equations(variables, costs=cost, JtJ=normal, Jtr=gradient)
        shift = torch.zeros((), device="cuda", requires_grad=True)
        translated = rendered - 2 * shift
        expected = 0.5 * (translated - target).square().sum()
        if normalize and target_value:
            expected = expected * target.sum() / torch.maximum(translated, target).sum()
        expected.backward()
        torch.testing.assert_close(wp.to_torch(cost)[0], expected.detach())
        torch.testing.assert_close(wp.to_torch(gradient)[0, 6, 0], shift.grad)
        torch.testing.assert_close(
            wp.to_torch(gradient)[0, 0, 0], shift.grad if optimize_camera else torch.zeros_like(shift)
        )
        term._rendered_prop.copy_(rendered)
        term._normalize_mask_loss(True)
        cost.zero_()
        term.accumulate_normal_equations(variables, costs=cost)
        torch.testing.assert_close(wp.to_torch(cost)[0], expected.detach())

    def test_blur_preserves_constant_images_at_borders(self):
        masks = torch.ones((2, 16, 16), device="cuda")
        torch.testing.assert_close(mask_task.MaskAlignmentNormalEqTask()._blur(masks, 1.0), masks)

    @pytest.mark.parametrize("border", [False, True])
    def test_silhouette_boundary_and_border_gradients(self, border: bool):
        rendered = torch.zeros((1, 5, 5), device="cuda")
        row = 0 if border else 2
        rendered[0, row, 2] = 1
        target = torch.zeros_like(rendered)
        if not border:
            target[0, 2, 1] = 1
        positions = torch.zeros((1, 5, 5, 3), device="cuda")
        positions[0, row, 2, 2] = 1
        normal, gradient, cost = (
            wp.zeros((1, 6, 6), dtype=wp.float64, device="cuda"),
            wp.zeros((1, 6, 1), dtype=wp.float64, device="cuda"),
            wp.zeros(1, dtype=wp.float64, device="cuda"),
        )
        wp.launch(
            mask_task._accumulate_normal_equations_kernel,
            dim=(1, 5, 5),
            inputs=[
                wp.from_torch(rendered),
                wp.from_torch(target),
                wp.ones(1, device="cuda"),
                wp.zeros(1, device="cuda"),
                True,
                False,
                wp.from_torch(positions),
                wp.from_torch(torch.full((1, 2), 10.0, device="cuda")),
                5,
                5,
                1,
                wp.from_torch(torch.eye(6, device="cuda")[None], dtype=mask_task.wp_mat66),
                wp.zeros((1, 1, 1, 1), dtype=mask_task.wp_vec6, device="cuda"),
                wp.zeros((1, 5, 5), dtype=wp.int32, device="cuda"),
                0,
                6,
            ],
            outputs=[normal, gradient, cost],
            device="cuda",
        )
        assert float(wp.to_torch(gradient)[0, int(border), 0]) == pytest.approx(10.0 if border else 5.0)
        assert float(wp.to_torch(cost)[0]) == pytest.approx(0.5 if border else 1.0)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_pyramid_cost_preserves_observation_weight(self, toy_scene: dict, weighted: bool):
        """A constant image error must keep its weight relative to non-image terms at every level."""
        config = hec_config.HandEyeCalibratorConfig(
            levels=[hec_config.PyramidLevel(4, 0, 5), hec_config.PyramidLevel(1, 0, 5)],
            solver=PopulationSolverConfig(num_iters=1, select_at=[0], num_candidates=[1, 1], optimizer=_LM),
        )
        helper = hec_module.HandEyeCalibrator(config, toy_scene["robot"], toy_scene["intrinsic"], _H, _W)
        masks = torch.zeros_like(toy_scene["masks"])
        weights = torch.tensor([0.5, 1.0, 0.0, 2.0], device="cuda") if weighted else None
        helper._build(masks, None, weights, toy_scene["q"], None)
        expected = 0.5 * 0.25**2 * _H * _W * (_N_OBS if weights is None else float(weights.sum()))
        for terms in helper._solver.terms:
            term = terms[0]
            term._alloc_buffers(1, "cuda")
            term._blurred_prop.fill_(0.25)
            cost = wp.zeros(1, device="cuda")
            term.accumulate_normal_equations(helper._initial_var, costs=cost)
            torch.testing.assert_close(wp.to_torch(cost), torch.tensor([expected], device="cuda"))

    """One q vector is fitted per call; observations are views of that frame."""

    def test_accepted_render_reuse_with_mixed_q_steps(self, offset_scene: dict):
        scene = offset_scene
        config = replace(
            hec_module.presets.rgb_qpos,
            levels=[hec_config.PyramidLevel(1, 0.0, 5)],
            solver=PopulationSolverConfig(num_iters=1, num_candidates=[2], optimizer=_LM),
            max_restarts=0,
        )
        helper = hec_module.HandEyeCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)
        helper.solve(scene["gt_extrinsic"], target_masks=scene["qpos_masks"][:1])
        term, values = helper._solver.terms[0][0], helper._initial_var
        term._curr_is_fresh = False
        term.prepare(values, False, 1)
        step = torch.zeros((2, values.tangent_dim), device="cuda")
        step[:, 6:] = torch.tensor([[0.03, -0.05], [-0.04, 0.02]], device="cuda")
        proposed = values.integrate(wp.from_torch(step))
        term.prepare(proposed, True, 1)
        accepted = wp.array([1, 0], dtype=wp.int32, device="cuda")
        values.accept(accepted, proposed)
        term._accept_render(accepted)
        term.prepare(values, False, 2)
        cached = [
            term._rendered_curr.clone(),
            term._positions_curr.clone(),
            term._pixel_link.clone(),
            wp.to_torch(term._wp_screws).clone(),
        ]
        term._curr_is_fresh = False
        term.prepare(values, False, 3)
        fresh = [term._rendered_curr, term._positions_curr, term._pixel_link, wp.to_torch(term._wp_screws)]
        for actual, expected in zip(cached, fresh):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("terminal_sample", [False, True])
    def test_iou_scores_match_terminal_population(self, offset_scene, terminal_sample: bool):
        from robokit.opt.sampler import RobertsSamplerConfig
        from robokit.opt.var_values import VarKey

        robot, camera = offset_scene["robot"], offset_scene["gt_extrinsic"]
        truth = torch.tensor([0.35, -0.45], device="cuda")
        target = hec_module.compute_robot_masks(robot, truth[None], offset_scene["intrinsic"], camera, _H, _W)
        config = replace(
            hec_module.presets.rgb_qpos,
            candidate_metric="iou",
            levels=[hec_config.PyramidLevel(1, 0.0, 3)],
            solver=PopulationSolverConfig(
                num_iters=4,
                num_candidates=[4],
                optimizer=_LM,
                samplers=[RobertsSamplerConfig(var_key=VarKey.QPOS_OFFSET, scale=0.2)] if terminal_sample else [],
                sample_at=[4] if terminal_sample else [],
            ),
            candidate_noise_q=0.5,
            max_restarts=0,
        )
        helper = hec_module.HandEyeCalibrator(config, robot, offset_scene["intrinsic"], _H, _W)
        fitted_camera = helper.solve(camera, target_masks=target, q=truth)
        predicted = hec_module.compute_robot_masks(
            robot, helper.last_candidates, offset_scene["intrinsic"], fitted_camera, _H, _W
        ).clamp(0, 1)
        intersection = torch.minimum(predicted, target).sum((-2, -1))
        union = torch.maximum(predicted, target).sum((-2, -1))
        expected = 0.5 * target.sum() * (1 - intersection / union)
        torch.testing.assert_close(helper.last_candidate_costs, expected, atol=0.002, rtol=0.002)

    @pytest.mark.parametrize("candidate_metric", ["objective", "iou"])
    def test_recovery_without_initial_q(self, offset_scene, candidate_metric: str):
        robot, camera = offset_scene["robot"], offset_scene["gt_extrinsic"]
        truth = torch.tensor([1.0, -0.7], device="cuda")
        views = torch.eye(4, device="cuda").repeat(3, 1, 1)
        for i, angle in enumerate([-0.5, 0.0, 0.5]):
            views[i, :3, :3] = torch.as_tensor(Rotation.from_euler("y", angle).as_matrix(), device="cuda")
        masks = hec_module.compute_robot_masks(
            robot, truth[None].repeat(3, 1), offset_scene["intrinsic"], camera, _H, _W, T_mount_base=views
        )
        recipe = hec_module.presets.rgb_qpos
        config = replace(
            recipe,
            candidate_metric=candidate_metric,
            max_restarts=0,
            solver=replace(recipe.solver, num_candidates=[min(width, 16) for width in recipe.solver.num_candidates]),
        )
        helper = hec_module.HandEyeCalibrator(config, robot, offset_scene["intrinsic"], _H, _W)
        fitted_camera = helper.solve(camera, target_masks=masks, T_mount_base=views)
        assert helper.last_var.batch_size == 1  # Native ranking returns one winner from the final beam.
        fitted = (
            hec_module.compute_robot_masks(
                robot,
                helper.last_q[None].repeat(3, 1),
                offset_scene["intrinsic"],
                fitted_camera,
                _H,
                _W,
                T_mount_base=views,
            )
            > 0.5
        )
        target = masks > 0.5
        iou = (fitted & target).sum() / (fitted | target).sum()
        assert iou > 0.98
        torch.testing.assert_close(fitted_camera, camera, atol=1e-6, rtol=0.0)

    @pytest.mark.parametrize("optimize_camera", [False, True])
    def test_frame_recovery(self, offset_scene, optimize_camera):
        robot = offset_scene["robot"]
        truth = torch.tensor([0.35, -0.45], device="cuda")
        # Three known view transforms make both joints observable in this frame.
        views = torch.eye(4, device="cuda").repeat(3, 1, 1)
        for i, angle in enumerate([-0.45, 0.0, 0.45]):
            views[i, :3, :3] = torch.as_tensor(Rotation.from_euler("z", angle).as_matrix(), device="cuda")
        camera = offset_scene["gt_extrinsic"]
        masks = hec_module.compute_robot_masks(
            robot, truth[None].repeat(3, 1), offset_scene["intrinsic"], camera, _H, _W, T_mount_base=views
        )
        initial_q = truth + torch.tensor([0.25, -0.25], device="cuda")
        config = replace(
            hec_module.presets.rgb_qpos_camera if optimize_camera else hec_module.presets.rgb_qpos,
            levels=_OFFSET_LEVELS,
            solver=_TWO_LEVELS,
            candidate_noise_q=0.6,
            candidate_noise_t=0.0,
            candidate_noise_r=0.0,
            max_restarts=0,
        )
        if not optimize_camera:
            rest = configuration_task.ConfigurationTask(
                robot, rest_q=initial_q.cpu().numpy(), weight=0.05, var_key=mask_task.VarKey.QPOS_OFFSET
            )
            config = replace(config, terms=[*config.terms, rest])
        helper = hec_module.HandEyeCalibrator(config, robot, offset_scene["intrinsic"], _H, _W)
        initial_camera = camera.clone()
        if optimize_camera:
            initial_camera[0, 3] += 0.008
        fitted = helper.solve(initial_camera, target_masks=masks, q=initial_q, T_mount_base=views)
        torch.testing.assert_close(helper.last_q, truth, atol=0.03, rtol=0.0)
        assert _pose_error(fitted, camera)[0] < 0.005
        if not optimize_camera:
            torch.testing.assert_close(fitted, initial_camera, atol=1e-6, rtol=0.0)
        assert type(config.terms[0]) is mask_task.MaskAlignmentNormalEqTask
        # Re-solving cached observations must retain the q initial guess and output contract.
        helper.solve(initial_camera)
        torch.testing.assert_close(helper.last_q, truth, atol=0.03, rtol=0.0)

    def test_full_range_sampling_and_joint_limits(self, offset_scene):
        robot = offset_scene["robot"]
        config = hec_config.HandEyeCalibratorConfig(
            levels=[hec_config.PyramidLevel(4, 1.2, 1), hec_config.PyramidLevel(1, 0.0, 1)],
            solver=replace(_TWO_LEVELS, num_iters=2, select_at=[1], num_candidates=[8, 1]),
            optimize_q=True,
            optimize_camera=False,
            max_restarts=0,
        )
        helper = hec_module.HandEyeCalibrator(config, robot, offset_scene["intrinsic"], _H, _W)
        initial = torch.as_tensor(robot.spec.actuated_joint_limits[:, 1], device="cuda", dtype=torch.float32)
        helper.solve(offset_scene["gt_extrinsic"], target_masks=offset_scene["qpos_masks"][:1], q=initial)
        state = helper.last_var.get(mask_task.VarKey.QPOS_OFFSET)
        assert state.clamp_q_on_integrate
        limits = torch.as_tensor(robot.spec.actuated_joint_limits, device="cuda", dtype=torch.float32)
        assert (helper.last_q >= limits[:, 0]).all() and (helper.last_q <= limits[:, 1]).all()
        helper.solve(offset_scene["gt_extrinsic"], target_masks=offset_scene["qpos_masks"][:1])
        assert torch.isfinite(helper.last_q).all()

    @pytest.mark.parametrize("initial", [[[0.0, 0.0]], [float("nan"), 0.0], [100.0, 0.0]])
    def test_invalid_initial_q(self, offset_scene, initial):
        config = hec_config.HandEyeCalibratorConfig(optimize_q=True)
        helper = hec_module.HandEyeCalibrator(config, offset_scene["robot"], offset_scene["intrinsic"], _H, _W)
        with pytest.raises(ValueError):
            helper.solve(
                offset_scene["gt_extrinsic"],
                target_masks=offset_scene["qpos_masks"][:1],
                q=torch.tensor(initial, device="cuda"),
            )

    def test_incompatible_shared_offsets(self, offset_scene):
        config = hec_config.HandEyeCalibratorConfig(
            optimize_q=True, terms=[mask_task.MaskAlignmentOffsetTask(offset_joints=[0, 1])]
        )
        helper = hec_module.HandEyeCalibrator(config, offset_scene["robot"], offset_scene["intrinsic"], _H, _W)
        with pytest.raises(ValueError, match="shared model-offset"):
            helper.solve(
                offset_scene["gt_extrinsic"],
                target_masks=offset_scene["qpos_masks"][:1],
                q=torch.zeros(2, device="cuda"),
            )

    def test_invalid_configuration(self, offset_scene):
        with pytest.raises(ValueError, match="candidate_noise_q"):
            hec_config.HandEyeCalibratorConfig(candidate_noise_q=-0.1)
        with pytest.raises(ValueError, match="enable optimize"):
            helper = hec_module.HandEyeCalibrator(
                hec_config.HandEyeCalibratorConfig(optimize_camera=False),
                offset_scene["robot"],
                offset_scene["intrinsic"],
                _H,
                _W,
            )
            helper.solve(
                offset_scene["gt_extrinsic"], target_masks=offset_scene["qpos_masks"][:1], q=offset_scene["q"][:1]
            )


@_REQUIRES_CUDA
class TestQposTrajectoryCalibrator:
    @pytest.mark.parametrize("count", [1, 2])
    def test_initialization_extends_observed_motion(self, toy_scene: dict, monkeypatch, count: int):
        from robokit.helpers.hand_eye_calibration import calibrator

        def observed_fit(helper, initial, *, target_masks, **kwargs):
            value = target_masks[0, 0, 0]
            helper.last_q = value[None]
            helper.last_candidates = value.reshape(1, 1)
            helper.last_candidate_costs = torch.zeros(1, device="cuda")

        monkeypatch.setattr(calibrator.HandEyeCalibrator, "solve", observed_fit)
        config = hec_config.QposTrajectoryCalibratorConfig(initialization_frames=count)
        helper = hec_module.QposTrajectoryCalibrator(config, toy_scene["robot"], toy_scene["intrinsic"], _H, _W)
        masks = torch.zeros((6, 1, _H, _W), device="cuda")
        masks[2], masks[4] = 0.2, 0.4
        valid = torch.zeros((6, 1), device="cuda")
        valid[2], valid[4] = 1, 1
        times = torch.arange(6.0, device="cuda")
        q = helper._initialize_q(
            toy_scene["gt_extrinsic"][None],
            toy_scene["intrinsic"][None],
            masks,
            valid,
            times,
            valid,
            masks.sum((-2, -1)),
            ["tip"],
            torch.zeros((6, 1, 1, 2), device="cuda"),
            None,
        )
        expected = times[:, None] * 0.1 if count == 2 else torch.full_like(q, 0.2)
        torch.testing.assert_close(q, expected)

    @pytest.mark.parametrize("upper", [False, True])
    @pytest.mark.parametrize("outward", [False, True])
    def test_active_joint_limits_preserve_camera_updates(self, toy_scene: dict, upper: bool, outward: bool):
        scene = toy_scene
        limit = scene["robot"].spec.actuated_joint_limits[0, int(upper)]
        q = torch.full((3, 1), float(limit), device="cuda")
        direction = (1 if upper else -1) * (1 if outward else -1)
        masks = hec_module.compute_robot_masks(
            scene["robot"], q + direction * 0.1, scene["intrinsic"], scene["gt_extrinsic"], _H, _W
        )[:, None]
        config = hec_config.QposTrajectoryCalibratorConfig(
            levels=[hec_config.PyramidLevel(1, 0, 5)],
            solver=PopulationSolverConfig(num_iters=1, num_candidates=[1], optimizer=_LM),
            optimize_camera=True,
        )
        helper = hec_module.QposTrajectoryCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)
        helper._build(
            q,
            scene["gt_extrinsic"][None],
            scene["intrinsic"][None],
            masks,
            torch.ones((3, 1), device="cuda"),
            torch.arange(3.0, device="cuda"),
            None,
            None,
            None,
        )
        term = helper._solver.terms[0][0]
        term.prepare(helper._initial_var, False, 0)
        raw = wp.to_torch(term.vectors)[:, 6, 0]
        assert (raw * direction < 0).all()
        normal, gradient = wp.to_torch(term.normal)[0], wp.to_torch(term.gradient)[0, :, 0]
        assert normal[3:, 3:].abs().sum() > 0
        if outward:
            assert torch.count_nonzero(normal[:3]) == 0
            assert torch.count_nonzero(normal[:, :3]) == 0
            assert torch.count_nonzero(gradient[:3]) == 0
        else:
            torch.testing.assert_close(gradient[:3], raw)
            assert normal[:3, :3].abs().sum() > 0

    def test_coupled_pyramid_patience(self):
        from robokit.helpers.hand_eye_calibration.qpos_trajectory_calibrator import _TrajectoryMasks

        wp.init()
        task = _TrajectoryMasks.__new__(_TrajectoryMasks)
        task.frames, task.terms, task.patience = 2, [], 2
        task._best_cost, task._last_improvement = float("inf"), 0
        accepted, costs = wp.zeros(1, dtype=wp.int32, device="cuda"), wp.ones(1, device="cuda")
        assert not task.on_step(accepted, 0, costs)
        assert not task.on_step(accepted, 1, costs)
        assert task.on_step(accepted, 2, costs)
        wp.to_torch(costs)[0] = 0.5
        assert not task.on_step(accepted, 2, costs)
        assert not task.on_step(accepted, 3, costs)
        assert task.on_step(accepted, 4, costs)

    def test_offset_search_cannot_raise_native_cost(self, toy_scene: dict, monkeypatch):
        scene = toy_scene
        q = torch.full((3, 1), 0.2, device="cuda")
        masks = hec_module.compute_robot_masks(scene["robot"], q, scene["intrinsic"], scene["gt_extrinsic"], _H, _W)[
            :, None
        ]
        config = hec_config.QposTrajectoryCalibratorConfig(
            levels=[hec_config.PyramidLevel(1, 0, 5)],
            solver=PopulationSolverConfig(num_iters=1, num_candidates=[1], optimizer=_LM),
            velocity_weight=0,
            acceleration_weight=0,
        )
        helper = hec_module.QposTrajectoryCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)

        def misleading_rank(paths, *args, **kwargs):
            scores = np.ones(len(paths))
            scores[1] = -1
            return scores, np.zeros(paths.shape[:2])

        monkeypatch.setattr(helper, "_rank_paths", misleading_rank)
        helper.solve(scene["gt_extrinsic"][None], masks, torch.arange(3.0, device="cuda"), q=q)
        torch.testing.assert_close(helper.last_q, q, atol=1e-5, rtol=0)
        assert helper.last_final_cost < 1e-6

    def test_global_offset_search_uses_images(self, toy_scene: dict, monkeypatch):
        """Recover a distant visible pose without a posture target or a supplied true q."""
        scene = dict(toy_scene)
        scene["robot"] = Robot.load(
            _URDF.replace("<visual><geometry>", '<visual><origin xyz="0 .1 0"/><geometry>').replace(
                'lower="-3.14" upper="3.14"', 'lower="-6.2831853" upper="6.2831853"'
            ),
            mesh_dir=scene["asset_directory"],
            load_meshes=True,
        )
        truth = torch.full((3, 1), -2.0, device="cuda")
        masks = hec_module.compute_robot_masks(
            scene["robot"], truth, scene["intrinsic"], scene["gt_extrinsic"], _H, _W
        )[:, None]
        config = hec_config.QposTrajectoryCalibratorConfig(
            levels=[hec_config.PyramidLevel(1, 0, 5)],
            solver=PopulationSolverConfig(num_iters=1, num_candidates=[1], optimizer=_LM),
            velocity_weight=0,
            acceleration_weight=0,
        )
        helper = hec_module.QposTrajectoryCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)
        banks = []
        rank = helper._rank_paths

        def record_candidates(paths, *args, **kwargs):
            banks.append(paths.copy())
            return rank(paths, *args, **kwargs)

        monkeypatch.setattr(helper, "_rank_paths", record_candidates)
        helper.solve(scene["gt_extrinsic"][None], masks, torch.arange(3.0, device="cuda"), q=torch.ones_like(truth))
        offsets = np.abs(banks[0][1:, 0, 0] - banks[0][0, 0, 0])
        assert offsets.max() >= 3.0 and offsets[offsets > 1e-8].min() < 0.02
        torch.testing.assert_close(helper.last_q, truth, atol=0.25, rtol=0)
        predicted = hec_module.compute_robot_masks(
            scene["robot"], helper.last_q, scene["intrinsic"], scene["gt_extrinsic"], _H, _W
        )[:, None]
        weights = masks.sum((-2, -1)).clamp(min=_H * _W * 0.05).reciprocal()
        expected = 0.5 * ((predicted - masks).square() * weights[..., None, None]).sum()
        assert helper.last_final_cost == pytest.approx(float(expected), abs=1e-5)
        helper.solve(scene["gt_extrinsic"][None], masks, torch.arange(3.0, device="cuda"), q=truth + 2 * torch.pi)
        torch.testing.assert_close(helper.last_q, truth, atol=0.25, rtol=0)

    def test_unobserved_joint_keeps_initial_state(self, toy_scene: dict):
        """A joint absent from the rendered geometry has no implied midpoint target."""
        scene = toy_scene
        config = hec_config.QposTrajectoryCalibratorConfig(
            levels=[hec_config.PyramidLevel(1, 0, 5)],
            solver=PopulationSolverConfig(num_iters=20, num_candidates=[1], optimizer=_LM),
            velocity_weight=0,
            acceleration_weight=0,
        )
        helper = hec_module.QposTrajectoryCalibrator(
            config, scene["robot"], scene["intrinsic"], _H, _W, render_link_names=["base"]
        )
        masks = hec_module.compute_robot_masks(
            scene["robot"],
            torch.zeros((3, 1), device="cuda"),
            scene["intrinsic"],
            scene["gt_extrinsic"],
            _H,
            _W,
            link_names=["base"],
        )
        helper.solve(
            scene["gt_extrinsic"][None],
            masks[:, None],
            torch.arange(3.0, device="cuda"),
            q=torch.ones((3, 1), device="cuda"),
        )
        torch.testing.assert_close(helper.last_q, torch.ones_like(helper.last_q), atol=1e-4, rtol=0)

    @pytest.mark.parametrize("build", [False, True])
    def test_additive_accumulation_capture(self, build: bool):
        """Nonzero shared buffers accumulate eagerly and on repeated graph replay."""
        from robokit.helpers.hand_eye_calibration.qpos_trajectory_calibrator import _TrajectoryMasks
        from robokit.lie.se3 import SE3Var
        from robokit.opt.var_values import VarKey, VarValues

        wp.init()
        with wp.ScopedStream(wp.Stream("cuda")):
            term = _TrajectoryMasks.__new__(_TrajectoryMasks)
            term.normal = wp.full((1, 3, 3), 2.0, device="cuda")
            term.gradient = wp.full((1, 3, 1), 3.0, device="cuda")
            term.current_cost = wp.full(1, 5.0, device="cuda")
            term.proposed_cost = wp.full(1, 7.0, device="cuda")
            normal = wp.empty_like(term.normal)
            gradient = wp.empty_like(term.gradient)
            costs = wp.empty_like(term.current_cost)
            variables = VarValues({VarKey.ROBOT: SE3Var(se3_identity(1, device="cuda"))})
            args = dict(costs=costs, JtJ=normal if build else None, Jtr=gradient if build else None)
            for capture in (False, True):
                normal.fill_(11.0)
                gradient.fill_(13.0)
                costs.fill_(17.0)
                if capture:
                    with wp.ScopedCapture() as graph:
                        term.accumulate_normal_equations(variables, **args)
                for repeat in range(1, 3):
                    if capture:
                        wp.capture_launch(graph.graph)
                    else:
                        term.accumulate_normal_equations(variables, **args)
                    wp.synchronize_stream()
                    for result, expected in (
                        (costs, 17.0 + repeat * (5.0 if build else 7.0)),
                        (normal, 11.0 + repeat * 2.0 if build else 11.0),
                        (gradient, 13.0 + repeat * 3.0 if build else 13.0),
                    ):
                        actual = wp.to_torch(result)
                        torch.testing.assert_close(actual, torch.full_like(actual, expected))

    @pytest.mark.parametrize("linear_solver", ["cholesky", "torch_cholesky"])
    def test_long_sequence_with_shared_camera_priors(self, offset_scene: dict, linear_solver: str):
        """96 two-joint frames and two cameras produce the 204-DOF regression."""
        times = torch.linspace(0, 4, 96, device="cuda")
        truth = torch.stack([0.2 * times.sin(), -0.3 * times.cos()], dim=-1)
        robot, k = offset_scene["robot"], offset_scene["intrinsic"]
        cameras = offset_scene["gt_extrinsic"][None].repeat(2, 1, 1)
        cameras[1, 0, 3] += 0.15
        masks = torch.stack([hec_module.compute_robot_masks(robot, truth, k, c, _H, _W) for c in cameras], dim=1)
        config = replace(
            hec_module.presets.rgb_qpos_trajectory_camera,
            levels=[hec_config.PyramidLevel(1, 0, 5)],
            solver=PopulationSolverConfig(
                num_iters=4, num_candidates=[1], optimizer=replace(_LM, linear_solver=linear_solver)
            ),
        )
        helper = hec_module.QposTrajectoryCalibrator(config, robot, k, _H, _W)
        refined = helper.solve(cameras, masks, times, q=truth.clone() + 0.03)
        assert torch.isfinite(refined).all() and torch.isfinite(helper.last_q).all()
        assert (helper.last_q - truth).abs().mean() < 0.03

    def test_mesh_compaction_preserves_render(self, toy_scene: dict):
        scene = toy_scene
        robot, q = scene["robot"], scene["q"]
        meshes = robot.spec.get_link_meshes(mode="visual")
        names = [n for n in robot.spec.link_names if n in meshes]
        indices = [robot.spec.link_names.index(n) for n in names]
        vertices = [torch.tensor(np.asarray(meshes[n].vertices), dtype=torch.float32, device="cuda") for n in names]
        faces = [torch.tensor(np.asarray(meshes[n].faces), dtype=torch.int32, device="cuda") for n in names]
        raw_faces, raw_vertices = mask_task._precompute_merged_mesh(
            vertices, faces, robot.forward_kinematics_via_matrix_torch(q)[:, indices]
        )
        raw, _ = mask_task._render_batch(
            mask_task.dr.RasterizeCudaContext(),
            scene["gt_extrinsic"],
            scene["intrinsic"],
            raw_faces,
            raw_vertices,
            _H,
            _W,
        )
        compact = hec_module.compute_robot_masks(robot, q, scene["intrinsic"], scene["gt_extrinsic"], _H, _W)
        torch.testing.assert_close(compact, raw, atol=1e-6, rtol=0)

    @pytest.mark.parametrize(
        "free_cameras, camera_prior_weight, initialization_frames, initialization_paths, initialization_repairs, initialization_seconds",
        [
            (False, 0.0, None, 1, 0, None),
            (True, 0.0, None, 1, 0, None),
            (True, 10000.0, None, 1, 0, None),
            (False, 0.0, 2, 1, 0, None),
            (True, 10000.0, 2, 1, 0, None),
            (False, 0.0, 2, 4, 0, None),
            (False, 0.0, 2, 4, 1, None),
            (False, 0.0, 2, 4, 1, 1e-9),
        ],
    )
    def test_recovers_sequence(
        self,
        toy_scene: dict,
        free_cameras: bool,
        camera_prior_weight: float,
        initialization_frames: Optional[int],
        initialization_paths: int,
        initialization_repairs: int,
        initialization_seconds: Optional[float],
    ):
        scene = toy_scene
        robot, k = scene["robot"], scene["intrinsic"]
        times = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.2], device="cuda")
        truth = (0.5 * times - 0.3)[:, None]
        cameras = scene["gt_extrinsic"][None].repeat(2, 1, 1)
        cameras[1, 0, 3] += 0.2
        intrinsics = k[None].repeat(2, 1, 1)
        intrinsics[1, 0, 0] *= 0.85
        masks = torch.stack(
            [hec_module.compute_robot_masks(robot, truth, intrinsics[v], cameras[v], _H, _W) for v in range(2)], dim=1
        )
        initial_cameras = cameras.clone()
        if free_cameras:
            initial_cameras[:, 0, 3] += 0.02
        frame = hec_config.HandEyeCalibratorConfig(
            optimize_q=True,
            optimize_camera=False,
            levels=[hec_config.PyramidLevel(4, 1.0, 5), hec_config.PyramidLevel(1, 0.0, 5)],
            solver=PopulationSolverConfig(num_iters=40, select_at=[15], num_candidates=[32, 4], optimizer=_LM),
            max_restarts=0,
        )
        config = hec_config.QposTrajectoryCalibratorConfig(
            frame=frame,
            initialization_frames=initialization_frames,
            initialization_paths=initialization_paths,
            initialization_repairs=initialization_repairs,
            initialization_seconds=initialization_seconds,
            velocity_weight=0.001,
            acceleration_weight=0.001,
            optimize_camera=free_cameras,
            camera_prior_weight=camera_prior_weight,
            solver=PopulationSolverConfig(num_iters=40, select_at=[20], num_candidates=[1, 1], optimizer=_LM),
        )
        helper = hec_module.QposTrajectoryCalibrator(config, robot, intrinsics, _H, _W)
        # Missing view differs from an observed, explicitly empty silhouette.
        weights = torch.ones((len(times), 2), device="cuda")
        weights[2, 1] = 0
        fitted_cameras = helper.solve(initial_cameras, masks, times, weights)
        repairs = initialization_repairs if initialization_seconds is None else 0
        assert len(helper.last_initialization_indices) == (
            len(times) if initialization_frames is None else initialization_frames + repairs
        )
        torch.testing.assert_close(helper.last_q, truth, atol=0.035, rtol=0)
        if camera_prior_weight:
            torch.testing.assert_close(fitted_cameras, initial_cameras, atol=1e-4, rtol=0)
        elif free_cameras:
            assert (fitted_cameras - cameras).square().sum() < (initial_cameras - cameras).square().sum() * 0.2
        else:
            torch.testing.assert_close(fitted_cameras, cameras, atol=0, rtol=0)
        limits = torch.as_tensor(robot.spec.actuated_joint_limits, device="cuda")
        assert ((helper.last_q >= limits[:, 0]) & (helper.last_q <= limits[:, 1])).all()
        assert helper.last_candidates.shape == (len(helper.last_initialization_indices), 5, 1)
        assert helper.last_candidate_costs.shape == (len(helper.last_initialization_indices), 5)
        assert len(helper.last_repair_indices) == repairs
        if repairs:
            assert helper.last_repair_indices[0] not in [0, len(times) - 1]
            initial = helper.last_q.clone()
            helper.solve(initial_cameras, masks, times, weights, q=initial.clone())
            torch.testing.assert_close(helper.last_initial_q, initial)
            assert helper.last_candidates is helper.last_candidate_costs is helper.last_path_scores is None
            assert helper.last_initialization_indices.size == 0 and helper.last_repair_indices == []

    def test_path_ranking_uses_intermediate_masks_and_missing_views(self, toy_scene: dict):
        scene = toy_scene
        times = torch.tensor([0.0, 0.3, 0.6, 1.0], device="cuda")
        truth = torch.tensor([[-0.4], [0.4], [0.4], [-0.4]], device="cuda")
        wrong = torch.full_like(truth, -0.4)
        masks = hec_module.compute_robot_masks(
            scene["robot"], truth, scene["intrinsic"], scene["gt_extrinsic"], _H, _W
        )[:, None].repeat(1, 2, 1, 1)
        weights = torch.ones((len(times), 2), device="cuda")
        weights[:, 1] = 0
        masks[:, 1] = 1 - masks[:, 1]
        config = hec_config.QposTrajectoryCalibratorConfig(velocity_weight=0, acceleration_weight=0)
        helper = hec_module.QposTrajectoryCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)
        costs, errors = helper._rank_paths(
            torch.stack([wrong, truth]).cpu().numpy(),
            scene["gt_extrinsic"][None].repeat(2, 1, 1),
            masks,
            times,
            weights,
            None,
            None,
            None,
        )
        assert costs[1] < costs[0] * 0.25
        assert errors[0, 1] > errors[1, 1] + 0.1

    def test_temporal_coupling_and_missing_frame(self, toy_scene: dict):
        scene = toy_scene
        times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device="cuda")
        truth = (times * 0.4 - 0.2)[:, None]
        masks = hec_module.compute_robot_masks(
            scene["robot"], truth, scene["intrinsic"], scene["gt_extrinsic"], _H, _W
        )[:, None]
        weights = torch.ones((5, 1), device="cuda")
        weights[2] = 0
        initial = truth.clone()
        initial[2] += 1.0
        config = hec_config.QposTrajectoryCalibratorConfig(
            velocity_weight=0.001,
            acceleration_weight=0.01,
        )
        helper = hec_module.QposTrajectoryCalibrator(config, scene["robot"], scene["intrinsic"], _H, _W)
        helper.solve(scene["gt_extrinsic"][None], masks, times, weights, q=initial)
        torch.testing.assert_close(helper.last_q, truth, atol=0.04, rtol=0)
        assert helper.last_q[-1] - helper.last_q[0] > 0.3
        with pytest.raises(ValueError, match="timestamps"):
            helper.solve(scene["gt_extrinsic"][None], masks, times.flip(0), q=initial)
        with pytest.raises(ValueError, match="within joint limits"):
            helper.solve(scene["gt_extrinsic"][None], masks, times, q=initial + 100)


@_REQUIRES_CUDA
class TestKeypointAlignment:
    @pytest.mark.parametrize("empty", [False, True])
    def test_iou_ranking_differs_from_pixel_mse(self, empty: bool):
        from robokit.helpers.hand_eye_calibration.calibrator import _MaskCandidateScore
        from robokit.opt.var_values import VarKey, VarValues, VectorVar

        term = mask_task.MaskAlignmentNormalEqTask()
        term.batch_size, term.N, term.H, term.W = 2, 1, 1, 8
        term.target_masks = torch.tensor([[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], device="cuda")
        if empty:
            term.target_masks.zero_()
        term.observation_weights = torch.ones(1, device="cuda")
        term._rendered_curr = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]], device="cuda"
        )
        term._curr_is_fresh = True
        values = VarValues({VarKey.ROBOT: VectorVar(wp.zeros((2, 1), device="cuda"))})
        score = wp.to_torch(_MaskCandidateScore([term]).compute_weighted_residual(values))[:, 0].square() / 2
        mse = (term._rendered_curr - term.target_masks).square().sum((-2, -1)) / 2
        assert mse[0] < mse[1]
        if empty:
            torch.testing.assert_close(score, mse)
        else:
            assert score[1] < score[0]

    def test_camera_and_joint_jacobian(self, toy_scene: dict):
        from robokit.helpers.hand_eye_calibration.keypoint_alignment_task import KeypointAlignmentTask
        from robokit.lie.se3 import SE3Var, se3_from_matrix
        from robokit.opt.var_values import VarKey, VarValues

        robot = toy_scene["robot"]
        q = torch.tensor([[0.35], [-0.25]], device="cuda")
        camera = toy_scene["gt_extrinsic"][None].repeat(2, 1, 1)
        mount = torch.eye(4, device="cuda")[None].repeat(2, 1, 1)
        mount[1, 0, 3] = 0.2
        pixels = torch.tensor(
            [[[75.0, 78.0], [90.0, 60.0]], [[65.0, 80.0], [float("inf"), float("nan")]]], device="cuda"
        )
        term = KeypointAlignmentTask(robot, ["tip", "arm"], pixels, toy_scene["intrinsic"], T_mount_base=mount)
        values = VarValues(
            {
                VarKey.ROBOT: SE3Var(se3_from_matrix(wp.from_torch(camera, dtype=wp.mat44))),
                VarKey.QPOS_OFFSET: robot.state(wp.from_torch(q)),
            }
        )
        term.prepare(values, False, 0)
        analytic = term.jacobian.clone()
        numerical = torch.empty_like(analytic)
        epsilon = 0.001
        for column in range(values.tangent_dim):
            step = torch.zeros((2, values.tangent_dim), device="cuda")
            step[:, column] = epsilon
            plus, minus = values.integrate(wp.from_torch(step)), values.integrate(wp.from_torch(-step))
            a = wp.to_torch(term.compute_weighted_residual(plus)).clone()
            b = wp.to_torch(term.compute_weighted_residual(minus)).clone()
            numerical[:, :, column] = (a - b) / (2 * epsilon)
        torch.testing.assert_close(analytic, numerical, atol=0.002, rtol=0.02)
        torch.testing.assert_close(analytic[:, -2:], torch.zeros_like(analytic[:, -2:]), atol=0, rtol=0)

    def test_unknown_sequence_with_missing_masks(self, toy_scene: dict):
        robot = toy_scene["robot"]
        times = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.2], device="cuda")
        truth = (0.5 * times - 0.3)[:, None]
        cameras = toy_scene["gt_extrinsic"][None].repeat(2, 1, 1)
        cameras[1, 0, 3] += 0.2
        k = toy_scene["intrinsic"]
        masks = torch.stack([hec_module.compute_robot_masks(robot, truth, k, c, _H, _W) for c in cameras], dim=1)
        world = robot.forward_kinematics_via_matrix_torch(truth)[:, robot.spec.link_names.index("tip"), :3, 3]
        points = torch.einsum("vij,tj->tvi", cameras[:, :3, :3], world) + cameras[None, :, :3, 3]
        pixels = (points[..., :2] / points[..., 2:] * torch.diag(k)[:2] + k[:2, 2])[:, :, None]
        pixels[2, 1] = float("nan")
        weights = torch.zeros((5, 2), device="cuda")
        weights[0] = 1
        frame = hec_config.HandEyeCalibratorConfig(
            optimize_q=True,
            optimize_camera=False,
            levels=[hec_config.PyramidLevel(4, 1.0, 5), hec_config.PyramidLevel(1, 0.0, 5)],
            solver=PopulationSolverConfig(num_iters=30, select_at=[10], num_candidates=[32, 4], optimizer=_LM),
            max_restarts=0,
            keypoint_warmup_iters=8,
            candidate_metric="iou",
        )
        config = hec_config.QposTrajectoryCalibratorConfig(
            frame=frame, velocity_weight=0.001, acceleration_weight=0.001
        )
        helper = hec_module.QposTrajectoryCalibrator(config, robot, k, _H, _W)
        fitted_camera = helper.solve(cameras, masks, times, weights, keypoint_links=["tip"], target_keypoints=pixels)
        torch.testing.assert_close(helper.last_q, truth, atol=0.03, rtol=0)
        torch.testing.assert_close(fitted_camera, cameras, atol=0, rtol=0)
        assert float(helper.last_q[-1] - helper.last_q[0]) > 0.5
