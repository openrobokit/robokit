"""Guard against MotionPlanner CUDA-graph memory leaks (#246) and frees during capture.

Regressions:
- Rebuilds must release graph memory, which torch/warp allocators do not track.
- GC during capture must not free a dropped planner's SDF volumes and invalidate that capture.
"""

import ctypes
import gc
import weakref
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import trimesh
import warp as wp

from robokit.geom import BoxGeom, MeshGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.utils.warp_utils import wp_vec7


# Use driver CUDA memory attributes; libcudart is absent from some Warp builds, but libcuda.so.1 is present.
_CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0

_WARP_VERSION = tuple(int(x) for x in wp.__version__.split(".")[:2])


@pytest.mark.slow
@pytest.mark.skipif(_WARP_VERSION < (1, 9), reason="warp < 1.9 cannot capture the MotionPlanner solve at all")
class TestMotionPlannerGraphMemory:
    def test_repeated_construction_does_not_grow_graph_memory(self, panda_robot_with_collision):
        if not wp.get_cuda_device_count():
            pytest.skip("CUDA device required")  # CI runners are CPU-only
        device = wp.get_device("cuda:0")
        robot = panda_robot_with_collision

        box = trimesh.creation.box(extents=(0.08, 0.5, 0.4))
        pose = np.eye(4, dtype=np.float32)[None]
        pose[0, :3, 3] = (0.45, 0.0, 0.2)
        rng = np.random.default_rng(0)
        limits = robot.spec.actuated_joint_limits
        mid = 0.5 * (limits[:, 0] + limits[:, 1])
        span = 0.2 * (limits[:, 1] - limits[:, 0])
        batch = 4

        driver = ctypes.CDLL("libcuda.so.1")
        samples = []
        for _ in range(4):
            # Create a fresh planner per changed scene to reproduce the caller pattern that leaked.
            scene = WarpScene(1, device).add(
                MeshGeom(
                    [box],
                    np.array([0, 1], dtype=np.int32),
                    poses=wp.from_numpy(pose, dtype=wp.mat44, device=device),
                )
            )
            planner = MotionPlanner(
                presets.offline_joint_lbfgs,
                robot,
                "panda_hand",
                scene=scene,
                device=str(device),
            )
            start_q = (mid + rng.uniform(-1, 1, (batch, mid.shape[0])) * span).astype(np.float32)
            target_q = (mid + rng.uniform(-1, 1, (batch, mid.shape[0])) * span).astype(np.float32)
            planner.solve_offline_numpy([start_q, target_q])  # launches the graph
            wp.synchronize_device(device)
            value = ctypes.c_size_t(0)
            attr = _CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT
            rc = driver.cuDeviceGetGraphMemAttribute(ctypes.c_int(device.ordinal), attr, ctypes.byref(value))
            assert rc == 0, f"cuDeviceGetGraphMemAttribute failed with {rc}"
            samples.append(value.value)

        assert samples[1:] == samples[:-1], f"graph memory grew across constructions: {samples}"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("reuse_scene", [False, True])
    @pytest.mark.parametrize("preset_name", ["offline_pose_lbfgs", "online_pose_lbfgs", "online_pose_mppi"])
    def test_reused_scene_releases_discarded_planners(
        self, panda_robot_with_collision, device: str, reuse_scene: bool, preset_name: str
    ):
        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA device required")
        robot = panda_robot_with_collision
        config = deepcopy(getattr(presets, preset_name))
        config.num_timesteps = 3
        config.max_retries = 0
        config.goal_ik.solver = PopulationSolverConfig(
            num_iters=1, num_candidates=[1], optimizer=LMOptimizerConfig(), cuda_graph_mode="full"
        )
        stage = config.stages[0]
        mppi = stage.method == "mppi"
        options = (
            {
                "solver": replace(
                    stage.solver,
                    num_iters=1,
                    num_candidates=[1],
                    samplers=[replace(stage.solver.samplers[0], num_draws=4)],
                )
            }
            if mppi
            else {"solver": replace(stage.solver, num_iters=1, num_candidates=[3])}
        )
        config.stages = (replace(stage, **options),)
        scene = WarpScene(1, device)
        q0 = robot.spec.midrange_q.astype(np.float32)[None]
        start = robot.forward_kinematics(robot.state(q=wp.array(q0, dtype=wp.float32, device=device)))
        target = wp.array(
            start.T_world_link.numpy()[:, robot.link_names.index("panda_hand")], dtype=wp_vec7, device=device
        )
        driver = ctypes.CDLL("libcuda.so.1") if device.startswith("cuda") else None
        graph_bytes = []
        for cycle in range(4):
            if not reuse_scene or cycle == 0:
                scene = WarpScene(1, device).add(BoxGeom(np.full((1, 3), 0.01, np.float32), np.array([0, 1], np.int32)))
            planner = MotionPlanner(config, robot, "panda_hand", scene=scene, device=device)
            if preset_name.startswith("offline"):
                planner.solve_offline([start.q, target])
            else:
                planner.solve_online(start, T_world_target=target)
            wp.synchronize_device(device)
            refs = [
                weakref.ref(planner),
                weakref.ref(planner.ik),
                weakref.ref(planner._trajectory_optimizer._solver),
            ]
            if not reuse_scene:
                refs.append(weakref.ref(scene))
            del planner
            if not reuse_scene:
                del scene
            gc.collect()
            wp.synchronize_device(device)
            assert all(ref() is None for ref in refs), [type(ref()).__name__ for ref in refs if ref() is not None]
            if driver is not None:
                value = ctypes.c_size_t(0)
                rc = driver.cuDeviceGetGraphMemAttribute(
                    ctypes.c_int(wp.get_device(device).ordinal),
                    _CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
                    ctypes.byref(value),
                )
                assert rc == 0
                graph_bytes.append(value.value)
        if graph_bytes:
            assert len(set(graph_bytes)) == 1, graph_bytes

    def test_gc_during_capture_after_dropping_a_planner(self, panda_robot_with_collision, monkeypatch):
        if not wp.get_cuda_device_count():
            pytest.skip("CUDA device required")
        device = wp.get_device("cuda:0")
        robot = panda_robot_with_collision
        box = trimesh.creation.box(extents=(0.08, 0.5, 0.4))
        rng = np.random.default_rng(0)
        limits = robot.spec.actuated_joint_limits
        mid = 0.5 * (limits[:, 0] + limits[:, 1])
        span = 0.2 * (limits[:, 1] - limits[:, 0])
        start_q = (mid + rng.uniform(-1, 1, (4, mid.shape[0])) * span).astype(np.float32)
        target_q = (mid + rng.uniform(-1, 1, (4, mid.shape[0])) * span).astype(np.float32)
        enter = wp.ScopedCapture.__enter__

        def enter_then_collect(capture: wp.ScopedCapture) -> wp.ScopedCapture:
            out = enter(capture)
            gc.collect()  # a GC pass at the worst moment: every capture is live
            return out

        for i in range(2):  # rebuilding drops the first planner and its SDF scene before the second one captures
            scene = WarpScene(1, device).add(MeshGeom([box], np.array([0, 1], dtype=np.int32), enable_sdf=True))
            planner = MotionPlanner(presets.offline_joint_lbfgs, robot, "panda_hand", scene=scene, device=str(device))
            if i == 1:
                monkeypatch.setattr(wp.ScopedCapture, "__enter__", enter_then_collect)
            planner.solve_offline_numpy([start_q, target_q])
            wp.synchronize_device(device)
