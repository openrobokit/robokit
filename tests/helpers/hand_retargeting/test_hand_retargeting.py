"""Public hand-retargeting configuration and preset invariants."""

from dataclasses import is_dataclass, replace

import numpy as np
import warp as wp

from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandRetargeterOnline, config, presets
from robokit.robo import Robot


_PRESET_MODULES = (
    presets.ability,
    presets.allegro,
    presets.barrett,
    presets.dclaw,
    presets.inspire,
    presets.leap,
    presets.paxini,
    presets.shadow,
    presets.sharpa,
    presets.svh,
    presets.xarm7_ability,
    presets.xarm7_mano,
    presets.xarm7_xhand,
    presets.xhand,
)


class TestHandRetargetingPresets:
    """Preset metadata stays unit-testable; one fixture robot covers solver wiring."""

    def test_public_config_dataclasses(self) -> None:
        public_dataclasses = {
            name
            for name in config.__all__
            if is_dataclass(getattr(config, name)) and getattr(config, name).__module__ == config.__name__
        }
        assert public_dataclasses == {
            "HandRetargeterOfflineConfig",
            "HandRetargeterOnlineConfig",
            "HandSpec",
        }

    def test_all_named_topologies_are_internally_consistent(self) -> None:
        targets = set(presets.TARGET_NAMES)
        for module in _PRESET_MODULES:
            for name in module.__all__:
                value = getattr(module, name)
                if isinstance(value, config.HandSpec):
                    assert set(value.target_names) == targets
                    assert set(value.target_link_names) <= targets
                    assert value.root_target_name in value.target_link_names
                    assert set(value.contact_target_names) <= targets
                elif isinstance(value, config.HandRetargeterOnlineConfig):
                    # presets list only their active terms, so either vector term may be absent
                    weights = {term.name: term.weight for term in value.terms if term.name in ("vectors", "directions")}
                    vectors, directions = weights.get("vectors", {}), weights.get("directions", {})
                    assert {target for pair in (*vectors, *directions) for target in pair[2:]} <= targets
                    assert set(value.pinch_correspondences) <= set(vectors)

    def test_fixture_topology_resolves_and_solves(self, shadow_hand_robot: Robot) -> None:
        robot = shadow_hand_robot
        spec = presets.shadow.spec
        assert set(spec.target_link_names.values()) <= set(robot.spec.link_names)

        online = presets.shadow.online
        damping = online.solver.optimizer  # the final segment's, when it varies per segment
        online_solver = replace(
            online.solver,
            num_iters=0,
            select_at=[],
            num_candidates=[1],
            optimizer=damping[-1] if isinstance(damping, (list, tuple)) else damping,
            cuda_graph_mode="none",
        )
        online_helper = HandRetargeterOnline(replace(online, solver=online_solver), robot, spec, device="cpu")
        online_helper.warmup(1)
        online_helper.solve(wp.zeros((1, len(spec.target_names), 3), dtype=wp.float32, device="cpu"))

        offline = presets.shadow.offline
        offline_solver = replace(offline.solver, max_iter=0, early_stopping_interval=0, cuda_graph_mode="none")
        offline_helper = HandRetargeterOffline(replace(offline, solver=offline_solver), robot, spec, device="cpu")
        offline_helper.warmup(1, 3)
        points = wp.zeros((1, 3, len(spec.target_names), 3), dtype=wp.float32, device="cpu")
        root_quat_wxyz = np.zeros((1, 3, 4), dtype=np.float32)
        root_quat_wxyz[:, :, 0] = 1.0
        offline_helper.solve(points, wp.from_numpy(root_quat_wxyz, dtype=wp.float32, device="cpu"))
