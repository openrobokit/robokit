"""Spec-level self-collision ignore: parse loads the YAML verbatim, tasks use it."""

from pathlib import Path
from typing import Optional, Set, Tuple

from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot
from robokit.terms.dense.self_collision_task import SelfCollisionTask


_FIXTURES = Path(__file__).parent.parent / "fixtures"

_IGNORE_YAML = """
self_collision_ignore:
  panda_link0:
  - panda_link2
  - not_a_link
  panda_hand:
  - panda_link1
"""


def _write_ignore(tmp_path: Path) -> Path:
    path = tmp_path / "ignore.sphere.yml"
    path.write_text(_IGNORE_YAML)
    return path


def _load_panda(ignore_path: Optional[Path] = None) -> Robot:
    return Robot.load(
        load_robot_description("panda_description"),
        load_collision_spheres=True,
        collision_spheres_path=str(_FIXTURES / "franka_collision_spheres.yaml"),
        self_collision_ignore_path=ignore_path,
    )


class TestSelfCollisionIgnore:
    def test_parse_loads_ignore_pairs_verbatim(self, tmp_path: Path) -> None:
        robot = _load_panda(_write_ignore(tmp_path))
        names = robot.spec.link_names
        expected: Set[Tuple[int, int]] = {
            (names.index("panda_link0"), names.index("panda_link2")),
            (names.index("panda_hand"), names.index("panda_link1")),
        }
        # verbatim: unknown link names are skipped, no adjacency pairs are unioned in
        assert {(int(a), int(b)) for a, b in robot.spec.self_collision_ignored_pairs} == expected

    def test_parse_without_path_leaves_field_empty(self) -> None:
        assert _load_panda().spec.self_collision_ignored_pairs.shape == (0, 2)

    def test_task_uses_spec_pairs(self, tmp_path: Path) -> None:
        robot = _load_panda(_write_ignore(tmp_path))
        plain = _load_panda()
        task = SelfCollisionTask(robot=robot, weight=1.0)
        adjacency_only = SelfCollisionTask(plain, weight=1.0)
        # the yaml pairs prune on top of the adjacency filter
        assert 0 < task._num_pairs < adjacency_only._num_pairs
        gl = robot.spec.collision_spheres_link_indices
        names = robot.spec.link_names
        yaml_pair = {names.index("panda_link0"), names.index("panda_link2")}
        task_link_pairs = {frozenset((int(gl[i]), int(gl[j]))) for i, j in task._pair_indices_np}
        assert yaml_pair in {frozenset((int(gl[i]), int(gl[j]))) for i, j in adjacency_only._pair_indices_np}
        assert yaml_pair not in task_link_pairs

    def test_empty_ignore_mapping_loads_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yml"
        path.write_text("self_collision_ignore:\n")
        assert _load_panda(path).spec.self_collision_ignored_pairs.shape == (0, 2)

    def test_filter_off_ignores_spec_pairs(self, tmp_path: Path) -> None:
        robot = _load_panda(_write_ignore(tmp_path))
        task = SelfCollisionTask(robot=robot, weight=1.0, filter_adjacent_links=False)
        n = len(robot.spec.collision_spheres_link_indices)
        assert task._num_pairs == n * (n - 1) // 2
