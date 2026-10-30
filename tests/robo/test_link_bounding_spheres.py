import numpy as np

from robokit.robo.robot_spec import _compute_link_bounding_spheres


class TestComputeLinkBoundingSpheres:
    """Per-link Ritter bounding spheres: enclosure, edge cases, and link partitioning."""

    def test_conservativeness(self):
        rng = np.random.default_rng(0)
        num_links = 4
        centers = rng.uniform(-1.0, 1.0, size=(40, 3)).astype(np.float32)
        radii = rng.uniform(0.01, 0.2, size=40).astype(np.float32)
        link_indices = rng.integers(0, num_links, size=40).astype(np.int32)

        bound_centers, bound_radii = _compute_link_bounding_spheres(centers, radii, link_indices, num_links)

        assert bound_centers.shape == (num_links, 3)
        assert bound_radii.shape == (num_links,)
        for i in range(len(radii)):
            link = int(link_indices[i])
            dist = float(np.linalg.norm(centers[i] - bound_centers[link]))
            assert dist + float(radii[i]) <= float(bound_radii[link]) + 1e-5

    def test_single_sphere_per_link(self):
        centers = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        radii = np.array([0.5], dtype=np.float32)
        link_indices = np.array([0], dtype=np.int32)

        bound_centers, bound_radii = _compute_link_bounding_spheres(centers, radii, link_indices, 1)

        assert np.allclose(bound_centers[0], [1.0, 2.0, 3.0])
        assert np.isclose(bound_radii[0], 0.5)

    def test_nested_spheres_use_larger(self):
        # a small sphere fully inside a big one -> bound is exactly the big sphere
        centers = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([1.0, 0.1], dtype=np.float32)
        link_indices = np.array([0, 0], dtype=np.int32)

        bound_centers, bound_radii = _compute_link_bounding_spheres(centers, radii, link_indices, 1)

        assert np.allclose(bound_centers[0], [0.0, 0.0, 0.0], atol=1e-6)
        assert np.isclose(bound_radii[0], 1.0, atol=1e-6)

    def test_empty_input_is_zero(self):
        bound_centers, bound_radii = _compute_link_bounding_spheres(
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            3,
        )
        assert np.array_equal(bound_centers, np.zeros((3, 3), dtype=np.float32))
        assert np.array_equal(bound_radii, np.zeros(3, dtype=np.float32))

    def test_links_are_partitioned(self):
        # link 0 spheres are far away; link 1's bound must depend only on link 1's sphere
        centers = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([0.3, 0.3, 0.25], dtype=np.float32)
        link_indices = np.array([0, 0, 1], dtype=np.int32)

        bound_centers, bound_radii = _compute_link_bounding_spheres(centers, radii, link_indices, 2)

        assert np.allclose(bound_centers[1], [0.0, 0.0, 0.0])
        assert np.isclose(bound_radii[1], 0.25)
