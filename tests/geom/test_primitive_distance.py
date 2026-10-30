import numpy as np
import warp as wp

from robokit.geom.sdf_kernels import closest_segment_to_segment_func


@wp.kernel
def _segment_distance_kernel(
    a1: wp.array(dtype=wp.vec3),
    b1: wp.array(dtype=wp.vec3),
    a2: wp.array(dtype=wp.vec3),
    b2: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    cp = closest_segment_to_segment_func(a1[i], b1[i], a2[i], b2[i])
    out[i] = wp.length(cp.c1 - cp.c2)


def _brute_force_segment_distance(a1, b1, a2, b2, n: int = 200) -> float:
    """Reference closest distance between two segments via a dense grid over (s, t)."""
    s = np.linspace(0.0, 1.0, n)[:, None, None]
    t = np.linspace(0.0, 1.0, n)[None, :, None]
    p = a1 + s * (b1 - a1)
    q = a2 + t * (b2 - a2)
    return float(np.linalg.norm(p - q, axis=-1).min())


class TestClosestSegmentToSegment:
    def test_matches_brute_force(self):
        wp.init()
        device = wp.get_device("cpu")
        rng = np.random.default_rng(0)
        num = 64
        a1 = rng.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)
        b1 = rng.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)
        a2 = rng.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)
        b2 = rng.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)

        out = wp.empty(num, dtype=wp.float32, device=device)
        wp.launch(
            kernel=_segment_distance_kernel,
            dim=num,
            inputs=[
                wp.from_numpy(a1, dtype=wp.vec3, device=device),
                wp.from_numpy(b1, dtype=wp.vec3, device=device),
                wp.from_numpy(a2, dtype=wp.vec3, device=device),
                wp.from_numpy(b2, dtype=wp.vec3, device=device),
            ],
            outputs=[out],
            device=device,
        )
        got = out.numpy()
        ref = np.array([_brute_force_segment_distance(a1[i], b1[i], a2[i], b2[i]) for i in range(num)])
        # The warp result is exact; the brute-force grid slightly overestimates, so allow grid slack.
        assert np.all(got <= ref + 1e-6)
        assert np.allclose(got, ref, atol=2e-2)

    def test_parallel_segments(self):
        wp.init()
        device = wp.get_device("cpu")
        # Two parallel unit segments offset by 0.5 along y; closest distance is 0.5.
        a1 = np.array([[0.0, 0.0, 0.0]], np.float32)
        b1 = np.array([[1.0, 0.0, 0.0]], np.float32)
        a2 = np.array([[0.0, 0.5, 0.0]], np.float32)
        b2 = np.array([[1.0, 0.5, 0.0]], np.float32)
        out = wp.empty(1, dtype=wp.float32, device=device)
        wp.launch(
            kernel=_segment_distance_kernel,
            dim=1,
            inputs=[
                wp.from_numpy(a1, dtype=wp.vec3, device=device),
                wp.from_numpy(b1, dtype=wp.vec3, device=device),
                wp.from_numpy(a2, dtype=wp.vec3, device=device),
                wp.from_numpy(b2, dtype=wp.vec3, device=device),
            ],
            outputs=[out],
            device=device,
        )
        assert abs(float(out.numpy()[0]) - 0.5) < 1e-5
