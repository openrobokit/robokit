# pyright: reportArgumentType=false
import pytest
import torch

from robokit.geom.sample_farthest_points import sample_farthest_points


pytestmark = pytest.mark.torch


def fps_reference(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Reference PyTorch implementation for correctness verification."""
    n = points.shape[0]
    indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    min_dists = torch.full((n,), float("inf"), device=points.device)

    indices[0] = 0
    for i in range(1, num_samples):
        selected = points[indices[i - 1]]
        dists = ((points - selected) ** 2).sum(dim=-1)
        min_dists = torch.minimum(min_dists, dists)
        indices[i] = min_dists.argmax()

    return indices


class TestFarthestPointSampling:
    def test_fps_single_batch(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        points = torch.rand(100, 3, device=device)
        points_offsets = torch.tensor([0, 100], device=device, dtype=torch.int32)
        num_samples = torch.tensor([10], device=device, dtype=torch.int32)

        result = sample_farthest_points(points, points_offsets, num_samples)
        reference = fps_reference(points, 10)

        assert result.shape == (1, 10)
        assert torch.equal(result[0].to(torch.long), reference)

    def test_fps_batched(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        points = torch.rand(300, 3, device=device)
        points_offsets = torch.tensor([0, 100, 200, 300], device=device, dtype=torch.int32)
        num_samples = torch.tensor([10, 20, 15], device=device, dtype=torch.int32)

        result = sample_farthest_points(points, points_offsets, num_samples)

        assert result.shape == (3, 20)
        assert torch.equal(result[0, :10].to(torch.long), fps_reference(points[:100], 10))
        assert torch.equal(result[1, :20].to(torch.long), fps_reference(points[100:200], 20))
        assert torch.equal(result[2, :15].to(torch.long), fps_reference(points[200:], 15))

    def test_fps_uniqueness(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        points = torch.rand(50, 3, device=device)
        points_offsets = torch.tensor([0, 50], device=device, dtype=torch.int32)
        num_samples = torch.tensor([30], device=device, dtype=torch.int32)

        result = sample_farthest_points(points, points_offsets, num_samples)

        assert len(torch.unique(result[0])) == 30

    def test_fps_edge_case_k1(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        points = torch.rand(100, 3, device=device)
        points_offsets = torch.tensor([0, 100], device=device, dtype=torch.int32)
        num_samples = torch.tensor([1], device=device, dtype=torch.int32)

        result = sample_farthest_points(points, points_offsets, num_samples)
        assert result[0, 0] == 0

    def test_fps_various_sizes(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for n_points in [5, 17, 64, 128, 257, 500, 1024]:
            torch.manual_seed(42)
            points = torch.rand(n_points, 3, device=device)
            points_offsets = torch.tensor([0, n_points], device=device, dtype=torch.int32)
            k = min(n_points, 20)
            num_samples = torch.tensor([k], device=device, dtype=torch.int32)

            result = sample_farthest_points(points, points_offsets, num_samples)
            reference = fps_reference(points, k)

            assert result.shape == (1, k)
            assert torch.equal(result[0].to(torch.long), reference), f"Failed for n_points={n_points}"


if __name__ == "__main__":
    test_fps_single_batch()
    test_fps_batched()
    test_fps_uniqueness()
    test_fps_edge_case_k1()
    test_fps_various_sizes()
    print("All tests passed!")
