from typing import List, Tuple

import numpy as np
import pytest

from robokit.utils.ee_coord_utils import (
    AXIS,
    AXIS_TO_VEC,
    GripperCoordinateSpec,
    HandCoordinateSpec,
    MountCoordinateSpec,
    ee_coord_conversion,
)


_AXES: List[Tuple[AXIS, AXIS]] = [("x", "z"), ("y", "-z"), ("-x", "-y"), ("-y", "z"), ("z", "-x"), ("y", "z")]


class TestEECoordinates:
    def test_coord_conversion_return_tensors_pt_matches_np(self):
        torch = pytest.importorskip("torch")
        specs = [
            kind(a, b) for kind in (HandCoordinateSpec, GripperCoordinateSpec, MountCoordinateSpec) for a, b in _AXES
        ]
        for src in specs:
            for dst in specs:
                np_mat = ee_coord_conversion(src, dst)
                pt_mat = ee_coord_conversion(src, dst, return_tensors="pt")
                torch.testing.assert_close(pt_mat, torch.from_numpy(np_mat))

    def test_hand_and_gripper_mount_semantics(self):
        hand = HandCoordinateSpec(palm_forward_axis="z", four_fingers_up_axis="y")
        gripper = GripperCoordinateSpec(outward_axis="x", opening_axis="-z")
        assert hand.mount_spec == MountCoordinateSpec(outward_axis="y", right_axis="-x")
        assert gripper.mount_spec == MountCoordinateSpec(outward_axis="x", right_axis="-z")

    def test_mount_points_propagate_without_affecting_rotation(self):
        point = (0.03, -0.02, 0.05)
        hand = HandCoordinateSpec("z", "y", mount_point_xyz=point)
        gripper = GripperCoordinateSpec("x", "-z", mount_point_xyz=point)
        assert hand.mount_spec.mount_point_xyz == point
        assert gripper.mount_spec.mount_point_xyz == point
        np.testing.assert_allclose(
            ee_coord_conversion(hand, gripper),
            ee_coord_conversion(HandCoordinateSpec("z", "y"), GripperCoordinateSpec("x", "-z")),
        )

    @pytest.mark.parametrize("point", [(0.0, float("nan"), 0.0), (0.0, 0.0, float("inf"))])
    def test_mount_points_must_be_finite(self, point: Tuple[float, float, float]):
        for spec in (MountCoordinateSpec, GripperCoordinateSpec, HandCoordinateSpec):
            with pytest.raises(ValueError, match="three finite coordinates"):
                spec("z", "x", mount_point_xyz=point)

    @pytest.mark.parametrize("src_type", [HandCoordinateSpec, GripperCoordinateSpec, MountCoordinateSpec])
    @pytest.mark.parametrize("dst_type", [HandCoordinateSpec, GripperCoordinateSpec, MountCoordinateSpec])
    def test_conversion_aligns_outward_and_right(self, src_type: type, dst_type: type):
        src = src_type("z", "x")
        dst = dst_type("x", "-y")
        R_dst_src = ee_coord_conversion(src, dst)
        assert isinstance(R_dst_src, np.ndarray)
        src = src if isinstance(src, MountCoordinateSpec) else src.mount_spec
        dst = dst if isinstance(dst, MountCoordinateSpec) else dst.mount_spec
        assert np.allclose(R_dst_src @ AXIS_TO_VEC[src.outward_axis], AXIS_TO_VEC[dst.outward_axis])
        assert np.allclose(R_dst_src @ AXIS_TO_VEC[src.right_axis], AXIS_TO_VEC[dst.right_axis])
        np.testing.assert_allclose(R_dst_src @ R_dst_src.T, np.eye(3))
        assert np.isclose(np.linalg.det(R_dst_src), 1.0)

    def test_axes_must_be_orthogonal(self):
        with pytest.raises(ValueError, match="orthogonal"):
            MountCoordinateSpec(outward_axis="x", right_axis="-x")
        with pytest.raises(ValueError, match="orthogonal"):
            GripperCoordinateSpec(outward_axis="z", opening_axis="-z")
