"""Visualize robot collision spheres in world coordinates with viser, with joint control sliders.

Usage:
    uv run scripts/geom/spheres/visualize_collision_spheres.py --urdf <path> --collision-spheres <path>

Reference:
    https://viser.studio/main/examples/demos/urdf_visualizer/
"""

import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

from robokit.robo import Robot


def create_joint_sliders(server: viser.ViserServer, viser_urdf: ViserUrdf) -> List:
    """Create GUI sliders for each actuated joint.

    Args:
        server: Viser server instance.
        viser_urdf: ViserUrdf instance with joint limits.

    Returns:
        List of slider handles.
    """
    sliders = []
    joint_limits = viser_urdf.get_actuated_joint_limits()

    for joint_name, (lower, upper) in joint_limits.items():
        # handle unbounded or None limits
        lower_val: float = -np.pi if lower is None or lower == float("-inf") else lower
        upper_val: float = np.pi if upper is None or upper == float("inf") else upper
        initial = 0.0 if lower is None or upper is None else (lower_val + upper_val) / 2.0

        slider = server.gui.add_slider(
            label=joint_name,
            min=lower_val,
            max=upper_val,
            step=1e-3,
            initial_value=initial,
        )
        sliders.append(slider)

    return sliders


def update_collision_spheres(
    server: viser.ViserServer,
    robot: Robot,
    q: np.ndarray,
    unit_sphere_mesh: trimesh.Trimesh,
    sphere_handle: Optional[viser.BatchedMeshHandle],
    sphere_color: Tuple[float, float, float],
) -> viser.BatchedMeshHandle:
    """Update collision sphere positions after FK using batched rendering.

    Args:
        server: Viser server instance.
        robot: Robot instance with collision spheres loaded.
        q: Joint configuration array.
        unit_sphere_mesh: A unit sphere trimesh used as the base mesh.
        sphere_handle: Existing batched mesh handle, or None to create new.
        sphere_color: RGB color for the spheres in 0-1 float range.

    Returns:
        Batched mesh handle for the collision spheres.
    """
    state = robot.state(q=q)
    state = robot.forward_kinematics(state)

    state = robot.transform_collision_spheres(state)
    positions = state.collision_sphere_centers_world.numpy()[0]  # [num_spheres, 3]
    radii = robot.spec.collision_sphere_radii

    num_spheres = len(radii)
    wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_spheres, 1))

    if sphere_handle is not None:
        sphere_handle.remove()

    return server.scene.add_batched_meshes_simple(
        "/collision_spheres",
        vertices=unit_sphere_mesh.vertices,
        faces=unit_sphere_mesh.faces,
        batched_wxyzs=wxyzs,
        batched_positions=positions,
        batched_scales=radii,
        batched_colors=sphere_color,
    )


def main():
    """Main function for collision sphere visualization."""
    parser = argparse.ArgumentParser(description="Visualize robot collision spheres with viser.")
    parser.add_argument("--urdf", required=True, help="Path to the URDF file")
    parser.add_argument("--collision-spheres", required=True, help="Path to the collision-spheres YAML")
    args = parser.parse_args()

    urdf = yourdfpy.URDF.load(args.urdf)
    robot = Robot.load(
        urdf,
        load_collision_spheres=True,
        collision_spheres_path=args.collision_spheres,
    )

    print(f"Loaded robot with {len(robot.spec.local_collision_sphere_centers)} collision spheres")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    with server.gui.add_folder("Settings"):
        show_spheres = server.gui.add_checkbox("Show Collision Spheres", initial_value=True)

    with server.gui.add_folder("Joint Control"):
        joint_sliders = create_joint_sliders(server, urdf_vis)
        reset_button = server.gui.add_button("Reset Joints")

    unit_sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere_handle: Optional[viser.BatchedMeshHandle] = None
    sphere_color: Tuple[float, float, float] = (0.0, 0.8, 0.2)

    initial_values = [slider.value for slider in joint_sliders]
    initial_q = np.array(initial_values, dtype=np.float32)

    @reset_button.on_click
    def _(_):
        for slider, init_val in zip(joint_sliders, initial_values):
            slider.value = init_val

    urdf_vis.update_cfg(initial_q)
    sphere_handle = update_collision_spheres(server, robot, initial_q, unit_sphere_mesh, sphere_handle, sphere_color)

    while True:
        q = np.array([slider.value for slider in joint_sliders], dtype=np.float32)

        urdf_vis.update_cfg(q)

        if show_spheres.value:
            sphere_handle = update_collision_spheres(server, robot, q, unit_sphere_mesh, sphere_handle, sphere_color)
        else:
            if sphere_handle is not None:
                sphere_handle.remove()
                sphere_handle = None

        time.sleep(0.03)


if __name__ == "__main__":
    main()
