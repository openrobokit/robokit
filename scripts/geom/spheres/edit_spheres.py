"""Interactively create/edit robokit collision spheres with viser: load a URDF, optionally seeded from a YAML.

Usage:
    uv run scripts/geom/spheres/edit_spheres.py --urdf <path> --output out.yaml
    uv run scripts/geom/spheres/edit_spheres.py --urdf <path> --collision-spheres in.yaml --output out.yaml
"""

import argparse
import time
from typing import Dict, List, Optional

import numpy as np
import viser
import yaml
import yourdfpy

from robokit.robo import Robot


def main():
    parser = argparse.ArgumentParser(description="Interactively create/edit robokit collision spheres.")
    parser.add_argument("--urdf", required=True, help="Path to the URDF file")
    parser.add_argument("--collision-spheres", default=None, help="Existing collision-spheres YAML to start from")
    parser.add_argument("-o", "--output", required=True, help="YAML path the Save button writes to")
    args = parser.parse_args()

    urdf = yourdfpy.URDF.load(args.urdf)
    robot = Robot.load(urdf)
    link_names = robot.spec.link_names

    spheres: Dict[str, List[dict]] = {name: [] for name in link_names}
    if args.collision_spheres is not None:
        with open(args.collision_spheres) as f:
            loaded = yaml.safe_load(f).get("collision_spheres", {}) or {}
        for name, entries in loaded.items():
            if name in spheres:
                spheres[name] = [
                    {"center": [float(c) for c in e["center"]], "radius": float(e["radius"])} for e in entries
                ]

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    link_frames = {name: server.scene.add_frame(f"/link_frames/{name}", show_axes=False) for name in link_names}

    # Meshes are parented under our own per-link frames, so FK in `update_pose` moves both (unlike `ViserUrdf`).
    link_meshes: Dict[str, List[viser.MeshHandle]] = {name: [] for name in link_names}
    if urdf.scene is not None:
        for mesh_name, mesh in urdf.scene.geometry.items():
            parent = urdf.scene.graph.transforms.parents[mesh_name]
            mesh = mesh.copy()
            mesh.apply_transform(urdf.get_transform(mesh_name, parent))
            link_meshes[parent].append(
                server.scene.add_mesh_simple(
                    f"{link_frames[parent].name}/mesh_{len(link_meshes[parent])}",
                    mesh.vertices,
                    mesh.faces,
                    color=(180, 180, 180),
                )
            )

    sphere_nodes: Dict[str, List[viser.IcosphereHandle]] = {name: [] for name in link_names}
    sel = {"link": next(n for n in link_names if urdf.link_map[n].collisions), "idx": None}
    gizmo: Dict[str, Optional[viser.TransformControlsHandle]] = {"handle": None}

    def sphere_opacity(link: str, idx: int) -> float:
        return 1.0 if (link, idx) == (sel["link"], sel["idx"]) else 0.3

    def link_visible(link: str) -> bool:
        return not only_selected_cb.value or link == sel["link"]

    def update_selection_visuals():
        for link, nodes in sphere_nodes.items():
            visible = link_visible(link)
            for idx, node in enumerate(nodes):
                node.opacity = sphere_opacity(link, idx)
                node.visible = visible
        for link, meshes in link_meshes.items():
            visible = link_visible(link)
            for mesh in meshes:
                mesh.visible = visible

    def pick(link: str, idx: Optional[int]):
        """Select from the 3D view: clicking a sphere or a link mesh is the same as using the dropdowns."""
        previous, sel["link"] = sel["link"], link
        if previous != link:
            highlight(previous, link)
        link_dropdown.value = link
        refresh_sphere_dropdown()
        select_sphere(idx)

    for name, meshes in link_meshes.items():
        for mesh_handle in meshes:
            mesh_handle.on_click(lambda _, name=name: pick(name, None))

    def redraw_link_spheres(link: str):
        for node in sphere_nodes[link]:
            node.remove()
        sphere_nodes[link] = [
            server.scene.add_icosphere(
                f"{link_frames[link].name}/sphere_{idx}",
                radius=sph["radius"],
                position=np.array(sph["center"], dtype=np.float32),
                # The dropdowns are the only way to select, so colour is what tells you where you are.
                color=(250, 176, 5) if link == sel["link"] else (28, 126, 214),
                opacity=sphere_opacity(link, idx),
                visible=link_visible(link),
            )
            for idx, sph in enumerate(spheres[link])
        ]
        for idx, node in enumerate(sphere_nodes[link]):
            node.on_click(lambda _, link=link, idx=idx: pick(link, idx))

    def highlight(previous: str, current: str):
        """Recolour the spheres of the link you just left and the one you just picked."""
        for link in {previous, current}:
            if link in sphere_nodes:
                redraw_link_spheres(link)

    def update_gizmo():
        if gizmo["handle"] is not None:
            gizmo["handle"].remove()
            gizmo["handle"] = None
        if sel["idx"] is None:
            return
        link, idx = sel["link"], sel["idx"]
        handle = server.scene.add_transform_controls(
            f"{link_frames[link].name}/gizmo",
            scale=0.1,
            disable_rotations=True,
            position=np.array(spheres[link][idx]["center"], dtype=np.float32),
        )

        @handle.on_update
        def _(_, link=link, idx=idx):
            spheres[link][idx]["center"] = [float(c) for c in handle.position]
            sphere_nodes[link][idx].position = handle.position

        gizmo["handle"] = handle

    with server.gui.add_folder("Joints"):
        joint_sliders = []
        for joint_name, joint in zip(urdf.actuated_joint_names, urdf.actuated_joints):
            lower = -np.pi if joint.limit is None else joint.limit.lower
            upper = np.pi if joint.limit is None else joint.limit.upper
            initial = min(max(0.0, lower), upper)
            joint_sliders.append(
                server.gui.add_slider(joint_name, min=lower, max=upper, step=1e-3, initial_value=initial)
            )

        def update_pose():
            q = np.array([s.value for s in joint_sliders], dtype=np.float32)
            state = robot.forward_kinematics(robot.state(q=q))
            for i, name in enumerate(link_names):
                T = state.get_T_world_link(i)
                link_frames[name].position = T.numpy()[..., :3][0]
                link_frames[name].wxyz = T.numpy()[..., 3:][0]

        for slider in joint_sliders:
            slider.on_update(lambda _: update_pose())

    with server.gui.add_folder("Sphere Editor"):
        # A link with no collision mesh has nothing to fit spheres to, so listing it only makes the menu long.
        editable = [n for n in link_names if urdf.link_map[n].collisions or spheres[n]]
        link_dropdown = server.gui.add_dropdown("Link", options=editable, initial_value=editable[0])
        sphere_dropdown = server.gui.add_dropdown("Sphere", options=["none"], initial_value="none")
        only_selected_cb = server.gui.add_checkbox("Only show selected link", initial_value=False)
        radius_slider = server.gui.add_slider("Radius", min=0.002, max=0.2, step=0.001, initial_value=0.03)
        add_button = server.gui.add_button("Add")
        delete_button = server.gui.add_button("Delete")
        save_button = server.gui.add_button(f"Save to {args.output}")
        status = server.gui.add_markdown("")

        def refresh_sphere_dropdown():
            options = [str(i) for i in range(len(spheres[sel["link"]]))] or ["none"]
            sphere_dropdown.options = options
            sphere_dropdown.value = options[0] if sel["idx"] is None else str(sel["idx"])

        def select_sphere(idx: Optional[int]):
            # Switching link fires this with the old link's index, which may be past the new link's list.
            if idx is not None and idx >= len(spheres[sel["link"]]):
                idx = None
            sel["idx"] = idx
            if idx is not None:
                radius_slider.value = spheres[sel["link"]][idx]["radius"]
            update_gizmo()
            update_selection_visuals()

        @link_dropdown.on_update
        def _(_):
            previous, sel["link"] = sel["link"], link_dropdown.value
            highlight(previous, sel["link"])
            refresh_sphere_dropdown()
            select_sphere(None if sphere_dropdown.value == "none" else int(sphere_dropdown.value))

        @sphere_dropdown.on_update
        def _(_):
            select_sphere(None if sphere_dropdown.value == "none" else int(sphere_dropdown.value))

        @only_selected_cb.on_update
        def _(_):
            update_selection_visuals()

        @add_button.on_click
        def _(_):
            spheres[sel["link"]].append({"center": [0.0, 0.0, 0.0], "radius": radius_slider.value})
            redraw_link_spheres(sel["link"])
            refresh_sphere_dropdown()
            select_sphere(len(spheres[sel["link"]]) - 1)

        @delete_button.on_click
        def _(_):
            if sel["idx"] is None:
                return
            spheres[sel["link"]].pop(sel["idx"])
            redraw_link_spheres(sel["link"])
            refresh_sphere_dropdown()
            select_sphere(None if sphere_dropdown.value == "none" else int(sphere_dropdown.value))

        @radius_slider.on_update
        def _(_):
            if sel["idx"] is not None:
                spheres[sel["link"]][sel["idx"]]["radius"] = radius_slider.value
                sphere_nodes[sel["link"]][sel["idx"]].radius = radius_slider.value

        @save_button.on_click
        def _(_):
            out = {link: entries for link, entries in spheres.items() if entries}
            with open(args.output, "w") as f:
                yaml.dump({"collision_spheres": out}, f, default_flow_style=None, sort_keys=False)
            status.content = f"Saved {sum(len(v) for v in out.values())} spheres to {args.output}"

    update_pose()
    for name in link_names:
        redraw_link_spheres(name)
    refresh_sphere_dropdown()

    while True:
        time.sleep(0.03)


if __name__ == "__main__":
    main()
