"""Optional true-3D renderer backed by PyVista/VTK."""

from __future__ import annotations

import base64
import math
import os
import tempfile
from collections import defaultdict

import numpy as np
from PIL import Image

from crystalfig.exceptions import OptionalDependencyError, RenderError
from crystalfig.geometry.polyhedra import visible_polyhedron_edges
from crystalfig.renderers.base import RenderOptions
from crystalfig.scene.primitives import (
    Arrow,
    Axis,
    Bond,
    CellEdge,
    Cylinder,
    Line,
    Polygon,
    Polyhedron,
    Polyline,
    Sphere,
    layer_priority,
)
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise OptionalDependencyError("pyvista", "render3d") from exc
    return pv


class PyVistaRenderer:
    """Render a backend-independent Scene as a depth-buffered PNG image."""

    def __init__(self, camera=None):
        self.camera = camera

    def render(self, scene: Scene, theme: FigureTheme, options: RenderOptions) -> str:
        """Render a PNG and return it as a data URI."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image:
            path = image.name
        try:
            self.export(scene, path, theme, options)
            with open(path, "rb") as image:
                return "data:image/png;base64," + base64.b64encode(image.read()).decode("ascii")
        finally:
            os.unlink(path)

    def export(
        self,
        scene: Scene,
        path: str,
        theme: FigureTheme,
        options: RenderOptions,
        fmt: str | None = None,
    ) -> None:
        """Export the scene to a high-resolution PNG file."""
        output_format = (fmt or os.path.splitext(os.fspath(path))[1].lstrip(".")).lower()
        if output_format != "png":
            raise RenderError(f"PyVistaRenderer supports PNG export only; received {output_format!r}.")

        window_size = self._window_size(options)
        pv = _pyvista()
        background, transparent = self._background(theme, options)

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        try:
            plotter.set_background(background)
            plotter.enable_lightkit()
            self._add_scene(plotter, scene, theme, pv)
            self._set_camera(plotter, scene, window_size[0] / window_size[1])
            plotter.enable_anti_aliasing("ssaa")
            plotter.enable_ssao(radius=0.5, bias=0.005, kernel_size=128)
            if any(isinstance(p, Polyhedron) and p.opacity < 1 for p in scene.all_primitives()):
                plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
                if not plotter.renderer.GetUseDepthPeeling():
                    raise RenderError(
                        "The active VTK render window does not support depth peeling for transparent polyhedra."
                    )
            plotter.screenshot(os.fspath(path), transparent_background=transparent, window_size=window_size)
            with Image.open(path) as image:
                image.save(path, dpi=(options.dpi, options.dpi))
        finally:
            plotter.close()

    @staticmethod
    def _window_size(options: RenderOptions) -> tuple[int, int]:
        width = options.width
        height = options.height if options.height is not None else width
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise RenderError("PNG width and height must be finite positive millimetres.")
        if not isinstance(options.dpi, int) or options.dpi <= 0:
            raise RenderError("PNG dpi must be a positive integer.")
        size = (round(width * options.dpi / 25.4), round(height * options.dpi / 25.4))
        if min(size) < 1:
            raise RenderError("PNG dimensions must produce at least one pixel in each direction.")
        return size

    @staticmethod
    def _background(theme: FigureTheme, options: RenderOptions) -> tuple[str, bool]:
        background = options.background or theme.background or "white"
        transparent = options.transparent or background == "transparent"
        return ("white" if transparent else background), transparent

    def _add_scene(self, plotter, scene: Scene, theme: FigureTheme, pv) -> None:
        primitives = sorted(scene.all_primitives(), key=layer_priority)
        atom_radii = {}
        atom_colors = {}
        for primitive in primitives:
            if isinstance(primitive, Sphere):
                metadata = primitive.metadata or {}
                site = metadata.get("site_index", -1)
                image = tuple(metadata.get("image_offset", (0, 0, 0)))
                if site >= 0:
                    atom_radii[(site, image)] = primitive.radius
                    atom_colors[(site, image)] = primitive.color

        source_sphere = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24)
        layer_groups = {}
        for primitive in primitives:
            layer_groups.setdefault(layer_priority(primitive), []).append(primitive)
        for layer in sorted(layer_groups):
            layer_primitives = layer_groups[layer]
            sphere_batches = defaultdict(list)
            for primitive in layer_primitives:
                if isinstance(primitive, Sphere):
                    sphere_batches[(primitive.color, primitive.opacity, primitive.render_style)].append(primitive)
            for (color, opacity, style), spheres in sphere_batches.items():
                points = np.asarray([sphere.position for sphere in spheres])
                cloud = pv.PolyData(points)
                cloud["radius"] = np.asarray([sphere.radius for sphere in spheres])
                glyphs = cloud.glyph(geom=source_sphere, scale="radius", orient=False)
                materials = {
                    "matte": {"pbr": True, "roughness": 0.72, "metallic": 0.0},
                    "glossy": {"pbr": True, "roughness": 0.32, "metallic": 0.0},
                    "flat": {"lighting": False},
                }
                if theme.atom_material not in materials:
                    raise RenderError(
                        f"Unknown atom material {theme.atom_material!r}; use matte, glossy, or flat."
                    )
                plotter.add_mesh(
                    glyphs,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    style="wireframe" if style == "wireframe" else "surface",
                    show_edges=(style == "wireframe"),
                    ambient=0.18,
                    diffuse=0.82,
                    specular=0.16 if theme.atom_material == "glossy" else 0.08,
                    specular_power=28,
                    **materials[theme.atom_material],
                )

            for primitive in layer_primitives:
                if isinstance(primitive, Sphere):
                    continue
                if isinstance(primitive, Bond):
                    start, end = self._clip_bond(primitive, atom_radii)
                    color_i = atom_colors.get((primitive.site_i, (0, 0, 0)), primitive.color)
                    color_j = atom_colors.get((primitive.site_j, tuple(primitive.jimage)), primitive.color)
                    if theme.bond_color_mode == "uniform":
                        segments = [(0.0, 1.0, primitive.color)]
                    elif theme.bond_color_mode == "split_soft":
                        from matplotlib.colors import to_hex, to_rgb

                        middle = tuple(
                            (a + b) / 2
                            for a, b in zip(to_rgb(color_i), to_rgb(color_j), strict=True)
                        )
                        segments = [
                            (0.0, 0.45, color_i),
                            (0.45, 0.55, to_hex(middle)),
                            (0.55, 1.0, color_j),
                        ]
                    else:
                        segments = [(0.0, 0.5, color_i), (0.5, 1.0, color_j)]
                    vector = end - start
                    for lower, upper, color in segments:
                        self._add_cylinder(
                            plotter, pv, start + lower * vector, start + upper * vector,
                            primitive.radius, color, primitive.opacity,
                        )
                elif isinstance(primitive, Cylinder):
                    self._add_cylinder(
                        plotter, pv, primitive.start, primitive.end, primitive.radius,
                        primitive.color, primitive.opacity,
                    )
                elif isinstance(primitive, (CellEdge, Line)):
                    self._add_line(plotter, pv.Line(primitive.start, primitive.end), primitive)
                elif isinstance(primitive, Polyline):
                    points = primitive.points
                    if primitive.closed and len(points) > 2:
                        points = [*points, points[0]]
                    if len(points) > 1:
                        mesh = pv.PolyData(np.asarray(points))
                        mesh.lines = np.asarray([len(points), *range(len(points))], dtype=np.int64)
                        self._add_line(plotter, mesh, primitive)
                elif isinstance(primitive, Polygon) and len(primitive.points) >= 3:
                    face = np.asarray([len(primitive.points), *range(len(primitive.points))], dtype=np.int64)
                    mesh = pv.PolyData(np.asarray(primitive.points), face)
                    plotter.add_mesh(
                        mesh,
                        color=primitive.fill_color or primitive.color,
                        opacity=primitive.opacity,
                        show_edges=primitive.edge_color is not None,
                        edge_color=primitive.edge_color or primitive.color,
                        line_width=primitive.linewidth,
                    )
                elif isinstance(primitive, Polyhedron) and primitive.faces:
                    packed_faces = np.asarray(
                        [item for face in primitive.faces for item in (len(face), *face)], dtype=np.int64
                    )
                    mesh = pv.PolyData(np.asarray(primitive.vertices), packed_faces)
                    plotter.add_mesh(
                        mesh,
                        color=primitive.fill_color or primitive.color,
                        opacity=primitive.opacity,
                        smooth_shading=True,
                        show_edges=False,
                    )
                    edge_mode = primitive.edge_mode or theme.polyhedron_edge_mode
                    edges = visible_polyhedron_edges(
                        primitive.vertices, primitive.faces, self.camera.rotation_matrix(), edge_mode
                    )
                    back_mode = primitive.back_edges or theme.polyhedron_back_edges
                    for start_index, end_index, front in edges:
                        if not front and back_mode == "hide":
                            continue
                        opacity = theme.polyhedron_edge_opacity * (1.0 if front else 0.35)
                        start = np.asarray(primitive.vertices[start_index], dtype=float)
                        end = np.asarray(primitive.vertices[end_index], dtype=float)
                        if not front and back_mode == "dashed":
                            vector = end - start
                            for fraction in np.arange(0.0, 1.0, 0.18):
                                dash_end = min(fraction + 0.11, 1.0)
                                edge = pv.Line(start + fraction * vector, start + dash_end * vector)
                                self._add_styled_line(
                                    plotter, edge, primitive.edge_color or primitive.color,
                                    opacity, primitive.edge_width,
                                )
                        else:
                            self._add_styled_line(
                                plotter, pv.Line(start, end), primitive.edge_color or primitive.color,
                                opacity, primitive.edge_width,
                            )
                elif isinstance(primitive, (Arrow, Axis)):
                    self._add_arrow(plotter, pv, primitive)

    @staticmethod
    def _add_cylinder(plotter, pv, start, end, radius, color, opacity) -> None:
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            return
        mesh = pv.Cylinder(
            center=(np.asarray(start) + np.asarray(end)) / 2,
            direction=direction / length,
            radius=radius,
            height=length,
            resolution=20,
        )
        plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True)

    @staticmethod
    def _add_line(plotter, mesh, primitive) -> None:
        PyVistaRenderer._add_styled_line(
            plotter, mesh, primitive.color, primitive.opacity, primitive.linewidth
        )

    @staticmethod
    def _add_styled_line(plotter, mesh, color, opacity, linewidth) -> None:
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            line_width=max(1.0, linewidth),
            render_lines_as_tubes=True,
        )

    @staticmethod
    def _add_arrow(plotter, pv, primitive) -> None:
        start = np.asarray(primitive.start, dtype=float)
        direction = np.asarray(primitive.direction, dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            return
        if primitive.normalize:
            direction = direction / length
            length = 1.0
        unit = direction / np.linalg.norm(direction)
        head_length = min(max(primitive.head_length, 0.0), length)
        shaft_length = length - head_length
        if shaft_length > 1e-12:
            end = start + unit * shaft_length
            shaft = pv.Cylinder(
                center=(start + end) / 2,
                direction=unit,
                radius=primitive.shaft_radius,
                height=shaft_length,
                resolution=16,
            )
            plotter.add_mesh(shaft, color=primitive.color, opacity=primitive.opacity, smooth_shading=True)
        if head_length > 1e-12 and primitive.head_radius > 0:
            tip = pv.Cone(
                center=start + unit * (length - head_length / 2),
                direction=unit,
                height=head_length,
                radius=primitive.head_radius,
                resolution=20,
            )
            plotter.add_mesh(tip, color=primitive.color, opacity=primitive.opacity, smooth_shading=True)

    @staticmethod
    def _clip_bond(bond, atom_radii):
        start = np.asarray(bond.start, dtype=float).copy()
        end = np.asarray(bond.end, dtype=float).copy()
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            return start, end
        direction = vector / length
        start_radius = atom_radii.get((getattr(bond, "site_i", -1), (0, 0, 0)), 0.0)
        end_image = tuple(getattr(bond, "jimage", (0, 0, 0)))
        end_radius = atom_radii.get((getattr(bond, "site_j", -1), end_image), 0.0)
        if length > start_radius + end_radius:
            start += direction * start_radius
            end -= direction * end_radius
        return start, end

    def _set_camera(self, plotter, scene: Scene, aspect: float) -> None:
        camera = self.camera
        if camera is None:
            from crystalfig.scene.camera import Camera

            camera = Camera()
        if camera.auto_fit:
            camera.fit_to_scene(scene)
        rotation = camera.rotation_matrix()
        target = np.asarray(camera.target, dtype=float)
        position = target + rotation[2] * camera.distance
        view_up = rotation[1]
        plotter.camera_position = [position.tolist(), target.tolist(), view_up.tolist()]
        plotter.camera.parallel_projection = camera.projection == "orthographic"
        plotter.camera.parallel_scale = (1.0 / max(camera.scale, 1e-12)) / min(1.0, aspect)
        if camera.projection == "perspective":
            plotter.camera.view_angle = camera.fov
