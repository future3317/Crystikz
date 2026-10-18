"""Optional true-3D renderer backed by PyVista/VTK."""

from __future__ import annotations

import base64
import math
import os
import tempfile
import warnings
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
    LegendItem,
    Line,
    Polygon,
    Polyhedron,
    Polyline,
    Sphere,
    Text,
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

        camera = self.camera
        if camera is None:
            from crystalfig.scene.camera import Camera

            camera = Camera()
            self.camera = camera
        if camera.auto_fit:
            camera.fit_to_scene(scene)
        quality = {
            "preview": {"sphere": 12, "tube": 10, "aa": "fxaa", "ssao": False, "kernel": 0, "max_pixels": 4_000_000},
            "standard": {"sphere": 24, "tube": 20, "aa": "ssaa", "ssao": True, "kernel": 64, "max_pixels": 16_000_000},
            "publication": {"sphere": 48, "tube": 32, "aa": "ssaa", "ssao": True, "kernel": 128, "max_pixels": 36_000_000},
        }.get(theme.render_quality)
        if quality is None:
            raise RenderError("render_quality must be preview, standard, or publication.")
        window_size = self._window_size(options, scene, camera, quality["max_pixels"])
        pv = _pyvista()
        background, transparent = self._background(theme, options)

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        try:
            plotter.set_background(background)
            plotter.enable_lightkit()
            self._add_scene(plotter, scene, theme, pv, window_size, options.dpi)
            self._set_camera(plotter, scene, window_size[0] / window_size[1])
            try:
                plotter.enable_anti_aliasing(quality["aa"])
            except (AttributeError, RuntimeError) as exc:
                warnings.warn(f"VTK anti-aliasing unavailable: {exc}", RuntimeWarning, stacklevel=2)
            if quality["ssao"]:
                try:
                    plotter.enable_ssao(radius=0.5, bias=0.005, kernel_size=quality["kernel"])
                except (AttributeError, RuntimeError) as exc:
                    warnings.warn(f"VTK SSAO unavailable: {exc}", RuntimeWarning, stacklevel=2)
            if any(isinstance(p, Polyhedron) and p.opacity < 1 for p in scene.all_primitives()):
                try:
                    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
                except (AttributeError, RuntimeError) as exc:
                    warnings.warn(f"VTK depth peeling unavailable: {exc}", RuntimeWarning, stacklevel=2)
                if not plotter.renderer.GetUseDepthPeeling():
                    warnings.warn(
                        "The active VTK render window does not support depth peeling; transparent polyhedra may have approximate occlusion.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            plotter.screenshot(os.fspath(path), transparent_background=transparent, window_size=window_size)
            with Image.open(path) as image:
                image.save(path, dpi=(options.dpi, options.dpi))
        finally:
            plotter.close()

    @staticmethod
    def _window_size(
        options: RenderOptions,
        scene: Scene | None = None,
        camera=None,
        quality_pixel_limit: int | None = None,
    ) -> tuple[int, int]:
        width = options.width
        if options.height is not None:
            height = options.height
        else:
            frame = camera.scene_frame(scene) if scene is not None and camera is not None else None
            height = width / frame.aspect if frame is not None else width
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise RenderError("PNG width and height must be finite positive millimetres.")
        if not isinstance(options.dpi, int) or options.dpi <= 0:
            raise RenderError("PNG dpi must be a positive integer.")
        size = (round(width * options.dpi / 25.4), round(height * options.dpi / 25.4))
        if min(size) < 1:
            raise RenderError("PNG dimensions must produce at least one pixel in each direction.")
        max_pixels = min(options.max_pixels, quality_pixel_limit or options.max_pixels)
        if size[0] * size[1] > max_pixels:
            raise RenderError(
                f"PNG exceeds the pixel limit: {size[0] * size[1]:,} pixels requested, "
                f"configured limit is {max_pixels:,}."
            )
        return size

    @staticmethod
    def _background(theme: FigureTheme, options: RenderOptions) -> tuple[str, bool]:
        background = options.background or theme.background or "white"
        transparent = options.transparent or background == "transparent"
        return ("white" if transparent else background), transparent

    def _add_scene(self, plotter, scene: Scene, theme: FigureTheme, pv, window_size, dpi) -> None:
        quality = {
            "preview": {"sphere": 12, "tube": 10},
            "standard": {"sphere": 24, "tube": 20},
            "publication": {"sphere": 48, "tube": 32},
        }.get(theme.render_quality)
        if quality is None:
            raise RenderError("render_quality must be preview, standard, or publication.")
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

        source_sphere = pv.Sphere(
            radius=1.0, theta_resolution=quality["sphere"], phi_resolution=quality["sphere"]
        )
        layer_groups = {}
        legend_items = []
        for primitive in primitives:
            if isinstance(primitive, LegendItem):
                legend_items.append(primitive)
                continue
            layer_groups.setdefault(layer_priority(primitive), []).append(primitive)
        for layer in sorted(layer_groups):
            layer_primitives = layer_groups[layer]
            sphere_batches = defaultdict(list)
            bond_batches = defaultdict(list)
            for primitive in layer_primitives:
                if isinstance(primitive, Sphere):
                    sphere_batches[(primitive.color, primitive.opacity, primitive.render_style)].append(primitive)
                elif isinstance(primitive, Bond):
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
                        bond_batches[(color, primitive.radius, primitive.opacity)].append(
                            (start + lower * vector, start + upper * vector)
                        )
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

            for (color, radius, opacity), segments in bond_batches.items():
                points = np.asarray([point for segment in segments for point in segment])
                line_cells = np.asarray(
                    [value for index in range(len(segments)) for value in (2, 2 * index, 2 * index + 1)],
                    dtype=np.int64,
                )
                lines = pv.PolyData(points)
                lines.lines = line_cells
                tubes = lines.tube(radius=radius, n_sides=quality["tube"], capping=True)
                plotter.add_mesh(tubes, color=color, opacity=opacity, smooth_shading=True)

            for primitive in layer_primitives:
                if isinstance(primitive, (Sphere, Bond)):
                    continue
                if isinstance(primitive, Cylinder):
                    self._add_cylinder(
                        plotter, pv, primitive.start, primitive.end, primitive.radius,
                        primitive.color, primitive.opacity,
                        quality["tube"],
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
                    face_roles = primitive.face_metadata
                    if face_roles and any(face.get("kind") == "cut_face" for face in face_roles):
                        mesh.cell_data["face_role"] = np.asarray(
                            [1 if face.get("kind") == "cut_face" else 0 for face in face_roles],
                            dtype=np.uint8,
                        )
                        plotter.add_mesh(
                            mesh,
                            scalars="face_role",
                            cmap=[primitive.fill_color or primitive.color, "#cbd5df"],
                            clim=(0, 1),
                            preference="cell",
                            opacity=primitive.opacity,
                            smooth_shading=False,
                            show_edges=False,
                            show_scalar_bar=False,
                        )
                    else:
                        camera_vertices = np.asarray(primitive.vertices) @ self.camera.rotation_matrix().T
                        front_faces = []
                        back_faces = []
                        for face in primitive.faces:
                            points = camera_vertices[np.asarray(face, dtype=int)]
                            normal = np.zeros(3)
                            for index in range(len(points)):
                                normal += np.cross(points[index], points[(index + 1) % len(points)])
                            (front_faces if normal[2] >= 0 else back_faces).append(face)
                        for faces, opacity in (
                            (back_faces, primitive.opacity * theme.polyhedron_back_opacity),
                            (front_faces, primitive.opacity * theme.polyhedron_face_opacity),
                        ):
                            if not faces:
                                continue
                            packed = np.asarray(
                                [item for face in faces for item in (len(face), *face)],
                                dtype=np.int64,
                            )
                            face_mesh = pv.PolyData(np.asarray(primitive.vertices), packed)
                            plotter.add_mesh(
                                face_mesh,
                                color=primitive.fill_color or primitive.color,
                                opacity=opacity,
                                smooth_shading=False,
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
                    self._add_arrow(plotter, pv, primitive, quality["tube"])
                    if isinstance(primitive, Axis) and primitive.name:
                        self._add_world_text(
                            plotter,
                            primitive.start + primitive.direction * 1.04,
                            primitive.name,
                            primitive.color,
                            theme.label_size,
                        )
                elif isinstance(primitive, Text):
                    if primitive.coordinate_space == "screen":
                        x, y = primitive.position[:2]
                        plotter.add_text(
                            primitive.text,
                            position=(round(x * window_size[0]), round((1 - y) * window_size[1])),
                            font_size=max(1, round(primitive.fontsize * dpi / 72)),
                            color=primitive.color,
                        )
                    else:
                        self._add_world_text(
                            plotter, primitive.position, primitive.text, primitive.color, primitive.fontsize
                        )
                else:
                    raise RenderError(
                        f"PyVistaRenderer does not support scene primitive {type(primitive).__name__}."
                    )
        if legend_items:
            plotter.add_legend(
                labels=[(item.text, item.color) for item in legend_items],
                bcolor=None,
                border=False,
                loc="upper right",
            )

    @staticmethod
    def _add_world_text(plotter, position, text, color, fontsize) -> None:
        plotter.add_point_labels(
            np.asarray([position], dtype=float),
            [text],
            font_size=max(1, round(fontsize)),
            text_color=color,
            shape=None,
            show_points=False,
            always_visible=True,
            margin=2,
        )

    @staticmethod
    def _add_cylinder(plotter, pv, start, end, radius, color, opacity, resolution=20) -> None:
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            return
        mesh = pv.Cylinder(
            center=(np.asarray(start) + np.asarray(end)) / 2,
            direction=direction / length,
            radius=radius,
            height=length,
            resolution=resolution,
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
    def _add_arrow(plotter, pv, primitive, resolution=20) -> None:
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
                resolution=resolution,
            )
            plotter.add_mesh(shaft, color=primitive.color, opacity=primitive.opacity, smooth_shading=True)
        if head_length > 1e-12 and primitive.head_radius > 0:
            tip = pv.Cone(
                center=start + unit * (length - head_length / 2),
                direction=unit,
                height=head_length,
                radius=primitive.head_radius,
                resolution=resolution,
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
        if camera.auto_fit:
            camera.fit_to_scene(scene)
        rotation = camera.rotation_matrix()
        target = np.asarray(camera.target, dtype=float)
        position = target + rotation[2] * camera.distance
        view_up = rotation[1]
        plotter.camera_position = [position.tolist(), target.tolist(), view_up.tolist()]
        plotter.camera.parallel_projection = camera.projection == "orthographic"
        frame = camera.scene_frame(scene)
        sphere = camera.scene_bounding_sphere(scene)
        if frame is not None and camera.auto_fit:
            plotter.camera.parallel_scale = max(
                frame.height / 2,
                frame.width / (2 * aspect),
            ) / max(1.0 - camera.padding, 1e-6)
        if sphere is not None and camera.auto_fit:
            _, radius = sphere
            if camera.projection == "perspective":
                distance = radius / max(math.tan(math.radians(camera.fov / 2)), 1e-6) * 1.1 + radius
            else:
                distance = 2.5 * radius + 1.0
            distance = max(camera.distance, distance)
        else:
            distance = camera.distance
        if camera.auto_fit:
            position = target + rotation[2] * distance
            plotter.camera_position = [position.tolist(), target.tolist(), view_up.tolist()]
        if camera.projection == "perspective":
            plotter.camera.view_angle = camera.fov
        if sphere is not None:
            clipping_radius = max(sphere[1] * 1.5, 1.0)
            plotter.camera.clipping_range = (
                max(distance - clipping_radius, 0.01),
                distance + clipping_radius,
            )
