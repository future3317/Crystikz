"""True 3D Matplotlib renderer using mpl_toolkits.mplot3d.

This renderer draws the scene in a genuine 3D axes, preserving depth through
Matplotlib's 3D projection.  It is useful for quick interactive previews and
for figures where a perspective/3D look is preferred over the flat 2D vector
projection used by MatplotlibRenderer.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from crystalfig.renderers.base import RenderOptions
from crystalfig.scene.primitives import (
    Arrow,
    Axis,
    Bond,
    CellEdge,
    Cylinder,
    LegendItem,
    Line,
    Plane,
    Polyhedron,
    Sphere,
)
from crystalfig.scene.primitives import (
    Polygon as Poly,
)
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


class Matplotlib3DRenderer:
    """Render a Scene using Matplotlib's 3D projection.

    Output is raster for PNG/TIFF and vector-ish for PDF/SVG (Matplotlib 3D
    produces a fixed projection, so the result is not a pure vector scene).
    """

    def __init__(self, camera=None):
        # Camera is optional; 3D view is controlled via view_init.
        self.camera = camera

    def render(self, scene: Scene, theme: FigureTheme, options: RenderOptions) -> str:
        """Render to SVG string."""
        import io

        buf = io.BytesIO()
        self.export(scene, buf, theme, options, fmt="svg")
        return buf.getvalue().decode("utf-8")

    def export(
        self,
        scene: Scene,
        path,
        theme: FigureTheme,
        options: RenderOptions,
        fmt: str | None = None,
    ) -> None:
        """Export scene to a file or file-like object."""
        if fmt is None:
            fmt = self._guess_format(path)

        width_inch = options.width / 25.4
        height_inch = options.height / 25.4 if options.height else width_inch

        fig = plt.figure(figsize=(width_inch, height_inch), dpi=options.dpi)
        ax = fig.add_subplot(111, projection="3d")

        bg = options.background or theme.background
        if bg and bg != "transparent":
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
        elif options.transparent or bg == "transparent":
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

        ax.set_axis_off()
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_alpha(0)
            axis.pane.set_edgecolor("none")

        # Default view; can be overridden by camera elevation/azimuth when available.
        # Camera stores elevation/azimuth in degrees.
        elev, azim = 25.0, -60.0
        if self.camera is not None:
            elev = float(self.camera.elevation) if hasattr(self.camera, "elevation") else elev
            azim = float(self.camera.azimuth) if hasattr(self.camera, "azimuth") else azim
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])

        # Compute extent first so atom marker sizes can be scaled to the scene.
        self._compute_scene_metrics(scene, width_inch, options.dpi)
        self._draw_scene(ax, scene, theme)
        self._set_bounds(ax, scene)

        if isinstance(path, str):
            if fmt in ("png", "tiff", "tif"):
                transparent = options.transparent or bg == "transparent"
                fig.savefig(path, dpi=options.dpi, transparent=transparent)
            elif fmt in ("pdf", "svg", "eps", "pgf"):
                matplotlib.rcParams["pdf.fonttype"] = 42
                matplotlib.rcParams["ps.fonttype"] = 42
                fig.savefig(path, format=fmt)
            else:
                fig.savefig(path)
        else:
            fig.savefig(path, format=fmt)
        plt.close(fig)

    def _guess_format(self, path) -> str:
        if isinstance(path, str):
            return path.split(".")[-1].lower()
        return "png"

    def _to_rgba(self, color, alpha: float = 1.0):
        from matplotlib.colors import to_rgba

        return to_rgba(color, alpha)

    def _compute_scene_metrics(self, scene: Scene, width_inch: float, dpi: int):
        """Compute data extent and a scale factor for marker sizes."""
        pts = []
        for p in scene.all_primitives():
            if isinstance(p, Sphere):
                pts.append(p.position)
            elif isinstance(p, (Line, Bond, CellEdge, Cylinder, Arrow, Axis)):
                pts.append(p.start)
                pts.append(p.end if hasattr(p, "end") else p.start + p.direction)
            elif isinstance(p, Polyhedron):
                pts.extend(p.vertices)
            elif isinstance(p, (Poly, Plane)):
                pts.extend(p.points)
        pts = np.array(pts)
        if pts.size == 0:
            self._data_extent = 1.0
            self._marker_scale = 1.0
            return
        self._data_center = np.mean(pts, axis=0)
        extent = float(np.ptp(pts, axis=0).max())
        if extent <= 0:
            extent = 1.0
        self._data_extent = extent
        # Target: atom diameter as a fraction of the figure width.
        # Map radius/extent to a point diameter, then to scatter area.
        figure_width_pts = width_inch * 72.0
        self._marker_scale = figure_width_pts / extent

    def _draw_scene(self, ax, scene: Scene, theme: FigureTheme):
        # Ensure atom map is available for bond clipping.
        self._build_atom_map(scene)

        # Collect unique species for legend.
        legend_items = [p for p in scene.all_primitives() if isinstance(p, LegendItem)]

        # Draw polyhedra as wireframe cages so they do not occlude the cell.
        for p in scene.all_primitives():
            if isinstance(p, Polyhedron):
                self._draw_polyhedron(ax, p)

        # Draw bonds, lines, cell edges.
        for p in scene.all_primitives():
            if isinstance(p, (Bond, Cylinder)):
                self._draw_bond(ax, p)
            elif isinstance(p, (Line, CellEdge)):
                self._draw_line(ax, p)

        # Draw spheres (atoms). Matplotlib 3D handles its own depth sorting.
        spheres = [p for p in scene.all_primitives() if isinstance(p, Sphere)]
        for s in spheres:
            self._draw_sphere(ax, s)

        # Draw arrows/axes.
        for p in scene.all_primitives():
            if isinstance(p, (Arrow, Axis)):
                self._draw_arrow(ax, p)

        # Draw polygons / planes.
        for p in scene.all_primitives():
            if isinstance(p, Poly):
                self._draw_polygon(ax, p)
            elif isinstance(p, Plane):
                self._draw_plane(ax, p)

        # Text in 3D is tricky; skip unless explicitly requested later.
        if theme.show_legend and legend_items:
            self._draw_legend(ax, legend_items)

    def _draw_sphere(self, ax, p: Sphere):
        color = self._to_rgba(p.color, p.opacity)
        edge = "black" if p.render_style != "wireframe" else color
        # Diameter in points = 2 * radius * scale; convert to area for scatter.
        diameter_pts = 2.0 * p.radius * self._marker_scale
        size_pts = np.pi * (diameter_pts / 2.0) ** 2
        ax.scatter(
            *p.position,
            c=[color],
            s=size_pts,
            edgecolors=edge,
            linewidths=0.5,
            depthshade=False,
            alpha=p.opacity,
        )

    def _draw_bond(self, ax, p: Bond):
        start = np.asarray(p.start, dtype=float)
        end = np.asarray(p.end, dtype=float)
        # Clip bond ends so they stop at atom surfaces when metadata is available.
        if hasattr(p, "site_i") and hasattr(p, "site_j"):
            start, end = self._clip_bond_endpoints(p, start, end)
        color = self._to_rgba(p.color, p.opacity)
        lw = max(1.0, getattr(p, "radius", 0.05) * 40)
        style = "--" if getattr(p, "dashed", False) else "-"
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=lw,
            linestyle=style,
            alpha=p.opacity,
        )

    def _clip_bond_endpoints(self, p: Bond, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shorten a 3D bond so it stops at the surface of endpoint atoms."""
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length <= 1e-8:
            return start, end
        direction = vec / length

        # Build a simple lookup for canonical atoms in the scene.
        atom_map = self._atom_map
        key_i = (getattr(p, "site_i", -1), (0, 0, 0))
        if key_i in atom_map:
            r_i = atom_map[key_i]
            if length > r_i:
                start = start + direction * r_i
        key_j = (getattr(p, "site_j", -1), tuple(getattr(p, "jimage", (0, 0, 0))))
        if key_j in atom_map:
            r_j = atom_map[key_j]
            if length > r_j:
                end = end - direction * r_j
        return start, end

    @property
    def _atom_map(self) -> dict:
        if not hasattr(self, "_atom_map_cache"):
            self._atom_map_cache = {}
        return self._atom_map_cache

    def _build_atom_map(self, scene: Scene):
        self._atom_map_cache = {}
        for p in scene.all_primitives():
            if isinstance(p, Sphere):
                meta = p.metadata or {}
                key = (meta.get("site_index", -1), tuple(meta.get("image_offset", (0, 0, 0))))
                if key[0] >= 0:
                    self._atom_map_cache[key] = p.radius

    def _draw_line(self, ax, p: Line):
        color = self._to_rgba(p.color, p.opacity)
        lw = getattr(p, "linewidth", 1.0)
        style = "--" if getattr(p, "dashed", False) or getattr(p, "is_back", False) else "-"
        ax.plot(
            [p.start[0], p.end[0]],
            [p.start[1], p.end[1]],
            [p.start[2], p.end[2]],
            color=color,
            linewidth=lw,
            linestyle=style,
            alpha=p.opacity,
        )

    def _draw_polyhedron(self, ax, p: Polyhedron):
        if not p.faces:
            return
        # In 3D filled polyhedra with periodic-image vertices tend to dominate
        # the view, so render the cage as a wireframe with faint translucent fill.
        edge_rgba = self._to_rgba(p.edge_color or p.color, 0.55)
        fill_rgba = self._to_rgba(p.fill_color or p.color, p.opacity * 0.35)
        verts = [np.array([p.vertices[i] for i in face]) for face in p.faces]
        poly3d = Poly3DCollection(
            verts,
            facecolors=fill_rgba,
            edgecolors=edge_rgba,
            linewidths=0.6,
            alpha=p.opacity * 0.35,
        )
        ax.add_collection3d(poly3d)

    def _draw_polygon(self, ax, p: Poly):
        if len(p.points) < 3:
            return
        verts = [np.array(p.points)]
        fill = self._to_rgba(p.fill_color or p.color, p.opacity)
        edge = self._to_rgba(p.edge_color or p.color, 0.8)
        poly3d = Poly3DCollection(
            verts,
            facecolors=fill,
            edgecolors=edge,
            linewidths=p.linewidth,
            alpha=p.opacity,
        )
        ax.add_collection3d(poly3d)

    def _draw_plane(self, ax, p: Plane):
        # Draw a simple rectangular plane patch oriented by the normal.
        normal = np.asarray(p.normal, dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        # Two arbitrary orthogonal directions in the plane.
        u = np.cross(normal, [0, 0, 1]) if abs(normal[2]) < 0.9 else np.cross(normal, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        w, h = p.width, p.height
        corners = np.array([
            p.origin + (-w / 2) * u + (-h / 2) * v,
            p.origin + (w / 2) * u + (-h / 2) * v,
            p.origin + (w / 2) * u + (h / 2) * v,
            p.origin + (-w / 2) * u + (h / 2) * v,
        ])
        fill = self._to_rgba(p.color, p.opacity)
        poly3d = Poly3DCollection(
            [corners],
            facecolors=fill,
            edgecolors=fill,
            linewidths=0.5,
            alpha=p.opacity,
        )
        ax.add_collection3d(poly3d)

    def _draw_arrow(self, ax, p: Arrow):
        start = np.asarray(p.start, dtype=float)
        end = start + np.asarray(p.direction, dtype=float)
        color = self._to_rgba(p.color, p.opacity)
        ax.quiver(
            start[0], start[1], start[2],
            end[0] - start[0], end[1] - start[1], end[2] - start[2],
            color=color,
            arrow_length_ratio=0.3,
            linewidth=2.0,
            alpha=p.opacity,
        )

    def _draw_legend(self, ax, items: list[LegendItem]):
        handles = []
        labels = []
        for item in items[:8]:
            from matplotlib.lines import Line2D

            handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self._to_rgba(item.color, 1.0),
                    markersize=8,
                )
            )
            labels.append(item.text)
        ax.legend(handles, labels, loc="upper right", frameon=False)

    def _set_bounds(self, ax, scene: Scene):
        if hasattr(self, "_data_extent") and self._data_extent > 0:
            center = getattr(self, "_data_center", np.zeros(3))
            r = self._data_extent / 2.0 * 1.15
        else:
            pts = []
            for p in scene.all_primitives():
                if isinstance(p, Sphere):
                    pts.append(p.position)
                elif isinstance(p, (Line, Bond, CellEdge, Cylinder, Arrow, Axis)):
                    pts.append(p.start)
                    pts.append(p.end if hasattr(p, "end") else p.start + p.direction)
                elif isinstance(p, Polyhedron):
                    pts.extend(p.vertices)
                elif isinstance(p, (Poly, Plane)):
                    pts.extend(p.points)
            if not pts:
                return
            pts = np.array(pts)
            center = np.mean(pts, axis=0)
            r = np.ptp(pts, axis=0).max() / 2.0 * 1.15
            if r <= 0:
                r = 1.0
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)


