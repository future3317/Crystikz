"""Matplotlib-based 2D vector/raster renderer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless/export use
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

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
    Text,
)
from crystalfig.scene.primitives import (
    Polygon as Poly,
)
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


class MatplotlibRenderer:
    """Render a Scene using Matplotlib.

    This renderer projects all 3D primitives to 2D using an orthographic camera,
    sorts by depth, and draws using Matplotlib patches.  Output is true vector
    for PDF/SVG/EPS and raster for PNG/TIFF.
    """

    def __init__(self, camera):
        self.camera = camera

    def render(self, scene: Scene, theme: FigureTheme, options: RenderOptions) -> str:
        """Render to SVG string."""
        import io
        self.export(scene, io.BytesIO(), theme, options, fmt="svg")
        return io.BytesIO().getvalue().decode("utf-8")

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
        height_inch = (options.height / 25.4) if options.height else width_inch

        fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=options.dpi)
        ax.set_aspect("equal")
        ax.axis("off")

        bg = options.background or theme.background
        if bg and bg != "transparent":
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
        elif options.transparent or bg == "transparent":
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

        # Fit camera to scene bounding box
        bbox = scene.bounding_box()
        if bbox is not None and self.camera.auto_fit:
            self.camera.fit_to_bounding_box(bbox)

        primitives = self._sort_by_depth(scene)
        for p in primitives:
            self._draw_primitive(ax, p, theme)

        if options.title:
            ax.set_title(options.title, fontsize=theme.title_size)

        if options.show_legend and theme.show_legend:
            self._draw_legend(ax, scene, theme)

        # Tight layout and equal axis
        ax.autoscale_view(tight=True)
        margin = 0.05
        if bbox is not None:
            uv = self.camera.project(bbox)
            xmin, ymin = uv.min(axis=0)
            xmax, ymax = uv.max(axis=0)
            dx = xmax - xmin
            dy = ymax - ymin
            ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
            ax.set_ylim(ymin - margin * dy, ymax + margin * dy)

        if isinstance(path, str):
            if fmt in ("png", "tiff", "tif"):
                transparent = options.transparent or bg == "transparent"
                fig.savefig(path, dpi=options.dpi, transparent=transparent, bbox_inches="tight")
            elif fmt in ("pdf", "svg", "eps", "pgf"):
                matplotlib.rcParams["pdf.fonttype"] = 42
                matplotlib.rcParams["ps.fonttype"] = 42
                fig.savefig(path, format=fmt, bbox_inches="tight")
            else:
                fig.savefig(path, bbox_inches="tight")
        else:
            fig.savefig(path, format=fmt, bbox_inches="tight")
        plt.close(fig)

    def _guess_format(self, path) -> str:
        if isinstance(path, str):
            return path.split(".")[-1].lower()
        return "png"

    def _sort_by_depth(self, scene: Scene) -> list:
        """Sort primitives by average camera depth (painter's algorithm)."""
        scored = []
        for p in scene.all_primitives():
            pts = self._primitive_points(p)
            if len(pts) == 0:
                depth = 1e9
            else:
                depth = float(np.mean(self.camera.depth(np.array(pts))))
            scored.append((depth, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored]

    def _primitive_points(self, p) -> list[np.ndarray]:
        if isinstance(p, Sphere):
            return [p.position]
        if isinstance(p, (Line, Bond, CellEdge, Cylinder)):
            return [p.start, p.end]
        if isinstance(p, Polyhedron):
            return p.vertices
        if isinstance(p, (Poly, Plane)):
            return p.points
        if isinstance(p, (Arrow, Axis)):
            return [p.start, p.start + p.direction]
        if isinstance(p, Text):
            return [p.position]
        return []

    def _draw_primitive(self, ax, p, theme: FigureTheme):
        if isinstance(p, Sphere):
            uv = np.asarray(self.camera.project(p.position)).flatten()[:2]
            color = self._to_rgba(p.color, p.opacity)
            if p.render_style == "wireframe":
                circle = Circle(uv, p.radius, fill=False, edgecolor=color, linewidth=1.5, linestyle="--")
            else:
                circle = Circle(uv, p.radius, color=color, ec="black", linewidth=0.3)
            ax.add_patch(circle)
        elif isinstance(p, (Line, Bond, CellEdge, Cylinder)):
            uv1 = np.asarray(self.camera.project(p.start)).flatten()[:2]
            uv2 = np.asarray(self.camera.project(p.end)).flatten()[:2]
            lw = getattr(p, "linewidth", getattr(p, "radius", 1.0) * 10)
            alpha = p.opacity
            style = "--" if getattr(p, "dashed", False) or getattr(p, "is_back", False) else "-"
            color = self._to_rgba(p.color, alpha)
            ax.plot([uv1[0], uv2[0]], [uv1[1], uv2[1]], color=color, linewidth=lw, linestyle=style)
        elif isinstance(p, Polyhedron):
            for face in p.faces:
                pts = np.array([p.vertices[i] for i in face])
                uv = self.camera.project(pts)
                fill = self._to_rgba(p.fill_color or p.color, p.opacity)
                edge = self._to_rgba(p.edge_color or p.color, 0.6)
                poly = Polygon(uv, closed=True, facecolor=fill, edgecolor=edge, linewidth=p.edge_width)
                ax.add_patch(poly)
        elif isinstance(p, Poly):
            if len(p.points) < 3:
                return
            uv = self.camera.project(np.array(p.points))
            fill = self._to_rgba(p.fill_color or p.color, p.opacity)
            edge = self._to_rgba(p.edge_color or p.color, 0.8)
            poly = Polygon(uv, closed=True, facecolor=fill, edgecolor=edge, linewidth=p.linewidth)
            ax.add_patch(poly)
        elif isinstance(p, (Arrow, Axis)):
            uv1 = np.asarray(self.camera.project(p.start)).flatten()[:2]
            uv2 = np.asarray(self.camera.project(p.start + p.direction)).flatten()[:2]
            color = self._to_rgba(p.color, p.opacity)
            arrow = FancyArrowPatch(
                uv1, uv2,
                arrowstyle="-|>",
                mutation_scale=10,
                color=color,
                linewidth=2,
            )
            ax.add_patch(arrow)
        elif isinstance(p, Text):
            uv = np.asarray(self.camera.project(p.position)).flatten()[:2]
            ax.text(uv[0], uv[1], p.text, fontsize=p.fontsize, color=p.color, ha=p.halign, va=p.valign)

    def _draw_legend(self, ax, scene: Scene, theme: FigureTheme):
        legend_items = [p for p in scene.all_primitives() if isinstance(p, LegendItem)]
        if not legend_items:
            return
        handles = []
        labels = []
        for item in legend_items[:8]:
            handles.append(Circle((0, 0), 0.1, color=self._to_rgba(item.color, 1.0)))
            labels.append(item.text)
        ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=theme.label_size)

    def _to_rgba(self, color, alpha: float = 1.0) -> tuple[float, float, float, float]:
        from matplotlib.colors import to_rgba
        if isinstance(color, str) and color.startswith("#"):
            return to_rgba(color, alpha)
        if isinstance(color, (tuple, list)) and len(color) in (3, 4):
            return tuple(float(c) for c in color) + (alpha,) if len(color) == 3 else tuple(float(c) for c in color)
        return to_rgba(color, alpha)
