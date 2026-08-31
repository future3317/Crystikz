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

        # Fit camera to the full projected scene extent.
        if self.camera.auto_fit:
            self.camera.fit_to_scene(scene)

        # Compute final projected bounds and derive a figure size that matches
        # the scene's aspect ratio instead of forcing a square.
        bounds = self._projected_bounds(scene)
        if bounds is None:
            bounds = (-1.0, 1.0, -1.0, 1.0)
        xmin, xmax, ymin, ymax = bounds
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)

        self.theme = theme

        width_inch = options.width / 25.4
        if options.height:
            height_inch = options.height / 25.4
        else:
            aspect = dy / dx
            # Clamp extreme aspect ratios so very elongated cells do not produce
            # impractical canvas dimensions.
            aspect = max(0.35, min(aspect, 2.8))
            height_inch = width_inch * aspect

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

        if options.title:
            ax.set_title(options.title, fontsize=theme.title_size)

        if options.show_legend and theme.show_legend:
            self._draw_legend(ax, scene, theme)

        # Build atom lookup for bond clipping and color splitting.
        self._atom_map = self._build_atom_map(scene)
        self._sphere_list = [p for p in scene.all_primitives() if isinstance(p, Sphere)]
        primitives = self._sort_by_depth(scene)
        for p in primitives:
            self._draw_primitive(ax, p, theme)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

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

    def _projected_bounds(self, scene: Scene) -> tuple[float, float, float, float] | None:
        """Return (xmin, xmax, ymin, ymax) covering all projected primitives."""
        pts = []
        for p in scene.all_primitives():
            if isinstance(p, Sphere):
                uv = np.asarray(self.camera.project(p.position)).flatten()[:2]
                r = p.radius * self.camera.scale
                pts.extend([
                    uv + np.array([r, 0.0]),
                    uv - np.array([r, 0.0]),
                    uv + np.array([0.0, r]),
                    uv - np.array([0.0, r]),
                ])
            elif isinstance(p, (Line, Bond, CellEdge, Cylinder)):
                pts.append(np.asarray(self.camera.project(p.start)).flatten()[:2])
                pts.append(np.asarray(self.camera.project(p.end)).flatten()[:2])
            elif isinstance(p, Polyhedron):
                pts.extend(self.camera.project(np.array(p.vertices)))
            elif isinstance(p, (Poly, Plane)):
                pts.extend(self.camera.project(np.array(p.points)))
            elif isinstance(p, (Arrow, Axis)):
                pts.append(np.asarray(self.camera.project(p.start)).flatten()[:2])
                pts.append(np.asarray(self.camera.project(p.start + p.direction)).flatten()[:2])
            elif isinstance(p, Text):
                pts.append(np.asarray(self.camera.project(p.position)).flatten()[:2])
        if not pts:
            return None
        arr = np.array(pts)
        xmin, ymin = arr.min(axis=0)
        xmax, ymax = arr.max(axis=0)
        margin = 0.05
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        return (xmin - margin * dx, xmax + margin * dx, ymin - margin * dy, ymax + margin * dy)

    def _build_atom_map(self, scene: Scene) -> dict:
        """Map (site_index, image_offset) -> (position, radius) for bond clipping."""
        atom_map = {}
        for p in scene.all_primitives():
            if isinstance(p, Sphere):
                meta = p.metadata or {}
                key = (meta.get("site_index", -1), tuple(meta.get("image_offset", (0, 0, 0))))
                if key[0] >= 0 and key not in atom_map:
                    atom_map[key] = (p.position, p.radius)
        return atom_map

    def _clip_bond(self, bond: Bond) -> tuple[np.ndarray, np.ndarray]:
        """Shorten a bond so it stops at the surface of its endpoint atoms."""
        start = np.asarray(bond.start, dtype=float)
        end = np.asarray(bond.end, dtype=float)
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length <= 1e-8:
            return start, end
        direction = vec / length

        key_i = (bond.site_i, (0, 0, 0))
        if key_i in self._atom_map:
            _, r_i = self._atom_map[key_i]
            if length > r_i:
                start = start + direction * r_i

        key_j = (bond.site_j, tuple(bond.jimage))
        if key_j in self._atom_map:
            _, r_j = self._atom_map[key_j]
            if length > r_j:
                end = end - direction * r_j

        return start, end

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

    def _draw_sphere(self, ax, p: Sphere) -> None:
        """Draw a glossy sphere with soft radial shading (vector-friendly)."""
        uv = np.asarray(self.camera.project(p.position)).flatten()[:2]
        base_color = self._to_rgba(p.color, p.opacity)
        r = p.radius * self.camera.scale

        if p.render_style == "wireframe":
            circle = Circle(uv, r, fill=False, edgecolor=base_color, linewidth=1.5, linestyle="--", zorder=5)
            ax.add_patch(circle)
            return

        if p.render_style == "flat":
            circle = Circle(uv, r, color=base_color, ec="none", zorder=5)
            ax.add_patch(circle)
            return

        # Default "shaded" / "glossy" style.
        # Base fill.
        ax.add_patch(Circle(uv, r, color=base_color, ec="none", zorder=5))

        # Darker rim for volume.
        rim_color = self._darken(p.color, 0.25)
        ax.add_patch(Circle(uv, r, fill=False, edgecolor=self._to_rgba(rim_color, 0.35 * p.opacity), linewidth=0.6, zorder=6))

        # Soft diffuse shading: a darker crescent in the lower-right quadrant.
        crescent = Circle(uv + np.array([0.30 * r, -0.30 * r]), 0.70 * r,
                          color=self._to_rgba(self._darken(p.color, 0.35), 0.16 * p.opacity), zorder=5)
        ax.add_patch(crescent)

        # Main highlight upper-left.
        highlight = Circle(uv + np.array([-0.32 * r, 0.32 * r]), 0.26 * r,
                           color=(1.0, 1.0, 1.0, 0.28 * p.opacity), zorder=7)
        ax.add_patch(highlight)
        # Tiny bright spot.
        spot = Circle(uv + np.array([-0.36 * r, 0.36 * r]), 0.10 * r,
                      color=(1.0, 1.0, 1.0, 0.45 * p.opacity), zorder=7)
        ax.add_patch(spot)

    def _draw_bond(self, ax, p: Bond) -> None:
        """Draw a tube-like bond, optionally clipped and split-colored."""
        start, end = self._clip_bond(p)
        uv1 = np.asarray(self.camera.project(start)).flatten()[:2]
        uv2 = np.asarray(self.camera.project(end)).flatten()[:2]

        # Tube width in points, scaled with camera.
        lw_pt = max(1.2, self.theme.bond_width * self.camera.scale * 18)

        if getattr(self.theme, "bond_color_mode", "uniform") == "split":
            mid = (uv1 + uv2) / 2.0
            color_i = self._atom_color_at(p.site_i, (0, 0, 0))
            color_j = self._atom_color_at(p.site_j, tuple(p.jimage))
            alpha = p.opacity
            ax.plot([uv1[0], mid[0]], [uv1[1], mid[1]], color=self._to_rgba(color_i, alpha),
                    linewidth=lw_pt, solid_capstyle="round", zorder=4)
            ax.plot([mid[0], uv2[0]], [mid[1], uv2[1]], color=self._to_rgba(color_j, alpha),
                    linewidth=lw_pt, solid_capstyle="round", zorder=4)
        else:
            color = self._to_rgba(p.color, p.opacity)
            ax.plot([uv1[0], uv2[0]], [uv1[1], uv2[1]], color=color,
                    linewidth=lw_pt, solid_capstyle="round", zorder=4)

    def _atom_color_at(self, site_index: int, image_offset: tuple[int, int, int]) -> str | tuple:
        key = (site_index, tuple(image_offset))
        if key in self._atom_map:
            pos, radius = self._atom_map[key]
            # Find matching sphere to get color.
            for sp in self._sphere_list:
                meta = sp.metadata or {}
                if (meta.get("site_index"), tuple(meta.get("image_offset", (0, 0, 0)))) == key:
                    return sp.color
        return self.theme.bond_color

    def _darken(self, color, factor: float = 0.2) -> tuple[float, float, float]:
        from matplotlib.colors import to_rgb
        rgb = to_rgb(color)
        return tuple(max(0.0, c * (1.0 - factor)) for c in rgb)

    def _draw_primitive(self, ax, p, theme: FigureTheme):
        if isinstance(p, Sphere):
            self._draw_sphere(ax, p)
        elif isinstance(p, (Line, Bond, CellEdge, Cylinder)):
            if isinstance(p, Bond):
                self._draw_bond(ax, p)
            else:
                start, end = p.start, p.end
                lw = getattr(p, "linewidth", getattr(p, "radius", 1.0) * 10)
                uv1 = np.asarray(self.camera.project(start)).flatten()[:2]
                uv2 = np.asarray(self.camera.project(end)).flatten()[:2]
                alpha = p.opacity
                style = "--" if getattr(p, "dashed", False) or getattr(p, "is_back", False) else "-"
                color = self._to_rgba(p.color, alpha)
                ax.plot([uv1[0], uv2[0]], [uv1[1], uv2[1]], color=color, linewidth=lw, linestyle=style, solid_capstyle="round")
        elif isinstance(p, Polyhedron):
            if not p.faces:
                return
            # Compute face depths and camera-space normals for back-face culling.
            rot = self.camera.rotation_matrix()
            cam_vertices = (np.asarray(p.vertices, dtype=float) - self.camera.target) @ rot.T
            face_data = []
            for face in p.faces:
                pts = cam_vertices[face]
                depth = float(np.mean(pts[:, 2]))
                # Newell-method normal in camera space (only z component needed).
                n = np.zeros(3)
                for k in range(len(pts)):
                    v0 = pts[k]
                    v1 = pts[(k + 1) % len(pts)]
                    n[0] += (v0[1] - v1[1]) * (v0[2] + v1[2])
                    n[1] += (v0[2] - v1[2]) * (v0[0] + v1[0])
                    n[2] += (v0[0] - v1[0]) * (v0[1] + v1[1])
                face_data.append((depth, face, n[2]))
            # Back faces first, front faces last (painter's algorithm).
            face_data.sort(key=lambda x: x[0])

            for _, face, nz in face_data:
                pts = np.array([p.vertices[i] for i in face])
                uv = self.camera.project(pts)
                # Back faces are slightly more transparent so the cage interior reads cleanly.
                alpha = p.opacity * (0.55 if nz < 0 else 1.0)
                fill = self._to_rgba(p.fill_color or p.color, alpha)
                poly = Polygon(uv, closed=True, facecolor=fill, edgecolor="none")
                ax.add_patch(poly)

            # Draw silhouette / boundary edges only.  An edge is visible if it
            # separates a front-facing face from a back-facing face (or lies on
            # the polyhedron boundary), not if it is internal to the front set.
            edge_counts: dict[tuple[int, int], int] = {}
            edge_front: dict[tuple[int, int], int] = {}
            for _, face, nz in face_data:
                m = len(face)
                for k in range(m):
                    a, b = face[k], face[(k + 1) % m]
                    key = (min(a, b), max(a, b))
                    edge_counts[key] = edge_counts.get(key, 0) + 1
                    if nz >= 0:
                        edge_front[key] = edge_front.get(key, 0) + 1

            visible_edges: set[tuple[int, int]] = set()
            for key, total in edge_counts.items():
                front = edge_front.get(key, 0)
                # Silhouette: at least one front and one back adjacent face.
                if 0 < front < total:
                    visible_edges.add(key)
                # Boundary edge on the front surface (only one adjacent face).
                if total == 1 and front == 1:
                    visible_edges.add(key)

            edge_rgba = self._to_rgba(p.edge_color or p.color, 0.35)
            for a, b in visible_edges:
                uv = self.camera.project(np.array([p.vertices[a], p.vertices[b]]))
                ax.plot(uv[:, 0], uv[:, 1], color=edge_rgba, linewidth=max(0.35, p.edge_width * 0.6), solid_capstyle="round")
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
