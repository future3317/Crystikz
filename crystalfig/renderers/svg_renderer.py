"""Pure SVG publication renderer for crystal scenes."""

from __future__ import annotations

import io
from xml.sax.saxutils import escape

import numpy as np

from crystalfig.exceptions import RenderError
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


class SvgRenderer:
    """Render a Scene to a pure SVG publication figure.

    This renderer is intentionally lightweight: it projects 3D primitives with
    the same orthographic camera used by the Matplotlib backend, sorts them by
    camera depth, and emits plain SVG.  It targets the Nature-style crystal
    illustrations the user asked for: vector glossy-shaded atoms, smooth
    split-colour bonds, translucent polyhedra, and thin cell edges.
    """

    def __init__(self, camera):
        self.camera = camera

    def render(self, scene: Scene, theme: FigureTheme, options: RenderOptions) -> str:
        """Render scene to an SVG string."""
        buf = io.StringIO()
        self.export(scene, buf, theme, options)
        return buf.getvalue()

    def export(
        self,
        scene: Scene,
        path,
        theme: FigureTheme,
        options: RenderOptions,
        fmt: str | None = None,
    ) -> None:
        """Export scene to an SVG file or file-like object."""
        if self.camera.auto_fit:
            self.camera.fit_to_scene(scene)

        bounds = self._projected_bounds(scene)
        if bounds is None:
            bounds = (-1.0, 1.0, -1.0, 1.0)
        xmin, xmax, ymin, ymax = bounds
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)

        self.theme = theme
        width_mm = options.width
        if options.height:
            height_mm = options.height
        else:
            aspect = dy / dx
            aspect = max(0.35, min(aspect, 2.8))
            height_mm = width_mm * aspect

        self._user_units_per_mm = dx / width_mm
        self._user_units_per_point = self._user_units_per_mm * 25.4 / 72.0
        self._svg_y_sum = ymin + ymax

        # SVG viewBox uses camera units; physical size is in mm.
        svg_attrs = (
            f'xmlns="http://www.w3.org/2000/svg" '
            f'width="{width_mm:.3f}mm" '
            f'height="{height_mm:.3f}mm" '
            f'viewBox="{xmin:.4f} {ymin:.4f} {dx:.4f} {dy:.4f}"'
        )

        bg = options.background or theme.background
        transparent = options.transparent or bg == "transparent"

        self._atom_map = self._build_atom_map(scene)
        self._sphere_list = [p for p in scene.all_primitives() if isinstance(p, Sphere)]

        lines = [f"<svg {svg_attrs}>"]
        lines.append("  <defs>")
        lines.extend(self._gradient_defs(scene))
        lines.append("  </defs>")

        if not transparent and bg and bg != "transparent":
            lines.append(
                f'  <rect x="{xmin:.4f}" y="{ymin:.4f}" width="{dx:.4f}" height="{dy:.4f}" '
                f'fill="{self._color_to_hex(bg)}"/>'
            )

        if options.title:
            lines.append(
                f'  <text x="{(xmin + xmax) / 2:.4f}" y="{ymin:.4f}" '
                f'text-anchor="middle" font-size="{self._pt_to_user(theme.title_size):.4f}" '
                f'fill="black">{escape(options.title)}</text>'
            )

        primitives = self._sort_by_depth(scene)
        for p in primitives:
            lines.extend(self._draw_primitive(p))

        if options.show_legend and theme.show_legend:
            lines.extend(self._draw_legend(scene, xmin, xmax, ymin, ymax))

        # Annotations are drawn last so they sit on top of the scene.
        for p in scene.all_primitives():
            if isinstance(p, Text) and getattr(p, "layer", "default") == "annotation":
                lines.extend(self._draw_annotation(p, xmin, xmax, ymin, ymax))

        lines.append("</svg>")
        svg = "\n".join(lines)

        if isinstance(path, str):
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
        else:
            try:
                path.write(svg)
            except TypeError:
                path.write(svg.encode("utf-8"))

    # ------------------------------------------------------------------
    # Bounds and depth sorting
    # ------------------------------------------------------------------

    def _projected_bounds(self, scene: Scene) -> tuple[float, float, float, float] | None:
        pts = []
        for p in scene.all_primitives():
            if getattr(p, "layer", "geometry") == "annotation" or getattr(p, "coordinate_space", "world") == "screen":
                continue
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
            elif isinstance(p, Polyline):
                pts.extend(self.camera.project(np.array(p.points)))
            elif isinstance(p, Polyhedron):
                pts.extend(self.camera.project(np.array(p.vertices)))
            elif isinstance(p, Polygon):
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
        margin = self.camera.padding
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        return (xmin - margin * dx, xmax + margin * dx, ymin - margin * dy, ymax + margin * dy)

    def _build_atom_map(self, scene: Scene) -> dict:
        atom_map = {}
        for p in scene.all_primitives():
            if isinstance(p, Sphere):
                meta = p.metadata or {}
                key = (meta.get("site_index", -1), tuple(meta.get("image_offset", (0, 0, 0))))
                if key[0] >= 0 and key not in atom_map:
                    atom_map[key] = (p.position, p.radius)
        return atom_map

    def _sort_by_depth(self, scene: Scene) -> list:
        scored = []
        for p in scene.all_primitives():
            if getattr(p, "layer", "geometry") == "annotation" or getattr(p, "coordinate_space", "world") == "screen":
                continue
            pts = self._primitive_points(p)
            if len(pts) == 0:
                depth = 1e9
            else:
                depth = float(np.mean(self.camera.depth(np.array(pts))))
            scored.append((layer_priority(p), depth, p))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [p for _, _, p in scored]

    def _primitive_points(self, p) -> list[np.ndarray]:
        if isinstance(p, Sphere):
            return [p.position]
        if isinstance(p, (Line, Bond, CellEdge, Cylinder)):
            return [p.start, p.end]
        if isinstance(p, Polyline):
            return p.points
        if isinstance(p, Polyhedron):
            return p.vertices
        if isinstance(p, Polygon):
            return p.points
        if isinstance(p, (Arrow, Axis)):
            return [p.start, p.start + p.direction]
        if isinstance(p, Text):
            return [p.position]
        return []

    # ------------------------------------------------------------------
    # Gradient definitions
    # ------------------------------------------------------------------

    def _gradient_defs(self, scene: Scene) -> list[str]:
        """Emit SVG defs for gradients needed by atoms and bonds."""
        defs = []
        atom_ids = set()
        bond_ids = set()
        for p in scene.all_primitives():
            if isinstance(p, Sphere) and p.render_style != "flat" and p.render_style != "wireframe":
                sid = self._sphere_grad_id(p)
                if sid not in atom_ids:
                    atom_ids.add(sid)
                    defs.extend(self._sphere_gradient_def(p, sid))
            elif isinstance(p, Bond):
                mode = getattr(self.theme, "bond_color_mode", "uniform")
                if mode in ("split", "split_soft"):
                    bid = self._bond_grad_id(p)
                    if bid not in bond_ids:
                        bond_ids.add(bid)
                        defs.extend(self._bond_gradient_def(p, bid, mode))
        return defs

    def _sphere_grad_id(self, p: Sphere) -> str:
        base = self._color_to_hex(p.color).lstrip("#")
        return f"gsphere_{base}_{id(p) & 0xFFFFFF}"

    def _sphere_gradient_def(self, p: Sphere, grad_id: str) -> list[str]:
        """Define the clip path used by glossy atom shading."""
        r = p.radius * self.camera.scale
        # We do not emit a true radialGradient because the crescent/highlight
        # are offset circles; instead we use the gradient id as a marker and
        # draw the shading with explicit clipPath below.
        cid = f"cp_{grad_id}"
        uv = self._project_uv(p.position)
        return [
            f'    <clipPath id="{cid}">',
            f'      <circle cx="{uv[0]:.4f}" cy="{uv[1]:.4f}" r="{r:.4f}"/>',
            '    </clipPath>',
        ]

    def _bond_grad_id(self, p: Bond) -> str:
        return f"gbond_{id(p) & 0xFFFFFF}"

    def _bond_gradient_def(self, p: Bond, grad_id: str, mode: str) -> list[str]:
        color_i = self._atom_color_at(p.site_i, (0, 0, 0))
        color_j = self._atom_color_at(p.site_j, tuple(p.jimage))
        c1 = self._color_to_hex(color_i)
        c2 = self._color_to_hex(color_j)
        start, end = self._clip_bond(p)
        uv1 = self._project_uv(start)
        uv2 = self._project_uv(end)
        if mode == "split":
            mid = (uv1 + uv2) / 2.0
            return [
                f'    <linearGradient id="{grad_id}_i" gradientUnits="userSpaceOnUse" '
                f'x1="{uv1[0]:.4f}" y1="{uv1[1]:.4f}" x2="{mid[0]:.4f}" y2="{mid[1]:.4f}">',
                f'      <stop offset="0%" stop-color="{c1}"/>',
                f'      <stop offset="100%" stop-color="{c1}"/>',
                '    </linearGradient>',
                f'    <linearGradient id="{grad_id}_j" gradientUnits="userSpaceOnUse" '
                f'x1="{mid[0]:.4f}" y1="{mid[1]:.4f}" x2="{uv2[0]:.4f}" y2="{uv2[1]:.4f}">',
                f'      <stop offset="0%" stop-color="{c2}"/>',
                f'      <stop offset="100%" stop-color="{c2}"/>',
                '    </linearGradient>',
            ]
        # split_soft: blend from c1 to a muted midpoint and back to c2.
        cmid = self._blend_colors(c1, c2, 0.5)
        cmid_hex = self._color_to_hex(cmid)
        return [
            f'    <linearGradient id="{grad_id}" gradientUnits="userSpaceOnUse" '
            f'x1="{uv1[0]:.4f}" y1="{uv1[1]:.4f}" x2="{uv2[0]:.4f}" y2="{uv2[1]:.4f}">',
            f'      <stop offset="0%" stop-color="{c1}"/>',
            f'      <stop offset="50%" stop-color="{cmid_hex}"/>',
            f'      <stop offset="100%" stop-color="{c2}"/>',
            '    </linearGradient>',
        ]

    # ------------------------------------------------------------------
    # Primitive drawing
    # ------------------------------------------------------------------

    def _draw_primitive(self, p) -> list[str]:
        if isinstance(p, Sphere):
            return self._draw_sphere(p)
        if isinstance(p, Bond):
            return self._draw_bond(p)
        if isinstance(p, (Line, CellEdge)):
            return self._draw_line(p)
        if isinstance(p, Cylinder):
            return self._draw_line(p)
        if isinstance(p, Polyline):
            return self._draw_polyline(p)
        if isinstance(p, Polyhedron):
            return self._draw_polyhedron(p)
        if isinstance(p, Polygon):
            return self._draw_polygon(p)
        if isinstance(p, (Arrow, Axis)):
            return self._draw_arrow(p)
        if isinstance(p, Text):
            if getattr(p, "layer", "geometry") == "annotation":
                return []
            return self._draw_text(p)
        if isinstance(p, LegendItem):
            return []
        raise RenderError(f"SvgRenderer does not support primitive {type(p).__name__}.")

    def _draw_sphere(self, p: Sphere) -> list[str]:
        uv = self._project_uv(p.position)
        r = p.radius * self.camera.scale
        cx, cy = uv
        base_hex = self._color_to_hex(p.color)
        elements = []

        if p.render_style == "wireframe":
            elements.append(
                f'  <circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r:.4f}" '
                f'fill="none" stroke="{base_hex}" stroke-width="{self._pt_to_user(1.5):.4f}" '
                f'stroke-dasharray="{self._pt_to_user(1.5):.4f},{self._pt_to_user(1.5):.4f}"/>'
            )
            return elements

        if p.render_style == "flat":
            elements.append(
                f'  <circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r:.4f}" '
                f'fill="{base_hex}" fill-opacity="{p.opacity:.3f}"/>'
            )
            return elements

        grad_id = self._sphere_grad_id(p)
        cid = f"cp_{grad_id}"

        # Base disk.
        elements.append(
            f'  <circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r:.4f}" '
            f'fill="{base_hex}" fill-opacity="{p.opacity:.3f}"/>'
        )

        # Subtle darker rim.
        rim = self._darken(p.color, 0.25)
        elements.append(
            f'  <circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r:.4f}" '
            f'fill="none" stroke="{self._color_to_hex(rim)}" '
                f'stroke-width="{self._pt_to_user(0.6):.4f}" stroke-opacity="{0.35 * p.opacity:.3f}"/>'
        )

        # Soft lower-right crescent (diffuse shadow).
        crescent_r = 0.72 * r
        crescent_cx = cx + 0.28 * r
        crescent_cy = cy + 0.28 * r
        crescent_color = self._color_to_hex(self._darken(p.color, 0.30))
        elements.append(
            f'  <circle cx="{crescent_cx:.4f}" cy="{crescent_cy:.4f}" r="{crescent_r:.4f}" '
            f'fill="{crescent_color}" fill-opacity="{0.14 * p.opacity:.3f}" '
            f'clip-path="url(#{cid})"/>'
        )

        # Soft upper-left highlight.
        hi_r = 0.38 * r
        hi_cx = cx - 0.30 * r
        hi_cy = cy - 0.30 * r
        elements.append(
            f'  <circle cx="{hi_cx:.4f}" cy="{hi_cy:.4f}" r="{hi_r:.4f}" '
            f'fill="white" fill-opacity="{0.22 * p.opacity:.3f}" '
            f'clip-path="url(#{cid})"/>'
        )

        return elements

    def _draw_bond(self, p: Bond) -> list[str]:
        start, end = self._clip_bond(p)
        uv1 = self._project_uv(start)
        uv2 = self._project_uv(end)
        lw_pt = max(1.2, self.theme.bond_width * self.camera.scale * 18)
        lw = self._pt_to_user(lw_pt)
        mode = getattr(self.theme, "bond_color_mode", "uniform")

        if mode in ("split", "split_soft"):
            grad_id = self._bond_grad_id(p)
            if mode == "split":
                mid = (uv1 + uv2) / 2.0
                return [
                    f'  <line x1="{uv1[0]:.4f}" y1="{uv1[1]:.4f}" x2="{mid[0]:.4f}" y2="{mid[1]:.4f}" '
                    f'stroke="url(#{grad_id}_i)" stroke-width="{lw:.3f}" stroke-linecap="round" '
                    f'stroke-opacity="{p.opacity:.3f}"/>',
                    f'  <line x1="{mid[0]:.4f}" y1="{mid[1]:.4f}" x2="{uv2[0]:.4f}" y2="{uv2[1]:.4f}" '
                    f'stroke="url(#{grad_id}_j)" stroke-width="{lw:.3f}" stroke-linecap="round" '
                    f'stroke-opacity="{p.opacity:.3f}"/>',
                ]
            return [
                f'  <line x1="{uv1[0]:.4f}" y1="{uv1[1]:.4f}" x2="{uv2[0]:.4f}" y2="{uv2[1]:.4f}" '
                f'stroke="url(#{grad_id})" stroke-width="{lw:.3f}" stroke-linecap="round" '
                f'stroke-opacity="{p.opacity:.3f}"/>'
            ]

        color = self._color_to_hex(p.color)
        return [
            f'  <line x1="{uv1[0]:.4f}" y1="{uv1[1]:.4f}" x2="{uv2[0]:.4f}" y2="{uv2[1]:.4f}" '
            f'stroke="{color}" stroke-width="{lw:.3f}" stroke-linecap="round" '
            f'stroke-opacity="{p.opacity:.3f}"/>'
        ]

    def _clip_bond(self, bond: Bond) -> tuple[np.ndarray, np.ndarray]:
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

    def _draw_line(self, p: Line | CellEdge) -> list[str]:
        uv1 = self._project_uv(p.start)
        uv2 = self._project_uv(p.end)
        color = self._color_to_hex(p.color)
        lw = self._pt_to_user(getattr(p, "linewidth", 0.5))
        alpha = p.opacity
        if isinstance(p, CellEdge) and p.is_back:
            alpha *= 0.5
        return [
            f'  <line x1="{uv1[0]:.4f}" y1="{uv1[1]:.4f}" x2="{uv2[0]:.4f}" y2="{uv2[1]:.4f}" '
            f'stroke="{color}" stroke-width="{lw:.3f}" stroke-opacity="{alpha:.3f}" '
            f'stroke-linecap="round"/>'
        ]

    def _draw_polyhedron(self, p: Polyhedron) -> list[str]:
        if not p.faces:
            return []

        rot = self.camera.rotation_matrix()
        cam_vertices = (np.asarray(p.vertices, dtype=float) - self.camera.target) @ rot.T
        face_data = []
        for face in p.faces:
            pts = cam_vertices[face]
            depth = float(np.mean(pts[:, 2]))
            n = np.zeros(3)
            for k in range(len(pts)):
                v0 = pts[k]
                v1 = pts[(k + 1) % len(pts)]
                n[0] += (v0[1] - v1[1]) * (v0[2] + v1[2])
                n[1] += (v0[2] - v1[2]) * (v0[0] + v1[0])
                n[2] += (v0[0] - v1[0]) * (v0[1] + v1[1])
            face_data.append((depth, face, n[2]))
        face_data.sort(key=lambda x: x[0])

        elements = []
        fill_hex = self._color_to_hex(p.fill_color or p.color)
        for _, face, nz in face_data:
            pts = np.array([p.vertices[i] for i in face])
            uv = self._project_uv(pts)
            alpha = p.opacity * (0.18 if nz < 0 else 0.55)
            points_str = " ".join(f"{x:.4f},{y:.4f}" for x, y in uv)
            elements.append(
                f'  <polygon points="{points_str}" fill="{fill_hex}" '
                f'fill-opacity="{alpha:.3f}" stroke="none"/>'
            )

        # True hull edges: keep non-coplanar adjacent face edges and boundary edges.
        edge_faces: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = {}
        for _, face, nz in face_data:
            m = len(face)
            pts = np.array([cam_vertices[i] for i in face])
            n = np.zeros(3)
            for k in range(m):
                v0 = pts[k]
                v1 = pts[(k + 1) % m]
                n[0] += (v0[1] - v1[1]) * (v0[2] + v1[2])
                n[1] += (v0[2] - v1[2]) * (v0[0] + v1[0])
                n[2] += (v0[0] - v1[0]) * (v0[1] + v1[1])
            n = n / (np.linalg.norm(n) + 1e-12)
            for k in range(m):
                a, b = face[k], face[(k + 1) % m]
                key = (min(a, b), max(a, b))
                edge_faces.setdefault(key, []).append((n, nz))

        visible_edges: set[tuple[int, int]] = set()
        for key, faces in edge_faces.items():
            if len(faces) == 1:
                visible_edges.add(key)
            else:
                n1 = faces[0][0]
                n2 = faces[1][0]
                if abs(np.dot(n1, n2)) < 0.985:
                    visible_edges.add(key)

        edge_hex = self._color_to_hex(p.edge_color or p.color)
        for a, b in visible_edges:
            uv = self._project_uv(np.array([p.vertices[a], p.vertices[b]]))
            mid_cam_z = float(np.mean([cam_vertices[a][2], cam_vertices[b][2]]))
            is_front = mid_cam_z >= 0
            alpha = 0.45 if is_front else 0.22
            lw = self._pt_to_user(max(0.35, p.edge_width * (0.9 if is_front else 0.55)))
            elements.append(
                f'  <line x1="{uv[0,0]:.4f}" y1="{uv[0,1]:.4f}" x2="{uv[1,0]:.4f}" y2="{uv[1,1]:.4f}" '
                f'stroke="{edge_hex}" stroke-width="{lw:.3f}" stroke-opacity="{alpha:.3f}" '
                f'stroke-linecap="round"/>'
            )

        return elements

    def _draw_polygon(self, p: Polygon) -> list[str]:
        if len(p.points) < 3:
            return []
        uv = self._project_uv(np.array(p.points))
        points_str = " ".join(f"{x:.4f},{y:.4f}" for x, y in uv)
        fill = self._color_to_hex(p.fill_color or p.color)
        edge = self._color_to_hex(p.edge_color or p.color)
        return [
            f'  <polygon points="{points_str}" fill="{fill}" fill-opacity="{p.opacity:.3f}" '
                f'stroke="{edge}" stroke-width="{self._pt_to_user(p.linewidth):.4f}" stroke-opacity="0.8"/>'
        ]

    def _draw_arrow(self, p: Arrow | Axis) -> list[str]:
        uv1 = self._project_uv(p.start)
        uv2 = self._project_uv(p.start + p.direction)
        color = self._color_to_hex(p.color)
        shaft_w = self._pt_to_user(p.shaft_radius * self.camera.scale * 18)
        head_len = np.linalg.norm(p.direction) * self.camera.scale * 0.25
        head_w = head_len * 0.6
        elements = [
            f'  <line x1="{uv1[0]:.4f}" y1="{uv1[1]:.4f}" x2="{uv2[0]:.4f}" y2="{uv2[1]:.4f}" '
            f'stroke="{color}" stroke-width="{shaft_w:.3f}" stroke-linecap="round"/>'
        ]

        # Arrowhead triangle at uv2.
        if head_len > 1e-6:
            direction = uv2 - uv1
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            perp = np.array([-direction[1], direction[0]])
            tip = uv2
            base = uv2 - direction * head_len
            left = base + perp * head_w * 0.5
            right = base - perp * head_w * 0.5
            elements.append(
                f'  <polygon points="{tip[0]:.4f},{tip[1]:.4f} {left[0]:.4f},{left[1]:.4f} '
                f'{right[0]:.4f},{right[1]:.4f}" fill="{color}"/>'
            )

        if isinstance(p, Axis) and p.label:
            # Place label near the arrow tip.
            label_pos = uv2 + (uv2 - uv1) * 0.12
            elements.append(
                f'  <text x="{label_pos[0]:.4f}" y="{label_pos[1]:.4f}" '
                f'font-size="{self._pt_to_user(self.theme.label_size):.4f}" fill="{color}">'
                f'{escape(p.label)}</text>'
            )
        return elements

    def _draw_text(self, p: Text) -> list[str]:
        uv = self._project_uv(p.position)
        color = self._color_to_hex(p.color)
        anchor = {"left": "start", "center": "middle", "right": "end"}.get(p.halign, "middle")
        baseline = {"top": "auto", "center": "middle", "bottom": "auto"}.get(p.valign, "middle")
        return [
            f'  <text x="{uv[0]:.4f}" y="{uv[1]:.4f}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}" font-size="{self._pt_to_user(p.fontsize):.4f}" '
            f'fill="{color}">{escape(p.text)}</text>'
        ]

    def _draw_annotation(
        self,
        p: Text,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> list[str]:
        """Draw a screen-space annotation on top of the SVG."""
        meta = p.metadata or {}
        kind = meta.get("kind", "")
        color = self._color_to_hex(p.color)
        anchor = {"left": "start", "center": "middle", "right": "end"}.get(p.halign, "middle")
        baseline = {"top": "auto", "center": "middle", "bottom": "auto"}.get(p.valign, "middle")
        font_size = self._pt_to_user(p.fontsize)
        weight = getattr(p, "fontweight", "normal")
        weight_attr = f' font-weight="{weight}"' if weight != "normal" else ""

        if kind == "site_label":
            uv = self._project_uv(p.position)
            offset = meta.get("offset", (6, 2))
            x = uv[0] + self._pt_to_user(offset[0])
            y = uv[1] - self._pt_to_user(offset[1])
            return [
                f'  <text x="{x:.4f}" y="{y:.4f}" text-anchor="{anchor}" '
                f'dominant-baseline="{baseline}" font-size="{font_size:.4f}" '
                f'fill="{color}"{weight_attr}>{escape(p.text)}</text>'
            ]

        pos = np.asarray(p.position).flatten()
        if len(pos) < 2:
            return []
        x = xmin + pos[0] * (xmax - xmin)
        y = ymin + (1.0 - pos[1]) * (ymax - ymin)
        return [
            f'  <text x="{x:.4f}" y="{y:.4f}" text-anchor="{anchor}" '
                f'dominant-baseline="{baseline}" font-size="{font_size:.4f}" '
            f'fill="{color}"{weight_attr}>{escape(p.text)}</text>'
        ]

    def _draw_legend(
        self,
        scene: Scene,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> list[str]:
        items = [p for p in scene.all_primitives() if isinstance(p, LegendItem)]
        if not items:
            return []
        # Simple top-right legend laid out in screen space.
        x0 = xmax - (xmax - xmin) * 0.25
        y0 = ymin + (ymax - ymin) * 0.05
        step = (ymax - ymin) * 0.04
        elements = []
        for k, item in enumerate(items[:8]):
            y = y0 + k * step
            color = self._color_to_hex(item.color)
            elements.append(
                f'  <circle cx="{x0:.4f}" cy="{y:.4f}" r="{step * 0.25:.4f}" fill="{color}"/>'
            )
            elements.append(
                f'  <text x="{x0 + step * 0.4:.4f}" y="{y:.4f}" '
                f'font-size="{self._pt_to_user(self.theme.label_size):.4f}" fill="black">'
                f'{escape(item.text)}</text>'
            )
        return elements

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pt_to_user(self, points: float) -> float:
        """Convert a physical point value to the current SVG viewBox units."""
        return float(points) * self._user_units_per_point

    def _project_uv(self, points: np.ndarray) -> np.ndarray:
        """Project world points and flip y to match SVG's downward axis."""
        points = np.asarray(points, dtype=float)
        single = points.ndim == 1
        uv = np.asarray(self.camera.project(points), dtype=float)
        if single:
            uv = uv[0]
        uv = uv.copy()
        uv[..., 1] = self._svg_y_sum - uv[..., 1]
        return uv

    def _draw_polyline(self, p: Polyline) -> list[str]:
        if len(p.points) < 2:
            return []
        points = list(p.points)
        if p.closed:
            points.append(points[0])
        uv = self._project_uv(np.array(points))
        points_str = " ".join(f"{x:.4f},{y:.4f}" for x, y in uv)
        color = self._color_to_hex(p.color)
        return [
            f'  <polyline points="{points_str}" fill="none" stroke="{color}" '
            f'stroke-width="{self._pt_to_user(p.linewidth):.4f}" '
            f'stroke-opacity="{p.opacity:.3f}" stroke-linecap="round"/>'
        ]

    def _atom_color_at(self, site_index: int, image_offset: tuple[int, int, int]) -> str | tuple:
        key = (site_index, tuple(image_offset))
        if key in self._atom_map:
            for sp in self._sphere_list:
                meta = sp.metadata or {}
                if (meta.get("site_index"), tuple(meta.get("image_offset", (0, 0, 0)))) == key:
                    return sp.color
        return self.theme.bond_color

    def _color_to_hex(self, color) -> str:
        from matplotlib.colors import to_hex
        try:
            return to_hex(color, keep_alpha=False)
        except Exception:
            return str(color)

    def _blend_colors(self, color_a, color_b, t: float) -> tuple[float, float, float]:
        from matplotlib.colors import to_rgb
        a = np.array(to_rgb(color_a))
        b = np.array(to_rgb(color_b))
        return tuple((1.0 - t) * a + t * b)

    def _darken(self, color, factor: float = 0.2) -> tuple[float, float, float]:
        from matplotlib.colors import to_rgb
        rgb = to_rgb(color)
        return tuple(max(0.0, c * (1.0 - factor)) for c in rgb)
