"""TikZ/PGF vector renderer."""

from __future__ import annotations

import re

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
    Polyhedron,
    Polyline,
    Sphere,
    Text,
    layer_priority,
)
from crystalfig.scene.primitives import (
    Polygon as Poly,
)
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


class TikzRenderer:
    """Render a Scene to pure LaTeX TikZ code.

    The renderer projects primitives with the scene camera, sorts by depth,
    and emits TikZ commands.  It centralizes TikZ library loading to avoid
    missing-library compilation errors.
    """

    REQUIRED_LIBRARIES = [
        "shapes.geometric",
        "arrows.meta",
        "calc",
        "positioning",
        "shadings",
        "backgrounds",
        "fit",
    ]

    def __init__(self, camera, libraries: list[str] | None = None):
        self.camera = camera
        self.libraries = libraries or list(self.REQUIRED_LIBRARIES)
        self._color_map: dict = {}
        self._color_counter = 0

    def render(self, scene: Scene, theme: FigureTheme, options: RenderOptions, standalone: bool = False) -> str:
        """Render scene to TikZ string.

        Args:
            scene: The scene to render.
            theme: The figure theme.
            options: Render options.
            standalone: If True, wrap in a complete LaTeX document preamble.
        """
        self._color_map = {}
        self._color_counter = 0

        # Fit camera to the full projected scene extent.
        if self.camera.auto_fit:
            self.camera.fit_to_scene(scene)

        primitives = self._sort_by_depth(scene)
        palette = theme.palette

        lines = []
        if standalone:
            lines.append(r"\documentclass[tikz,border=10pt]{standalone}")
            lines.append(r"\usepackage{tikz}")
            lines.append(r"\usepackage{amsmath,amssymb}")
            lines.append(r"\usepackage{xcolor}")
            lines.append(r"\usetikzlibrary{" + ",".join(self.libraries) + "}")
            lines.append(r"\begin{document}")

        # Define palette colors
        for name, rgb in {**palette.element_colors, **palette.accents}.items():
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
            self._color_map[self._color_key(rgb)] = safe_name
            lines.append(f"\\definecolor{{{safe_name}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")

        # Collect arbitrary colors used in scene and define them
        for p in primitives:
            for attr in ("color", "fill_color", "edge_color"):
                val = getattr(p, attr, None)
                if val is not None:
                    self._ensure_color_defined(val, lines)

        lines.append(r"\begin{tikzpicture}[>=Stealth, font=\sffamily]")
        if options.title:
            lines.append(f"  \\node[above, font=\\sffamily\\bfseries\\small] at (0, 2.5) {{{self._escape(options.title)}}};")

        for p in primitives:
            self._draw_primitive(lines, p, theme)

        if options.show_legend and theme.show_legend:
            self._draw_legend(lines, scene, theme)

        bounds = self._projected_bounds(scene) or (-1.0, 1.0, -1.0, 1.0)
        for p in scene.all_primitives():
            if isinstance(p, Text) and getattr(p, "layer", "geometry") == "annotation":
                self._draw_annotation(lines, p, bounds)

        lines.append(r"\end{tikzpicture}")
        if standalone:
            lines.append(r"\end{document}")
        return "\n".join(lines)

    def export(self, scene: Scene, path: str, theme: FigureTheme, options: RenderOptions, standalone: bool = True) -> None:
        """Export scene to a standalone TikZ .tex file."""
        tex = self.render(scene, theme, options, standalone=standalone)
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)

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
        if isinstance(p, Poly):
            return p.points
        if isinstance(p, (Arrow, Axis)):
            return [p.start, p.start + p.direction]
        if isinstance(p, Text):
            return [p.position]
        return []

    def _draw_primitive(self, lines: list[str], p, theme: FigureTheme):
        if isinstance(p, Sphere):
            uv = np.asarray(self.camera.project(p.position)).flatten()[:2]
            color = self._color_name(p.color)
            r = p.radius * self.camera.scale
            if p.render_style == "wireframe":
                lines.append(f"  \\draw[dashed, {color}, thin] ({uv[0]:.4f}, {uv[1]:.4f}) circle ({r:.4f});")
            else:
                lines.append(f"  \\shade[shading=ball, ball color={color}, opacity={p.opacity:.2f}] ({uv[0]:.4f}, {uv[1]:.4f}) circle ({r:.4f});")
        elif isinstance(p, (Line, Bond, CellEdge, Cylinder)):
            uv1 = np.asarray(self.camera.project(p.start)).flatten()[:2]
            uv2 = np.asarray(self.camera.project(p.end)).flatten()[:2]
            color = self._color_name(p.color)
            lw = getattr(p, "linewidth", 1.0) * 0.5
            alpha = p.opacity
            dash = "dashed" if getattr(p, "dashed", False) or getattr(p, "is_back", False) else "solid"
            lines.append(f"  \\draw[{color}, line width={lw:.2f}pt, opacity={alpha:.2f}, {dash}] ({uv1[0]:.4f}, {uv1[1]:.4f}) -- ({uv2[0]:.4f}, {uv2[1]:.4f});")
        elif isinstance(p, Polyline):
            if len(p.points) < 2:
                return
            points = list(p.points)
            if p.closed:
                points.append(points[0])
            uv = self.camera.project(np.array(points))
            pts_str = " -- ".join(f"({u:.4f}, {v:.4f})" for u, v in uv)
            color = self._color_name(p.color)
            lines.append(f"  \\draw[{color}, line width={p.linewidth:.2f}pt] {pts_str};")
        elif isinstance(p, Polyhedron):
            for face in p.faces:
                pts = np.array([p.vertices[i] for i in face])
                uv = self.camera.project(pts)
                pts_str = " -- ".join(f"({u:.4f}, {v:.4f})" for u, v in uv)
                fill = self._color_name(p.fill_color or p.color)
                edge = self._color_name(p.edge_color or p.color)
                lines.append(
                    f"  \\filldraw[fill={fill}, fill opacity={p.opacity:.2f}, draw={edge}, draw opacity=0.6, line width={p.edge_width:.2f}pt] "
                    f"{pts_str} -- cycle;"
                )
        elif isinstance(p, Poly):
            if len(p.points) < 3:
                return
            uv = self.camera.project(np.array(p.points))
            pts_str = " -- ".join(f"({u:.4f}, {v:.4f})" for u, v in uv)
            fill = self._color_name(p.fill_color or p.color)
            edge = self._color_name(p.edge_color or p.color)
            lines.append(
                f"  \\filldraw[fill={fill}, fill opacity={p.opacity:.2f}, draw={edge}, draw opacity=0.8, line width={p.linewidth:.2f}pt] "
                f"{pts_str} -- cycle;"
            )
        elif isinstance(p, (Arrow, Axis)):
            uv1 = np.asarray(self.camera.project(p.start)).flatten()[:2]
            uv2 = np.asarray(self.camera.project(p.start + p.direction)).flatten()[:2]
            color = self._color_name(p.color)
            lines.append(
                f"  \\draw[very thick, ->, {color}] ({uv1[0]:.4f}, {uv1[1]:.4f}) -- ({uv2[0]:.4f}, {uv2[1]:.4f});"
            )
        elif isinstance(p, Text):
            if getattr(p, "layer", "geometry") == "annotation":
                return
            uv = np.asarray(self.camera.project(p.position)).flatten()[:2]
            text = p.text if p.raw_latex else self._escape(p.text)
            weight = "\\bfseries" if p.fontweight == "bold" else ""
            lines.append(f"  \\node[{p.halign}, font=\\sffamily {weight}] at ({uv[0]:.4f}, {uv[1]:.4f}) {{{text}}};")
        elif isinstance(p, LegendItem):
            return
        else:
            raise RenderError(f"TikzRenderer does not support primitive {type(p).__name__}.")

    def _projected_bounds(self, scene: Scene) -> tuple[float, float, float, float] | None:
        points = []
        for p in scene.all_primitives():
            if getattr(p, "layer", "geometry") == "annotation" or getattr(p, "coordinate_space", "world") == "screen":
                continue
            points.extend(self._primitive_points(p))
        if not points:
            return None
        uv = self.camera.project(np.array(points))
        xmin, ymin = uv.min(axis=0)
        xmax, ymax = uv.max(axis=0)
        return float(xmin), float(xmax), float(ymin), float(ymax)

    def _draw_annotation(
        self,
        lines: list[str],
        p: Text,
        bounds: tuple[float, float, float, float],
    ) -> None:
        xmin, xmax, ymin, ymax = bounds
        text = p.text if p.raw_latex else self._escape(p.text)
        weight = "\\bfseries" if p.fontweight == "bold" else ""
        font = f"\\sffamily {weight}"
        if p.metadata.get("kind") == "site_label":
            uv = np.asarray(self.camera.project(p.position)).flatten()[:2]
            offset = p.metadata.get("offset", (6, 2))
            lines.append(
                f"  \\node[{p.halign}, font={font}, xshift={offset[0]}pt, yshift={offset[1]}pt] "
                f"at ({uv[0]:.4f}, {uv[1]:.4f}) {{{text}}};"
            )
            return
        pos = np.asarray(p.position).flatten()
        if len(pos) < 2:
            return
        x = xmin + pos[0] * (xmax - xmin)
        y = ymin + pos[1] * (ymax - ymin)
        lines.append(f"  \\node[{p.halign}, font={font}] at ({x:.4f}, {y:.4f}) {{{text}}};")

    def _draw_legend(self, lines: list[str], scene: Scene, theme: FigureTheme):
        items = [p for p in scene.all_primitives() if isinstance(p, LegendItem)]
        if not items:
            return
        lines.append(r"  \begin{scope}[shift={(2.5, 2.0)}]")
        for idx, item in enumerate(items[:8]):
            y = -idx * 0.35
            color = self._color_name(item.color)
            lines.append(f"    \\fill[{color}] (0, {y:.2f}) circle (0.12);")
            lines.append(f"    \\node[right, font=\\sffamily\\scriptsize] at (0.25, {y:.2f}) {{{self._escape(item.text)}}};")
        lines.append(r"  \end{scope}")

    def _color_key(self, color) -> str:
        if isinstance(color, str):
            return color.lower()
        if isinstance(color, (tuple, list)):
            return "_".join(str(int(c * 255)) for c in color[:3])
        return str(color)

    def _ensure_color_defined(self, color, lines: list[str]) -> str:
        key = self._color_key(color)
        if key in self._color_map:
            return self._color_map[key]
        name = f"cfga_color_{self._color_counter}"
        self._color_counter += 1
        self._color_map[key] = name
        if isinstance(color, str) and color.startswith("#"):
            lines.append(f"\\definecolor{{{name}}}{{HTML}}{{{color.lstrip('#').upper()}}}")
        elif isinstance(color, (tuple, list)):
            rgb = tuple(int(c * 255) for c in color[:3])
            lines.append(f"\\definecolor{{{name}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")
        else:
            lines.append(f"\\definecolor{{{name}}}{{RGB}}{{128,128,128}}")
        return name

    def _color_name(self, color) -> str:
        key = self._color_key(color)
        if key in self._color_map:
            return self._color_map[key]
        if isinstance(color, str) and not color.startswith("#"):
            safe = re.sub(r"[^A-Za-z0-9_]", "_", color)
            if safe:
                return safe
        return "black"

    def _escape(self, text: str) -> str:
        """Escape special LaTeX characters in plain text."""
        if not isinstance(text, str):
            text = str(text)
        # One-pass escape so replacements already emitted are not re-processed.
        chars = {
            "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
        }
        out = []
        for ch in text:
            out.append(chars.get(ch, ch))
        return "".join(out)
