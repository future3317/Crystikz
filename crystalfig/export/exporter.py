"""Unified export interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crystalfig.exceptions import ExportError
from crystalfig.export.latex import LatexCompiler
from crystalfig.renderers.base import RenderOptions
from crystalfig.renderers.matplotlib_renderer import MatplotlibRenderer
from crystalfig.renderers.tikz_renderer import TikzRenderer
from crystalfig.scene.camera import Camera
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


@dataclass
class ExportResult:
    """Result of an export operation."""

    path: str
    format: str
    vector_status: str  # "pure", "hybrid", "raster"
    metadata: dict


class Exporter:
    """Export a Scene to various publication formats."""

    def __init__(self, scene: Scene, theme: FigureTheme, camera: Camera | None = None):
        self.scene = scene
        self.theme = theme
        self.camera = camera or Camera()

    def export(self, path: str, fmt: str | None = None, options: RenderOptions | None = None) -> ExportResult:
        """Export scene to file.

        Supported formats: pdf, svg, png, tiff/tif, eps, pgf, tex, tikz.
        """
        path = Path(path)
        theme = self.theme
        fmt = (fmt or path.suffix.lstrip(".")).lower()
        options = options or RenderOptions(
            width=theme.figure_width,
            height=theme.figure_height,
            transparent=theme.transparent,
            dpi=theme.dpi,
        )

        if fmt in ("tex", "tikz"):
            renderer = TikzRenderer(camera=self.camera)
            renderer.export(self.scene, str(path), theme, options, standalone=True)
            return ExportResult(str(path), fmt, "pure", {"engine": "tikz"})

        if fmt in ("pdf", "svg", "png", "tif", "tiff", "eps", "pgf"):
            renderer = MatplotlibRenderer(camera=self.camera)
            renderer.export(self.scene, str(path), theme, options, fmt=fmt)
            vector_status = "pure" if fmt in ("pdf", "svg", "eps", "pgf") else "raster"
            return ExportResult(str(path), fmt, vector_status, {"dpi": options.dpi})

        raise ExportError(f"Unsupported export format: {fmt}")

    def export_pdf_with_latex(self, path: str, options: RenderOptions | None = None) -> ExportResult:
        """Export via TikZ and compile to PDF using LaTeX."""
        path = Path(path)
        tex_path = path.with_suffix(".tex")
        renderer = TikzRenderer(camera=self.camera)
        renderer.export(self.scene, str(tex_path), self.theme, options or RenderOptions(), standalone=True)
        compiler = LatexCompiler.detect_engine()
        if compiler is None:
            raise ExportError("No LaTeX engine found; cannot compile TikZ to PDF.")
        latex = LatexCompiler(engine=compiler)
        result = latex.compile(tex_path.read_text(encoding="utf-8"), str(path))
        if not result.success:
            raise ExportError("LaTeX compilation failed.")
        return ExportResult(str(path), "pdf", "pure", {"engine": compiler})
