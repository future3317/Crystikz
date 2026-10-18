"""Unified export interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crystalfig.exceptions import ExportError
from crystalfig.export.latex import LatexCompiler
from crystalfig.renderers.base import RenderOptions
from crystalfig.renderers.matplotlib_3d_renderer import Matplotlib3DRenderer
from crystalfig.renderers.matplotlib_renderer import MatplotlibRenderer
from crystalfig.renderers.svg_renderer import SvgRenderer
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

    def export(
        self,
        path: str,
        fmt: str | None = None,
        options: RenderOptions | None = None,
        backend: str | None = None,
    ) -> ExportResult:
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

        if backend is not None:
            backend = backend.lower()
            if backend == "pyvista":
                if fmt != "png":
                    raise ExportError("The pyvista backend supports PNG output only.")
                from crystalfig.renderers.pyvista_renderer import PyVistaRenderer

                PyVistaRenderer(camera=self.camera).export(
                    self.scene, str(path), theme, options, fmt=fmt
                )
                return ExportResult(str(path), fmt, "raster", {
                    "renderer": "pyvista",
                    "dpi": options.dpi,
                    "pixels": [round(options.width * options.dpi / 25.4),
                               round((options.height or options.width) * options.dpi / 25.4)],
                })
            if backend == "matplotlib3d":
                if fmt not in ("pdf", "svg", "png", "tif", "tiff", "eps", "pgf"):
                    raise ExportError(f"Unsupported matplotlib3d format: {fmt}")
                return self.export_3d(str(path), options)
            if backend == "svg":
                if fmt != "svg":
                    raise ExportError("The svg backend supports SVG output only.")
                SvgRenderer(camera=self.camera).export(self.scene, str(path), theme, options)
                return ExportResult(str(path), fmt, "pure", {"renderer": "svg"})
            if backend == "tikz":
                if fmt in ("tex", "tikz"):
                    TikzRenderer(camera=self.camera).export(
                        self.scene, str(path), theme, options, standalone=True
                    )
                    return ExportResult(str(path), fmt, "pure", {"renderer": "tikz"})
                raise ExportError("The tikz backend supports TEX/TIKZ output only.")
            if backend != "matplotlib":
                raise ExportError(
                    f"Unknown renderer backend {backend!r}; use matplotlib, svg, tikz, matplotlib3d, or pyvista."
                )
            if fmt not in ("pdf", "png", "tif", "tiff", "eps", "pgf"):
                raise ExportError(f"Unsupported matplotlib format: {fmt}")
            MatplotlibRenderer(camera=self.camera).export(
                self.scene, str(path), theme, options, fmt=fmt
            )
            vector_status = "pure" if fmt in ("pdf", "eps", "pgf") else "raster"
            return ExportResult(str(path), fmt, vector_status, {"renderer": "matplotlib", "dpi": options.dpi})

        if fmt in ("tex", "tikz"):
            renderer = TikzRenderer(camera=self.camera)
            renderer.export(self.scene, str(path), theme, options, standalone=True)
            return ExportResult(str(path), fmt, "pure", {"engine": "tikz"})

        if fmt == "svg":
            renderer = SvgRenderer(camera=self.camera)
            renderer.export(self.scene, str(path), theme, options)
            return ExportResult(str(path), "svg", "pure", {"renderer": "svg"})

        if fmt in ("pdf", "png", "tif", "tiff", "eps", "pgf"):
            renderer = MatplotlibRenderer(camera=self.camera)
            renderer.export(self.scene, str(path), theme, options, fmt=fmt)
            vector_status = "pure" if fmt in ("pdf", "eps", "pgf") else "raster"
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

    def export_hybrid_pdf(
        self,
        path: str,
        options: RenderOptions | None = None,
        background_backend: str = "pyvista",
    ) -> ExportResult:
        """Put a high-resolution 3D render behind editable screen-space labels."""
        import tempfile

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        options = options or RenderOptions(
            width=self.theme.figure_width,
            height=self.theme.figure_height,
            dpi=self.theme.dpi,
            transparent=self.theme.transparent,
        )
        pdf_fonttype = matplotlib.rcParams["pdf.fonttype"]
        ps_fonttype = matplotlib.rcParams["ps.fonttype"]
        fig = None
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            image_path = Path(temp.name)
        try:
            self.export(str(image_path), fmt="png", options=options, backend=background_backend)
            width_in = options.width / 25.4
            height_in = (options.height or options.width) / 25.4
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42
            fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=options.dpi)
            with Image.open(image_path) as image:
                ax.imshow(np.asarray(image).copy(), extent=(0, 1, 0, 1), origin="lower", aspect="auto")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_position([0, 0, 1, 1])
            ax.set_aspect("auto")
            ax.axis("off")

            from crystalfig.scene.primitives import Text

            annotations = [
                primitive for primitive in self.scene.all_primitives()
                if isinstance(primitive, Text) and primitive.layer == "annotation"
            ]
            camera = self.camera
            if camera.auto_fit:
                camera.fit_to_scene(self.scene)
            aspect = options.width / (options.height or options.width)
            aspect_scale = min(1.0, aspect)

            def annotation_position(position):
                projected = camera.project(position)[0]
                x = 0.5 + projected[0] * aspect_scale / (2.0 * aspect)
                y = 0.5 + projected[1] * aspect_scale / 2.0
                return x, y

            for annotation in annotations:
                meta = annotation.metadata or {}
                if meta.get("kind") == "site_label":
                    x, y = annotation_position(annotation.position)
                    offset = meta.get("offset", (6, 2))
                    ax.annotate(
                        annotation.text, (x, y), xytext=offset, textcoords="offset points",
                        fontsize=annotation.fontsize, color=annotation.color,
                        fontweight=annotation.fontweight, ha=annotation.halign,
                        va=annotation.valign,
                    )
                else:
                    x, y = annotation.position[:2]
                    ax.text(
                        x, y, annotation.text, transform=ax.transAxes,
                        fontsize=annotation.fontsize, color=annotation.color,
                        fontweight=annotation.fontweight, ha=annotation.halign,
                        va=annotation.valign,
                    )
            fig.savefig(path, format="pdf", transparent=options.transparent)
            plt.close(fig)
        finally:
            if fig is not None:
                plt.close(fig)
            matplotlib.rcParams["pdf.fonttype"] = pdf_fonttype
            matplotlib.rcParams["ps.fonttype"] = ps_fonttype
            image_path.unlink(missing_ok=True)
        return ExportResult(
            str(path), "pdf", "hybrid",
            {"renderer": background_backend, "dpi": options.dpi, "vector_annotations": len(annotations)},
        )

    def export_3d(self, path: str, options: RenderOptions | None = None) -> ExportResult:
        """Export using Matplotlib's true 3D projection (mpl_toolkits.mplot3d)."""
        path = Path(path)
        fmt = path.suffix.lstrip(".").lower()
        options = options or RenderOptions(
            width=self.theme.figure_width,
            height=self.theme.figure_height,
            transparent=self.theme.transparent,
            dpi=self.theme.dpi,
        )
        renderer = Matplotlib3DRenderer(camera=self.camera)
        renderer.export(self.scene, str(path), self.theme, options, fmt=fmt)
        # Matplotlib 3D projection bakes the view into the output; vector formats
        # contain vector primitives but are not a pure, camera-independent scene.
        vector_status = "hybrid" if fmt in ("pdf", "svg", "eps", "pgf") else "raster"
        return ExportResult(str(path), fmt, vector_status, {"renderer": "matplotlib3d"})
