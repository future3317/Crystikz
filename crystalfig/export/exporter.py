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
from crystalfig.renderers.registry import get_renderer
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
            try:
                renderer, capabilities = get_renderer(backend, self.camera)
            except ValueError as exc:
                raise ExportError(str(exc)) from exc
            if backend == "pyvista":
                if fmt == "pdf":
                    return self.export_hybrid_pdf(str(path), options, background_backend="pyvista")
                if fmt not in capabilities.formats:
                    raise ExportError("The pyvista backend supports PNG output and hybrid PDF.")
                pixels = renderer._window_size(options, self.scene, self.camera)
                renderer.export(self.scene, str(path), theme, options, fmt=fmt)
                return ExportResult(str(path), fmt, "raster", {
                    "renderer": "pyvista",
                    "dpi": options.dpi,
                    "pixels": list(pixels),
                    "depth_peeling_used": renderer.last_depth_peeling_used,
                })
            if backend == "matplotlib3d":
                if fmt not in capabilities.formats:
                    raise ExportError(f"Unsupported matplotlib3d format: {fmt}")
                renderer.export(self.scene, str(path), theme, options, fmt=fmt)
                vector_status = "hybrid" if fmt in capabilities.vector_formats else "raster"
                return ExportResult(str(path), fmt, vector_status, {"renderer": "matplotlib3d"})
            if backend == "svg":
                if fmt not in capabilities.formats:
                    raise ExportError("The svg backend supports SVG output only.")
                renderer.export(self.scene, str(path), theme, options)
                return ExportResult(str(path), fmt, "pure", {"renderer": "svg"})
            if backend == "tikz":
                if fmt not in capabilities.formats:
                    raise ExportError("The tikz backend supports TEX/TIKZ output only.")
                renderer.export(self.scene, str(path), theme, options, standalone=True)
                return ExportResult(str(path), fmt, "pure", {"renderer": "tikz"})
            if fmt not in capabilities.formats:
                raise ExportError(f"Unsupported matplotlib format: {fmt}")
            renderer.export(self.scene, str(path), theme, options, fmt=fmt)
            vector_status = "pure" if fmt in capabilities.vector_formats else "raster"
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

        from crystalfig.scene.primitives import Text

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
            background_scene = self.scene.filter(
                lambda primitive: not (
                    isinstance(primitive, Text)
                    and getattr(primitive, "layer", "geometry") == "annotation"
                )
            )
            background_exporter = Exporter(background_scene, self.theme, self.camera)
            background_result = background_exporter.export(
                str(image_path), fmt="png", options=options, backend=background_backend
            )
            width_in = options.width / 25.4
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42
            with Image.open(image_path) as image:
                pixels = np.asarray(image).copy()
                height_in = width_in * image.height / image.width
            fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=options.dpi)
            ax.imshow(pixels, extent=(0, 1, 0, 1), origin="upper", aspect="auto")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_position([0, 0, 1, 1])
            ax.set_aspect("auto")
            ax.axis("off")

            annotations = [
                primitive for primitive in self.scene.all_primitives()
                if isinstance(primitive, Text) and primitive.layer == "annotation"
            ]
            camera = self.camera
            if camera.auto_fit:
                camera.fit_to_scene(background_scene)

            def annotation_position(position):
                camera_position = (np.asarray(position) - camera.target) @ camera.rotation_matrix().T
                aspect = pixels.shape[1] / pixels.shape[0]
                if camera.projection == "perspective":
                    sphere = camera.scene_bounding_sphere(background_scene)
                    radius = sphere[1] if sphere is not None else 1.0
                    distance = max(
                        camera.distance,
                        radius / max(np.tan(np.radians(camera.fov / 2)), 1e-6) * 1.1 + radius,
                    )
                    depth = max(distance - camera_position[2], 1e-6)
                    half_height = depth * np.tan(np.radians(camera.fov / 2))
                    return (
                        0.5 + camera_position[0] / (2 * half_height * aspect),
                        0.5 + camera_position[1] / (2 * half_height),
                    )
                frame = camera.scene_frame(background_scene)
                if frame is None:
                    return 0.5, 0.5
                half_height = max(frame.height / 2, frame.width / (2 * aspect)) / max(1 - camera.padding, 1e-6)
                half_width = half_height * aspect
                return (
                    0.5 + camera_position[0] / (2 * half_width),
                    0.5 + camera_position[1] / (2 * half_height),
                )

            for annotation in annotations:
                meta = annotation.metadata or {}
                if meta.get("kind") == "site_label" or annotation.coordinate_space == "world":
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
            {
                "renderer": background_backend,
                "dpi": options.dpi,
                "vector_annotations": len(annotations),
                "depth_peeling_used": background_result.metadata.get("depth_peeling_used"),
            },
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
