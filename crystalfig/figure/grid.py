"""Matplotlib composition for publication panels containing crystal structures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from crystalfig.export.exporter import Exporter
from crystalfig.figure.builder import CrystalFigure
from crystalfig.renderers.base import RenderOptions
from crystalfig.renderers.matplotlib_renderer import MatplotlibRenderer
from crystalfig.scene.primitives import Sphere


class FigureGrid:
    """Arrange existing crystal figures in a compact rows-by-columns layout."""

    def __init__(
        self,
        rows: int,
        cols: int,
        panel_labels: bool = True,
        background: str = "white",
        *,
        backend: str = "vector",
        width_mm: float | None = None,
        height_mm: float | None = None,
        panel_gap_mm: float = 3.0,
        caption: str | None = None,
        legend_mode: str = "none",
        font_size_pt: float = 7.0,
    ):
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        if backend not in {"vector", "pyvista", "hybrid"}:
            raise ValueError("backend must be vector, pyvista, or hybrid")
        if width_mm is not None and width_mm <= 0 or height_mm is not None and height_mm <= 0:
            raise ValueError("figure dimensions must be positive")
        if panel_gap_mm < 0:
            raise ValueError("panel_gap_mm must be non-negative")
        if legend_mode not in {"none", "panel"}:
            raise ValueError("legend_mode must be none or panel")
        self.rows = rows
        self.cols = cols
        self.panel_labels = panel_labels
        self.background = background
        self.backend = backend
        self.width_mm = width_mm or 88.9 * cols
        self.height_mm = height_mm or self.width_mm * rows / cols * 0.8
        self.panel_gap_mm = panel_gap_mm
        self.caption = caption
        self.legend_mode = legend_mode
        self.font_size_pt = font_size_pt
        self._panels: dict[tuple[int, int], tuple[CrystalFigure, str | None, str | None]] = {}
        self.last_export_metadata: dict[str, object] = {}

    def add_structure(
        self,
        crystal,
        row: int,
        col: int,
        *,
        style: str | None = None,
        title: str | None = None,
        label: str | None = None,
        **render_options,
    ) -> FigureGrid:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("panel row/col is outside the figure grid")
        key = (row, col)
        if key in self._panels:
            raise ValueError(f"panel {key} already contains a structure")
        is_existing_figure = isinstance(crystal, CrystalFigure)
        figure = crystal if is_existing_figure else CrystalFigure.from_structure(crystal)
        if style is not None:
            figure.render(style=style, **render_options)
        elif not is_existing_figure:
            figure.publication(**render_options)
        elif render_options:
            if figure.config is None:
                figure.publication(style=figure.theme.name, **render_options)
            else:
                display_keys = {
                    "intent": "intent", "region": "region", "centers": "centers",
                    "strategy": "neighbor_strategy", "coordination_shell": "coordination_shell",
                    "show_atoms": "show_atoms", "show_bonds": "show_bonds",
                    "show_unit_cell": "show_unit_cell",
                }
                render_keys = {
                    "backend": "backend", "quality": "quality", "dpi": "dpi",
                    "width": "width_mm", "height": "height_mm", "transparent": "transparent",
                }
                display = replace(figure.config.display, **{
                    target: render_options[key]
                    for key, target in display_keys.items() if key in render_options
                })
                render = replace(figure.config.render, **{
                    target: render_options[key]
                    for key, target in render_keys.items() if key in render_options
                })
                figure.configure(replace(figure.config, display=display, render=render))
        self._panels[key] = (figure, title, label)
        return self

    def save(self, path: str | Path, dpi: int = 300) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        width_in = self.width_mm / 25.4
        height_in = self.height_mm / 25.4
        figure_face = "none" if self.background == "transparent" else self.background
        panel_dpi = []
        with TemporaryDirectory(prefix="crystalfig-grid-") as temp_dir:
            fig, axes = plt.subplots(
                self.rows, self.cols, figsize=(width_in, height_in), dpi=dpi,
                squeeze=False, facecolor=figure_face,
            )
            try:
                nominal_width_mm = self.width_mm / self.cols
                nominal_height_mm = self.height_mm / self.rows
                gap_x = self.panel_gap_mm / max(nominal_width_mm, 1.0)
                gap_y = self.panel_gap_mm / max(nominal_height_mm, 1.0)
                caption_band = 1.5 * self.font_size_pt / (72 * height_in) if self.caption else 0
                fig.subplots_adjust(left=0, right=1, bottom=caption_band, top=1, wspace=gap_x, hspace=gap_y)
                for (row, col), (crystal, title, label) in self._panels.items():
                    ax = axes[row][col]
                    scene = crystal.build_scene()
                    box_position = ax.get_position()
                    panel_width_mm = box_position.width * self.width_mm
                    panel_height_mm = box_position.height * self.height_mm
                    elements = []
                    legend_columns = 1
                    legend_band = 0.06
                    if self.legend_mode == "panel":
                        elements = sorted({
                            primitive.metadata["element"]
                            for primitive in scene.all_primitives()
                            if isinstance(primitive, Sphere) and primitive.metadata.get("element")
                        })
                        panel_width_pt = panel_width_mm * 72 / 25.4
                        legend_columns = max(1, min(len(elements), int(panel_width_pt / (3 * self.font_size_pt))))
                        legend_rows = (len(elements) + legend_columns - 1) // legend_columns
                        legend_band += legend_rows * 1.4 * self.font_size_pt * 25.4 / (72 * panel_height_mm)
                    if self.backend == "vector":
                        options = RenderOptions(
                            width=panel_width_mm, height=panel_height_mm, dpi=dpi,
                            transparent=self.background == "transparent",
                        )
                        MatplotlibRenderer(camera=crystal.camera).draw(ax, scene, crystal.theme, options)
                        ax.set_facecolor(figure_face)
                        if title:
                            ax.set_title(title, fontsize=crystal.theme.title_size)
                        if self.panel_labels:
                            panel_label = label or chr(ord("a") + row * self.cols + col).upper()
                            ax.text(0.01, 0.99, panel_label, transform=ax.transAxes, va="top", ha="left",
                                    fontsize=crystal.theme.title_size, fontweight="bold")
                    else:
                        image_path = Path(temp_dir) / f"panel-{row}-{col}.png"
                        box = (
                            0.03, 0.97,
                            legend_band,
                            0.88 if title or self.panel_labels else 0.97,
                        )
                        viewport_width_mm = (box[1] - box[0]) * panel_width_mm
                        viewport_height_mm = (box[3] - box[2]) * panel_height_mm
                        options = RenderOptions(
                            width=viewport_width_mm, height=viewport_height_mm,
                            dpi=dpi, transparent=self.background == "transparent",
                        )
                        panel_result = Exporter(scene, crystal.theme, crystal.camera).export(
                            str(image_path), backend="pyvista", options=options,
                        )
                        panel_pixels = panel_result.metadata["pixels"]
                        panel_dpi.append(min(
                            panel_pixels[0] / (viewport_width_mm / 25.4),
                            panel_pixels[1] / (viewport_height_mm / 25.4),
                        ))
                        from matplotlib.image import imread

                        image = imread(image_path)
                        box_width, box_height = box[1] - box[0], box[3] - box[2]
                        image_aspect = image.shape[1] / image.shape[0]
                        panel_aspect = panel_width_mm / panel_height_mm
                        image_width = min(box_width, box_height * image_aspect / panel_aspect)
                        image_height = image_width * panel_aspect / image_aspect
                        left = (box[0] + box[1] - image_width) / 2
                        bottom = (box[2] + box[3] - image_height) / 2
                        viewport = (left, left + image_width, bottom, bottom + image_height)
                        ax.imshow(
                            image, interpolation="antialiased", extent=viewport, aspect="auto",
                        )
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.set_axis_off()
                        panel_label = label or chr(ord("a") + row * self.cols + col).upper()
                        if self.panel_labels:
                            ax.text(0.03, 0.98, panel_label, transform=ax.transAxes, va="top", ha="left",
                                    fontsize=7.5, fontweight="bold", color="#20242c")
                        if title:
                            ax.text(0.5, 0.98, title, transform=ax.transAxes, va="top", ha="center",
                                    fontsize=self.font_size_pt, color="#303842")
                    if elements:
                        from matplotlib.lines import Line2D

                        handles = [
                            Line2D([], [], marker="o", linestyle="", markersize=3.5,
                                   color=crystal.palette.hex(element), label=element)
                            for element in elements
                        ]
                        ax.legend(
                            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04),
                            ncol=legend_columns, frameon=False, fontsize=self.font_size_pt,
                            handlelength=0.85, handletextpad=0.25, columnspacing=0.65, borderaxespad=0,
                        )
                    ax.set_facecolor(figure_face)
                for row in range(self.rows):
                    for col in range(self.cols):
                        if (row, col) not in self._panels:
                            axes[row][col].set_axis_off()
                if self.caption:
                    fig.text(0.5, 0.006, self.caption, ha="center", va="bottom", fontsize=self.font_size_pt)
                with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
                    fig.savefig(path, dpi=dpi, transparent=self.background == "transparent", facecolor=figure_face)
            finally:
                plt.close(fig)
        self.last_export_metadata = {
            "backend": self.backend,
            "vector_status": {"vector": "pure", "hybrid": "hybrid", "pyvista": "raster"}[self.backend],
            "format": path.suffix.lstrip(".").lower(),
            "dpi": dpi,
            "embedded_panel_dpi": panel_dpi,
        }
        return path
