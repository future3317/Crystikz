"""Matplotlib composition for publication panels containing crystal structures."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from crystalfig.figure.builder import CrystalFigure


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
        self.last_export_metadata: dict[str, str] = {}

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
        figure = crystal if isinstance(crystal, CrystalFigure) else CrystalFigure.from_structure(crystal)
        if style is None:
            figure.publication(**render_options)
        else:
            figure.render(style=style, **render_options)
        self._panels[key] = (figure, title, label)
        return self

    def save(self, path: str | Path, dpi: int = 300) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        from matplotlib.image import imread

        width_in = self.width_mm / 25.4
        height_in = self.height_mm / 25.4
        figure_face = "none" if self.background == "transparent" else self.background
        with TemporaryDirectory(prefix="crystalfig-grid-") as temp_dir:
            fig, axes = plt.subplots(
                self.rows, self.cols, figsize=(width_in, height_in), dpi=dpi,
                squeeze=False, facecolor=figure_face,
            )
            try:
                panel_width_mm = self.width_mm / self.cols
                panel_height_mm = self.height_mm / self.rows
                for (row, col), (crystal, title, label) in self._panels.items():
                    ax = axes[row][col]
                    scene = crystal.build_scene() if self.legend_mode == "panel" else None
                    if self.backend == "vector":
                        crystal.draw(ax)
                        ax.set_facecolor(figure_face)
                        if title:
                            ax.set_title(title, fontsize=crystal.theme.title_size)
                        if self.panel_labels:
                            panel_label = label or chr(ord("a") + row * self.cols + col).upper()
                            ax.text(0.01, 0.99, panel_label, transform=ax.transAxes, va="top", ha="left",
                                    fontsize=crystal.theme.title_size, fontweight="bold")
                    else:
                        image_path = Path(temp_dir) / f"panel-{row}-{col}.png"
                        crystal.export(
                            str(image_path), backend="pyvista", width=panel_width_mm,
                            height=panel_height_mm, dpi=dpi,
                            transparent=self.background == "transparent",
                        )
                        ax.imshow(
                            imread(image_path), interpolation="antialiased",
                            extent=(0.03, 0.97, 0.12, 0.90), aspect="auto",
                        )
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.set_axis_off()
                        panel_label = label or chr(ord("a") + row * self.cols + col).upper()
                        if self.panel_labels:
                            ax.text(0.025, 0.975, panel_label, transform=ax.transAxes, va="top", ha="left",
                                    fontsize=7.5, fontweight="bold", color="#20242c")
                        if title:
                            ax.text(0.5, 0.975, title, transform=ax.transAxes, va="top", ha="center",
                                    fontsize=self.font_size_pt, color="#303842")
                    if self.legend_mode == "panel":
                        from matplotlib.lines import Line2D

                        elements = sorted({
                            primitive.metadata.get("element")
                            for primitive in scene.all_primitives()
                            if primitive.__class__.__name__ == "Sphere"
                            and primitive.metadata.get("element")
                        })
                        handles = [
                            Line2D([], [], marker="o", linestyle="", markersize=3.5,
                                   color=crystal.palette.hex(element), label=element)
                            for element in elements
                        ]
                        ax.legend(
                            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                            ncol=min(len(handles), 4), frameon=False, fontsize=self.font_size_pt,
                            handletextpad=0.25, columnspacing=0.65, borderaxespad=0,
                        )
                    ax.set_facecolor(figure_face)
                for row in range(self.rows):
                    for col in range(self.cols):
                        if (row, col) not in self._panels:
                            axes[row][col].set_axis_off()
                gap_x = self.panel_gap_mm / max(panel_width_mm, 1.0)
                gap_y = self.panel_gap_mm / max(panel_height_mm, 1.0)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=gap_x, hspace=gap_y)
                if self.caption:
                    fig.text(0.5, 0.006, self.caption, ha="center", va="bottom", fontsize=self.font_size_pt)
                fig.savefig(path, dpi=dpi, transparent=self.background == "transparent", facecolor=figure_face)
            finally:
                plt.close(fig)
        self.last_export_metadata = {
            "backend": self.backend,
            "vector_status": {"vector": "pure", "hybrid": "hybrid", "pyvista": "raster"}[self.backend],
            "format": path.suffix.lstrip(".").lower(),
        }
        return path
