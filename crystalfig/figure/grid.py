"""Matplotlib composition for publication panels containing crystal structures."""

from __future__ import annotations

from pathlib import Path

from crystalfig.figure.builder import CrystalFigure


class FigureGrid:
    """Arrange existing crystal figures in a compact rows-by-columns layout."""

    def __init__(self, rows: int, cols: int, panel_labels: bool = True, background: str = "white"):
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        self.rows = rows
        self.cols = cols
        self.panel_labels = panel_labels
        self.background = background
        self._panels: dict[tuple[int, int], tuple[CrystalFigure, str | None, str | None]] = {}

    def add_structure(
        self,
        crystal,
        row: int,
        col: int,
        *,
        style: str = "nature",
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
        figure.render(style=style, **render_options)
        self._panels[key] = (figure, title, label)
        return self

    def save(self, path: str | Path, dpi: int = 300) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        fig, axes = plt.subplots(
            self.rows,
            self.cols,
            figsize=(3.5 * self.cols, 2.8 * self.rows),
            dpi=dpi,
            squeeze=False,
            facecolor="none" if self.background == "transparent" else self.background,
        )
        try:
            for (row, col), (crystal, title, label) in self._panels.items():
                ax = axes[row][col]
                crystal.draw(ax)
                ax.set_facecolor("none" if self.background == "transparent" else self.background)
                if title:
                    ax.set_title(title, fontsize=crystal.theme.title_size)
                panel_label = label or chr(ord("a") + row * self.cols + col).upper()
                if self.panel_labels and panel_label:
                    ax.text(0.01, 0.99, panel_label, transform=ax.transAxes, va="top", ha="left",
                            fontsize=crystal.theme.title_size, fontweight="bold")
            for row in range(self.rows):
                for col in range(self.cols):
                    if (row, col) not in self._panels:
                        axes[row][col].set_axis_off()
            fig.subplots_adjust(wspace=0.04, hspace=0.12)
            fig.savefig(path, dpi=dpi, transparent=self.background == "transparent", bbox_inches="tight")
        finally:
            plt.close(fig)
        return path
