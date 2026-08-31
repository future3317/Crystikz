"""Publication themes controlling typography, sizing, and colors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crystalfig.styles.palette import ColorPalette, get_palette


@dataclass
class FigureTheme:
    """A publication figure theme."""

    name: str = "publication_muted"
    palette: ColorPalette = field(default_factory=lambda: get_palette("muted"))
    figure_width: float = 89.0  # mm
    figure_height: float | None = None  # mm; None means auto
    font_size: float = 7.0  # pt
    label_size: float = 6.0  # pt
    title_size: float = 8.0  # pt
    background: str = "white"  # white, transparent, or color
    atom_style: str = "shaded"  # shaded, flat, space_filling
    bond_width: float = 0.08  # in angstroms for 3D backends
    bond_color: str = "gray"
    cell_edge_width: float = 1.0
    cell_front_style: str = "solid"
    cell_back_style: str = "dashed"
    polyhedron_opacity: float = 0.25
    polyhedron_edge_width: float = 0.5
    vector_scale: float = 1.0
    vector_head_size: float = 0.15
    show_legend: bool = True
    show_panel_labels: bool = True
    dpi: int = 300
    transparent: bool = False
    use_tex: bool = False
    custom_preamble: str = ""

    @classmethod
    def from_preset(cls, name: str) -> FigureTheme:
        presets = {
            "publication_muted": cls(name="publication_muted", palette=get_palette("muted")),
            "one_column": cls(name="one_column", figure_width=89.0),
            "two_column": cls(name="two_column", figure_width=183.0),
            "square": cls(name="square", figure_width=120.0, figure_height=120.0),
            "wide": cls(name="wide", figure_width=240.0),
            "colorblind_safe": cls(name="colorblind_safe", palette=get_palette("okabe_ito")),
            "monochrome": cls(name="monochrome", palette=get_palette("monochrome")),
        }
        if name not in presets:
            raise ValueError(f"Unknown theme preset '{name}'")
        return presets[name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "palette": self.palette.name,
            "figure_width": self.figure_width,
            "figure_height": self.figure_height,
            "font_size": self.font_size,
            "label_size": self.label_size,
            "title_size": self.title_size,
            "background": self.background,
            "atom_style": self.atom_style,
            "bond_width": self.bond_width,
            "bond_color": self.bond_color,
            "cell_edge_width": self.cell_edge_width,
            "cell_front_style": self.cell_front_style,
            "cell_back_style": self.cell_back_style,
            "polyhedron_opacity": self.polyhedron_opacity,
            "polyhedron_edge_width": self.polyhedron_edge_width,
            "vector_scale": self.vector_scale,
            "vector_head_size": self.vector_head_size,
            "show_legend": self.show_legend,
            "show_panel_labels": self.show_panel_labels,
            "dpi": self.dpi,
            "transparent": self.transparent,
            "use_tex": self.use_tex,
        }


JournalTheme = FigureTheme  # alias
