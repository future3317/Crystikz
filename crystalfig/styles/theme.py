"""Publication themes controlling typography, sizing, and colors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crystalfig.styles.palette import ColorPalette, get_palette


@dataclass
class FigureTheme:
    """A publication figure theme."""

    name: str = "publication"
    palette: ColorPalette = field(default_factory=lambda: get_palette("publication"))
    figure_width: float = 89.0  # mm
    figure_height: float | None = None  # mm; None means auto
    font_size: float = 7.0  # pt
    label_size: float = 6.0  # pt
    title_size: float = 8.0  # pt
    background: str = "white"  # white, transparent, or color
    atom_style: str = "shaded"  # shaded, flat, space_filling
    atom_material: str = "matte"  # matte, glossy, flat
    atom_radius_scale: float = 0.28  # multiplier for covalent radii in ball-stick mode
    atom_radius_scale_polyhedron: float = 0.24  # vertex atoms in polyhedron mode
    atom_radius_scale_polyhedron_center: float = 0.11  # central cation in polyhedron mode
    bond_width: float = 0.06  # in angstroms for 3D backends
    bond_color: str = "gray"
    bond_color_mode: str = "split_soft"  # "uniform", "split", or "split_soft"
    cell_edge_width: float = 0.55
    cell_front_style: str = "solid"
    cell_back_style: str = "dashed"
    polyhedron_opacity: float = 0.22
    polyhedron_edge_width: float = 0.5
    polyhedron_face_opacity: float = 0.55
    polyhedron_back_opacity: float = 0.18
    polyhedron_edge_opacity: float = 0.55
    polyhedron_edge_mode: str = "silhouette"
    polyhedron_back_edges: str = "faint"  # hide, faint, dashed
    cell_style: str = "light"  # none, light, full
    camera_preset: str = "auto"
    render_quality: str = "standard"  # preview, standard, publication
    vector_scale: float = 1.0
    vector_head_size: float = 0.15
    show_legend: bool = False
    show_panel_labels: bool = True
    dpi: int = 300
    transparent: bool = False
    use_tex: bool = False
    custom_preamble: str = ""

    @classmethod
    def from_preset(cls, name: str) -> FigureTheme:
        pub = cls(name="publication", palette=get_palette("publication"))
        presets = {
            "publication": pub,
            "publication_muted": cls(name="publication_muted", palette=get_palette("muted")),
            "publication_structure": pub,
            "publication_polyhedra": cls(
                name="publication_polyhedra",
                palette=get_palette("publication"),
                atom_radius_scale=0.26,
                atom_radius_scale_polyhedron=0.24,
                atom_radius_scale_polyhedron_center=0.10,
                polyhedron_opacity=0.20,
                polyhedron_edge_width=0.35,
                bond_width=0.04,
                cell_edge_width=0.50,
            ),
            "nature": cls(
                name="nature",
                palette=get_palette("publication"),
                atom_radius_scale=0.25,
                atom_radius_scale_polyhedron=0.22,
                atom_radius_scale_polyhedron_center=0.15,
                bond_color_mode="uniform",
                bond_color="#7b8793",
                polyhedron_opacity=0.40,
                polyhedron_face_opacity=0.58,
                polyhedron_back_opacity=0.04,
                polyhedron_edge_width=0.45,
                polyhedron_edge_opacity=0.82,
                polyhedron_edge_mode="silhouette",
                polyhedron_back_edges="hide",
                cell_style="full",
                camera_preset="auto",
                background="white",
            ),
            "journal_clean": cls(
                name="journal_clean",
                palette=get_palette("publication"),
                atom_material="matte",
                atom_radius_scale=0.25,
                atom_radius_scale_polyhedron=0.22,
                atom_radius_scale_polyhedron_center=0.12,
                bond_color_mode="uniform",
                bond_color="#7b8793",
                bond_width=0.035,
                polyhedron_opacity=0.56,
                polyhedron_face_opacity=0.66,
                polyhedron_back_opacity=0.14,
                polyhedron_edge_width=0.45,
                polyhedron_edge_opacity=0.78,
                polyhedron_edge_mode="silhouette",
                polyhedron_back_edges="faint",
                cell_style="light",
                camera_preset="auto",
                background="white",
            ),
            "journal_vivid": cls(
                name="journal_vivid",
                palette=get_palette("publication"),
                atom_material="matte",
                atom_radius_scale=0.25,
                atom_radius_scale_polyhedron=0.22,
                atom_radius_scale_polyhedron_center=0.12,
                bond_color_mode="uniform",
                bond_color="#6f7a84",
                bond_width=0.04,
                polyhedron_opacity=0.64,
                polyhedron_face_opacity=0.70,
                polyhedron_back_opacity=0.16,
                polyhedron_edge_width=0.50,
                polyhedron_edge_opacity=0.88,
                polyhedron_edge_mode="visible",
                polyhedron_back_edges="faint",
                cell_style="light",
                camera_preset="auto",
                background="white",
            ),
            "publication_overview": cls(
                name="publication_overview",
                palette=get_palette("publication"),
                atom_radius_scale=0.20,
                bond_width=0.03,
                cell_edge_width=0.45,
            ),
            "ballstick_clean": cls(
                name="ballstick_clean",
                palette=get_palette("publication"),
                atom_material="matte",
                atom_radius_scale=0.28,
                bond_width=0.045,
                bond_color_mode="uniform",
                bond_color="#7b8793",
                cell_style="light",
                camera_preset="auto",
            ),
            "polyhedra_clean": cls(
                name="polyhedra_clean",
                palette=get_palette("publication"),
                atom_radius_scale=0.24,
                atom_radius_scale_polyhedron=0.22,
                atom_radius_scale_polyhedron_center=0.08,
                polyhedron_opacity=0.35,
                polyhedron_face_opacity=0.52,
                polyhedron_back_opacity=0.12,
                polyhedron_edge_width=0.45,
                polyhedron_edge_mode="visible",
                cell_style="light",
            ),
            "layered_clean": cls(
                name="layered_clean",
                palette=get_palette("publication"),
                atom_radius_scale=0.25,
                bond_width=0.04,
                cell_style="light",
                camera_preset="layer_oblique",
            ),
            "overview_minimal": cls(
                name="overview_minimal",
                palette=get_palette("publication"),
                atom_radius_scale=0.20,
                bond_width=0.03,
                cell_style="light",
                show_legend=False,
            ),
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
            "atom_material": self.atom_material,
            "atom_radius_scale": self.atom_radius_scale,
            "atom_radius_scale_polyhedron": self.atom_radius_scale_polyhedron,
            "atom_radius_scale_polyhedron_center": self.atom_radius_scale_polyhedron_center,
            "bond_width": self.bond_width,
            "bond_color": self.bond_color,
            "bond_color_mode": self.bond_color_mode,
            "cell_edge_width": self.cell_edge_width,
            "cell_front_style": self.cell_front_style,
            "cell_back_style": self.cell_back_style,
            "polyhedron_opacity": self.polyhedron_opacity,
            "polyhedron_edge_width": self.polyhedron_edge_width,
            "polyhedron_face_opacity": self.polyhedron_face_opacity,
            "polyhedron_back_opacity": self.polyhedron_back_opacity,
            "polyhedron_edge_opacity": self.polyhedron_edge_opacity,
            "polyhedron_edge_mode": self.polyhedron_edge_mode,
            "polyhedron_back_edges": self.polyhedron_back_edges,
            "cell_style": self.cell_style,
            "camera_preset": self.camera_preset,
            "render_quality": self.render_quality,
            "vector_scale": self.vector_scale,
            "vector_head_size": self.vector_head_size,
            "show_legend": self.show_legend,
            "show_panel_labels": self.show_panel_labels,
            "dpi": self.dpi,
            "transparent": self.transparent,
            "use_tex": self.use_tex,
        }


JournalTheme = FigureTheme  # alias
