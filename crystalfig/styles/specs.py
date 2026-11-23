"""Composable visual settings for three-dimensional crystal figures."""

from __future__ import annotations

from dataclasses import dataclass, field

from crystalfig.styles.palette import ColorPalette, get_palette


@dataclass(frozen=True)
class AtomStyle:
    radius_scale: float = 0.28
    radius_scale_polyhedron: float = 0.24
    radius_scale_polyhedron_center: float = 0.11
    material: str = "ceramic"
    sphere_resolution: int = 64
    roughness: float = 0.34
    specular: float = 0.25
    ambient: float = 0.18


@dataclass(frozen=True)
class BondStyle:
    radius: float = 0.045
    color_mode: str = "element_split"
    color: str = "#7b8793"
    clip_to_atoms: bool = True
    smooth: bool = True


@dataclass(frozen=True)
class PolyhedronStyle:
    color_mode: str = "environment"
    face_opacity: float = 0.25
    back_face_opacity: float = 0.07
    edge_mode: str = "visible"
    edge_width: float = 0.8
    edge_opacity: float = 0.7


@dataclass(frozen=True)
class CellStyle:
    visible: bool = True
    edge_color: str = "#AAB2BA"
    edge_width: float = 0.6
    front_style: str = "solid"
    back_style: str = "dashed"


@dataclass(frozen=True)
class LightingStyle:
    key_intensity: float = 0.8
    fill_intensity: float = 0.35
    rim_intensity: float = 0.25
    ssao: bool = True


@dataclass(frozen=True)
class FigureStyle:
    """Renderer-independent 3D appearance settings."""

    name: str = "journal_clean"
    atoms: AtomStyle = field(default_factory=AtomStyle)
    bonds: BondStyle = field(default_factory=BondStyle)
    polyhedra: PolyhedronStyle = field(default_factory=PolyhedronStyle)
    cell: CellStyle = field(default_factory=CellStyle)
    lighting: LightingStyle = field(default_factory=LightingStyle)
    palette: ColorPalette = field(default_factory=lambda: get_palette("publication").copy())
    background: str = "white"

    def to_legacy_theme(self):
        """Map supported values to FigureTheme without changing effective face alpha."""
        from crystalfig.styles.theme import FigureTheme

        theme = FigureTheme.from_preset("publication")
        theme.figure_style = self
        theme.name = self.name
        theme.palette = self.palette.copy()
        theme.background = self.background
        theme.atom_radius_scale = self.atoms.radius_scale
        theme.atom_radius_scale_polyhedron = self.atoms.radius_scale_polyhedron
        theme.atom_radius_scale_polyhedron_center = self.atoms.radius_scale_polyhedron_center
        theme.atom_material = {"ceramic": "matte"}.get(self.atoms.material, self.atoms.material)
        theme.bond_width = self.bonds.radius
        theme.bond_color = self.bonds.color
        theme.bond_color_mode = {
            "element_split": "split",
            "split": "split",
            "split_soft": "split_soft",
            "uniform": "uniform",
        }.get(self.bonds.color_mode, self.bonds.color_mode)
        theme.cell_style = "light" if self.cell.visible else "none"
        theme.cell_edge_color = self.cell.edge_color
        theme.cell_edge_width = self.cell.edge_width
        theme.cell_front_style = self.cell.front_style
        theme.cell_back_style = self.cell.back_style
        # The legacy renderer multiplies these two values. Keep face alpha exact.
        theme.polyhedron_opacity = 1.0
        theme.polyhedron_face_opacity = self.polyhedra.face_opacity
        theme.polyhedron_back_opacity = self.polyhedra.back_face_opacity
        theme.polyhedron_edge_width = self.polyhedra.edge_width
        theme.polyhedron_edge_opacity = self.polyhedra.edge_opacity
        theme.polyhedron_edge_mode = self.polyhedra.edge_mode
        return theme
