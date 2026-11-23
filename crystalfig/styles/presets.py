"""Named 3D styles assembled from the composable style specifications."""

from __future__ import annotations

import colorsys

from crystalfig.styles.palette import get_palette
from crystalfig.styles.specs import (
    AtomStyle,
    BondStyle,
    CellStyle,
    FigureStyle,
    LightingStyle,
    PolyhedronStyle,
)

PRESETS: dict[str, FigureStyle] = {
    "vesta_ballstick": FigureStyle(
        name="vesta_ballstick",
        atoms=AtomStyle(
            radius_scale=0.34, radius_scale_polyhedron=0.29,
            radius_scale_polyhedron_center=0.20, material="ceramic",
            roughness=0.34, specular=0.42, ambient=0.28,
        ),
        bonds=BondStyle(radius=0.05, color_mode="element_split"),
        polyhedra=PolyhedronStyle(face_opacity=0.0, back_face_opacity=0.0, edge_mode="none"),
        cell=CellStyle(visible=True),
        lighting=LightingStyle(key_intensity=1.05, fill_intensity=0.65, rim_intensity=0.35),
        palette=get_palette("vesta_classic"),
    ),
    "vesta_coordination": FigureStyle(
        name="vesta_coordination",
        atoms=AtomStyle(
            radius_scale=0.28, radius_scale_polyhedron=0.29,
            radius_scale_polyhedron_center=0.20, material="ceramic",
            roughness=0.34, specular=0.42, ambient=0.28,
        ),
        bonds=BondStyle(radius=0.04, color_mode="element_split"),
        polyhedra=PolyhedronStyle(
            color_mode="environment", face_opacity=0.25,
            back_face_opacity=0.08, edge_mode="visible", edge_width=0.8,
        ),
        lighting=LightingStyle(key_intensity=1.05, fill_intensity=0.65, rim_intensity=0.35),
        palette=get_palette("vesta_classic"),
    ),
    "journal_clean": FigureStyle(
        name="journal_clean",
        atoms=AtomStyle(
            radius_scale=0.25, radius_scale_polyhedron=0.23,
            radius_scale_polyhedron_center=0.15, material="ceramic",
            roughness=0.42, specular=0.4,
        ),
        bonds=BondStyle(radius=0.035, color_mode="uniform", color="#7b8793"),
        polyhedra=PolyhedronStyle(
            color_mode="environment", face_opacity=0.23,
            back_face_opacity=0.055, edge_mode="silhouette", edge_width=0.45,
        ),
        cell=CellStyle(visible=True, edge_width=0.5),
        lighting=LightingStyle(key_intensity=0.85, fill_intensity=0.5, rim_intensity=0.28),
    ),
    "journal_vivid": FigureStyle(
        name="journal_vivid",
        atoms=AtomStyle(
            radius_scale=0.25, radius_scale_polyhedron=0.25,
            radius_scale_polyhedron_center=0.17, material="ceramic",
            roughness=0.34, specular=0.5,
        ),
        bonds=BondStyle(radius=0.04, color_mode="element_split"),
        polyhedra=PolyhedronStyle(
            color_mode="environment", face_opacity=0.32,
            back_face_opacity=0.085, edge_mode="visible", edge_width=0.5,
            edge_opacity=0.88,
        ),
        cell=CellStyle(visible=True, edge_width=0.5),
        lighting=LightingStyle(key_intensity=1.0, fill_intensity=0.5, rim_intensity=0.38),
    ),
    "layered_crystal": FigureStyle(
        name="layered_crystal",
        atoms=AtomStyle(radius_scale=0.25, material="ceramic", roughness=0.32),
        bonds=BondStyle(radius=0.04, color_mode="element_split"),
        polyhedra=PolyhedronStyle(face_opacity=0.22, back_face_opacity=0.06),
        cell=CellStyle(visible=True, edge_width=0.5),
    ),
}


def get_style(name: str) -> FigureStyle:
    """Return a named style with an independent palette copy."""
    try:
        style = PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown figure style {name!r}. Available: {list(PRESETS)}") from None
    return FigureStyle(
        name=style.name,
        atoms=style.atoms,
        bonds=style.bonds,
        polyhedra=style.polyhedra,
        cell=style.cell,
        lighting=style.lighting,
        palette=style.palette.copy(),
        background=style.background,
    )


def get_coordination_color(coordination_number: int, environment_key=None) -> str:
    """Return a fixed CN color or a deterministic variation for a full environment key."""
    colors = {
        3: "#7B6FD0", 4: "#D88932", 5: "#4F91C8", 6: "#56A879",
        7: "#C85D70", 8: "#8B75A8", 9: "#A78242", 10: "#4E9C9A",
        11: "#B65E9B", 12: "#71834A",
    }
    if coordination_number not in colors:
        return "#808080"
    if environment_key is None:
        return colors[coordination_number]

    # A center's atomic number rotates its hue through a golden angle. Ligand
    # atomic numbers then distinguish environments around the same center.
    bases = {3: 275, 4: 30, 5: 210, 6: 135, 7: 350, 8: 270, 9: 40, 10: 180, 11: 320, 12: 85}
    element_order = tuple(get_palette("jmol").element_colors)

    def atomic_number(symbol):
        try:
            return element_order.index(symbol) + 1
        except ValueError:
            return 0

    center_element, _, ligand_species = environment_key
    center_number = atomic_number(center_element)
    ligand_numbers = sorted(atomic_number(symbol) for symbol in ligand_species)
    ligand_statistic = sum((index + 1) * number for index, number in enumerate(ligand_numbers))
    hue = (bases[coordination_number] + center_number * 137.507764 + ligand_statistic * 4.13) % 360
    saturation = 0.78 + (center_number % 14) / 100
    value = 0.90 + (ligand_statistic % 8) / 100
    rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    return "#{:02X}{:02X}{:02X}".format(*(round(channel * 255) for channel in rgb))
