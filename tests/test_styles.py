"""Tests for color palettes and atomic radii."""

import warnings

import pytest

from crystalfig.figure.config import CameraConfig, FigureConfig, RenderConfig
from crystalfig.styles.palette import PUBLICATION, get_palette
from crystalfig.styles.presets import PRESETS, get_coordination_color, get_style
from crystalfig.styles.radii import COVALENT, get_radius
from crystalfig.styles.specs import (
    AtomStyle,
    BondStyle,
    CellStyle,
    FigureStyle,
    LightingStyle,
    PolyhedronStyle,
)
from crystalfig.styles.theme import FigureTheme


def test_journal_themes_separate_atom_and_coordination_visual_roles():
    clean = FigureTheme.from_preset("journal_clean")
    vivid = FigureTheme.from_preset("journal_vivid")
    assert clean.polyhedron_opacity * clean.polyhedron_face_opacity < (
        vivid.polyhedron_opacity * vivid.polyhedron_face_opacity
    )
    vivid_front_alpha = vivid.polyhedron_opacity * vivid.polyhedron_face_opacity
    assert 0.35 <= vivid_front_alpha <= 0.55
    assert vivid.polyhedron_back_opacity < vivid.polyhedron_face_opacity
    assert clean.name == "journal_clean"
    assert vivid.name == "journal_vivid"


def test_composable_figure_styles_and_requested_presets():
    assert set(PRESETS) == {
        "vesta_ballstick", "vesta_coordination", "journal_clean", "journal_vivid", "layered_crystal"
    }
    style = get_style("journal_clean")
    assert isinstance(style, FigureStyle)
    assert isinstance(style.atoms, AtomStyle)
    assert isinstance(style.bonds, BondStyle)
    assert isinstance(style.polyhedra, PolyhedronStyle)
    assert isinstance(style.cell, CellStyle)
    assert isinstance(style.lighting, LightingStyle)
    assert style is not get_style("journal_clean")
    assert style.palette is not get_style("journal_clean").palette
    assert get_style("vesta_coordination").palette.name == "vesta_classic"
    assert get_style("journal_clean").atoms.radius_scale_polyhedron_center == pytest.approx(0.15)
    assert get_style("journal_vivid").atoms.radius_scale_polyhedron_center == pytest.approx(0.17)
    first, second = FigureStyle(), FigureStyle()
    assert first.palette is not second.palette
    assert first.palette.element_colors is not PUBLICATION.element_colors


def test_figure_style_maps_to_legacy_theme_with_effective_alpha():
    style = FigureStyle(
        atoms=AtomStyle(radius_scale_polyhedron=0.23, radius_scale_polyhedron_center=0.15),
        polyhedra=PolyhedronStyle(face_opacity=0.42, back_face_opacity=0.13),
        cell=CellStyle(visible=False),
    )
    theme = style.to_legacy_theme()
    assert isinstance(theme, FigureTheme)
    assert theme.polyhedron_opacity * theme.polyhedron_face_opacity == pytest.approx(0.42)
    assert theme.polyhedron_back_opacity == pytest.approx(0.13)
    assert theme.cell_style == "none"
    assert theme.atom_radius_scale_polyhedron == pytest.approx(0.23)
    assert theme.atom_radius_scale_polyhedron_center == pytest.approx(0.15)


def test_coordination_colors_are_stable_without_observed_categories():
    colors = [get_coordination_color(n) for n in (4, 5, 6, 8)]
    assert len(set(colors)) == 4
    assert get_coordination_color(6) == "#56A879"
    assert get_coordination_color(13) == "#808080"
    first_environment = get_coordination_color(6, ("Li", 6, ("O",) * 6))
    second_environment = get_coordination_color(6, ("Fe", 6, ("O",) * 6))
    assert first_environment != second_environment
    assert get_coordination_color(6, ("Li", 6, ("O",) * 6)) == first_environment
    assert get_coordination_color(4, ("Zn", 4, ("O",))) != get_coordination_color(
        4, ("O", 4, ("Zn",))
    )


def test_figure_config_defaults_and_boundary_values():
    config = FigureConfig()
    assert config.style == "journal_clean"
    assert config.camera.projection == "orthographic"
    assert config.render.backend == "pyvista"
    assert config.render.dpi == 600
    with pytest.raises(ValueError, match="padding"):
        CameraConfig(padding=1.0)
    with pytest.raises(ValueError, match="quality"):
        RenderConfig(quality="ultra")


class TestColorPalette:
    def test_publication_covers_h_through_bi(self):
        """The publication palette must define a color for every element Z=1-83."""
        elements = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi",
        ]
        for el in elements:
            assert el in PUBLICATION.element_colors, f"Missing color for {el}"
            rgb = PUBLICATION.rgb(el)
            assert len(rgb) == 3
            assert all(0 <= c <= 255 for c in rgb)

    def test_palette_copy_is_deep(self):
        """copy() must produce an independent palette."""
        palette = get_palette("publication")
        copied = palette.copy()
        assert copied is not palette
        assert copied.element_colors is not palette.element_colors
        assert copied.accents is not palette.accents
        copied.element_colors["C"] = (0, 0, 0)
        assert palette.element_colors["C"] != (0, 0, 0)

    def test_published_palette_presets_have_element_colors(self):
        for name in ("jmol", "vesta_classic", "publication", "colorblind_safe"):
            palette = get_palette(name)
            assert palette.element_colors
            assert palette.hex("O") != "#808080"
        assert FigureTheme.from_preset("colorblind_safe").palette.hex("O") != "#808080"
        assert get_palette("vesta_classic").hex("O") != get_palette("jmol").hex("O")


class TestAtomicRadii:
    def test_covalent_covers_h_through_bi(self):
        """COVALENT radii must be defined for every element Z=1-83."""
        elements = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi",
        ]
        for el in elements:
            assert el in COVALENT.radii, f"Missing covalent radius for {el}"
            assert COVALENT.get(el) > 0

    def test_unknown_radius_default_and_warning(self):
        """Unknown elements fall back to 1.5 Å and emit a one-time warning."""
        # Reset the warning tracker so this test is deterministic.
        from crystalfig.styles import radii
        radii._MISSING_RADIUS_WARNED.discard("Xx")

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            value = get_radius("Xx", "covalent")
            assert value == pytest.approx(1.5)
            assert len(recorded) == 1
            assert "No covalent radius known for element 'Xx'" in str(recorded[0].message)

        # Second lookup should not warn again.
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            value2 = get_radius("Xx", "covalent")
            assert value2 == pytest.approx(1.5)
            assert len(recorded) == 0
