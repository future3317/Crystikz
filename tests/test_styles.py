"""Tests for color palettes and atomic radii."""

import warnings

import pytest

from crystalfig.styles.palette import PUBLICATION, get_palette
from crystalfig.styles.radii import COVALENT, get_radius


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
