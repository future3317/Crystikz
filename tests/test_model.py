"""Tests for the canonical internal model."""

import numpy as np
import pytest

from crystalfig.model.lattice import Lattice
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure


class TestLattice:
    def test_cubic_lengths_and_angles(self):
        lat = Lattice.cubic(4.0)
        assert lat.lengths == pytest.approx((4.0, 4.0, 4.0))
        assert lat.angles == pytest.approx((90.0, 90.0, 90.0))

    def test_frac_cart_round_trip(self):
        lat = Lattice.from_parameters(3.0, 4.0, 5.0, 90.0, 90.0, 90.0)
        frac = np.array([0.25, 0.5, 0.75])
        cart = lat.frac_to_cart(frac)
        frac_back = lat.cart_to_frac(cart)
        assert frac_back == pytest.approx(frac)

    def test_hexagonal_gamma(self):
        lat = Lattice.from_parameters(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
        assert lat.angles[2] == pytest.approx(120.0)
        # Verify a = b
        assert lat.lengths[0] == pytest.approx(lat.lengths[1])

    def test_reciprocal_convention(self):
        lat = Lattice.cubic(2.0 * np.pi)
        rec = lat.reciprocal_matrix
        # For cubic lattice with a=2π, reciprocal vector a* should be (1,0,0)
        assert rec[0] == pytest.approx([1.0, 0.0, 0.0])
        assert rec[1] == pytest.approx([0.0, 1.0, 0.0])
        assert rec[2] == pytest.approx([0.0, 0.0, 1.0])


class TestSite:
    def test_dominant_species(self):
        site = Site(frac_coords=[0, 0, 0], species="Ti")
        assert site.dominant_species == "Ti"

    def test_disordered_normalization(self):
        site = Site(frac_coords=[0, 0, 0], species={"Fe": 0.5, "Ni": 0.5})
        assert site.dominant_species in ("Fe", "Ni")
        assert sum(site.occupancy.values()) == pytest.approx(1.0)


class TestCrystalStructure:
    def test_formula(self):
        lat = Lattice.cubic(4.0)
        sites = [Site([0, 0, 0], "Na"), Site([0.5, 0.5, 0.5], "Cl")]
        struct = CrystalStructure(lat, sites)
        assert struct.formula == "ClNa" or struct.formula == "NaCl"
        assert struct.num_sites == 2

    def test_supercell(self):
        lat = Lattice.cubic(4.0)
        sites = [Site([0, 0, 0], "Na")]
        struct = CrystalStructure(lat, sites)
        super_struct = struct.make_supercell(2)
        assert len(super_struct) == 8
        assert super_struct.volume == pytest.approx(8 * struct.volume)

    def test_unique_species(self):
        lat = Lattice.cubic(4.0)
        sites = [Site([0, 0, 0], "Na"), Site([0.5, 0.5, 0.5], "Cl")]
        struct = CrystalStructure(lat, sites)
        assert set(struct.unique_species()) == {"Na", "Cl"}
