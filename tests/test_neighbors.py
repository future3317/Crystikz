"""Tests for neighbor strategies."""

import numpy as np
import pytest

from crystalfig.examples.presets import perovskite_structure, rocksalt_structure
from crystalfig.neighbors.strategies import CovalentRadiiStrategy, CutoffStrategy


class TestCutoffStrategy:
    def test_rocksalt_bonds(self):
        struct = rocksalt_structure()
        strategy = CutoffStrategy(cutoff=3.0)
        bonds = strategy.get_bonds(struct)
        # Each Na should have 6 Cl neighbors in rocksalt
        na_bonds = [b for b in bonds if struct.sites[b.i].dominant_species == "Na"]
        assert len(na_bonds) == 6

    def test_perovskite_ti_o_bonds(self):
        """Ti in the primitive perovskite cell has 6 nearest O neighbours."""
        struct = perovskite_structure()
        bonds = CutoffStrategy(cutoff=2.5).get_bonds(struct)
        ti_bonds = [b for b in bonds if struct.sites[b.i].dominant_species == "Ti"]
        assert len(ti_bonds) == 6
        for bond in ti_bonds:
            assert struct.sites[bond.j].dominant_species == "O"
            assert bond.distance < 2.5


class TestCovalentRadiiStrategy:
    def test_rocksalt_bonds(self):
        struct = rocksalt_structure()
        strategy = CovalentRadiiStrategy(tolerance=0.4)
        bonds = strategy.get_bonds(struct)
        # Each Na should have 6 Cl neighbors
        na_bonds = [b for b in bonds if struct.sites[b.i].dominant_species == "Na"]
        assert len(na_bonds) == 6


class TestPBCBondGeometry:
    def test_bond_endpoints_cross_boundary(self):
        """A bond across a periodic boundary must have a non-zero jimage."""
        struct = rocksalt_structure()
        bonds = CutoffStrategy(cutoff=3.0).get_bonds(struct)
        # Rocksalt primitive: Na at (0,0,0), Cl at (1/2,1/2,1/2). The Cl
        # nearest images to Na include both the in-cell copy and periodic images.
        assert any(bond.jimage != (0, 0, 0) for bond in bonds)

    def test_image_coordinate_consistency(self):
        """Cartesian endpoint reconstructed from jimage must match reported distance."""
        struct = rocksalt_structure()
        bonds = CovalentRadiiStrategy(tolerance=0.4).get_bonds(struct)
        for bond in bonds:
            pos_i = struct.sites[bond.i].cart_coords(struct.lattice)
            pos_j = struct.sites[bond.j].cart_coords(struct.lattice)
            image_j = pos_j + struct.lattice.frac_to_cart(np.array(bond.jimage, dtype=float))
            reconstructed = np.linalg.norm(image_j - pos_i)
            assert reconstructed == pytest.approx(bond.distance, abs=1e-6)
