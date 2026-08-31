"""Tests for neighbor strategies."""


from crystalfig.examples.presets import rocksalt_structure
from crystalfig.neighbors.strategies import CovalentRadiiStrategy, CutoffStrategy


class TestCutoffStrategy:
    def test_rocksalt_bonds(self):
        struct = rocksalt_structure()
        strategy = CutoffStrategy(cutoff=3.0)
        bonds = strategy.get_bonds(struct)
        # Each Na should have 6 Cl neighbors in rocksalt
        na_bonds = [b for b in bonds if struct.sites[b.i].dominant_species == "Na"]
        assert len(na_bonds) >= 6


class TestCovalentRadiiStrategy:
    def test_rocksalt_bonds(self):
        struct = rocksalt_structure()
        strategy = CovalentRadiiStrategy(tolerance=0.4)
        bonds = strategy.get_bonds(struct)
        # Each Na should have 6 Cl neighbors
        na_bonds = [b for b in bonds if struct.sites[b.i].dominant_species == "Na"]
        assert len(na_bonds) >= 6
