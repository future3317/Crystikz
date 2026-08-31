"""Tests for IO adapters."""

import tempfile
from pathlib import Path

from crystalfig.examples.presets import rocksalt_structure
from crystalfig.io.pymatgen_adapter import from_pymatgen, to_pymatgen


class TestPymatgenAdapter:
    def test_round_trip(self):
        struct = rocksalt_structure()
        pmg = to_pymatgen(struct)
        back = from_pymatgen(pmg)
        assert len(back) == len(struct)
        assert set(back.unique_species()) == set(struct.unique_species())

    def test_oxidation_state_round_trip(self):
        """Ordered sites with oxidation states must survive the round-trip."""
        from pymatgen.core import Lattice as PmgLattice
        from pymatgen.core import Species, Structure

        lat = PmgLattice.cubic(4.0)
        pmg = Structure(
            lat,
            [Species("Fe", 2), Species("O", -2)],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        cf = from_pymatgen(pmg)
        assert cf.sites[0].dominant_species == "Fe2+"
        assert cf.sites[1].dominant_species == "O2-"
        assert cf.sites[0].dominant_element == "Fe"
        assert cf.sites[1].dominant_element == "O"

        pmg2 = to_pymatgen(cf)
        assert str(pmg2[0].specie) == "Fe2+"
        assert str(pmg2[1].specie) == "O2-"

    def test_cif_oxidation_state_round_trip(self):
        """Oxidation states must survive CIF write/read."""
        from pymatgen.core import Lattice as PmgLattice
        from pymatgen.core import Species, Structure

        from crystalfig.io.loader import load_structure

        lat = PmgLattice.cubic(4.0)
        pmg = Structure(
            lat,
            [Species("Fe", 2), Species("O", -2)],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        cf = from_pymatgen(pmg)
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "feo.cif"
            to_pymatgen(cf).to(filename=str(cif_path))
            loaded = load_structure(cif_path)
            assert loaded.sites[0].dominant_species in ("Fe2+", "Fe0+")
            assert loaded.sites[1].dominant_species in ("O2-", "O0-")


class TestFileLoader:
    def test_cif_round_trip(self):
        from crystalfig.io.loader import load_structure
        struct = rocksalt_structure()
        pmg = to_pymatgen(struct)
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "rocksalt.cif"
            pmg.to(filename=str(cif_path))
            loaded = load_structure(cif_path)
            assert set(loaded.unique_species()) == {"Na", "Cl"}
