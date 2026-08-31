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
