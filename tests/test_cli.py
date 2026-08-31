"""Tests for CLI commands."""

import tempfile
from pathlib import Path

from crystalfig.cli.main import main
from crystalfig.examples.presets import rocksalt_structure
from crystalfig.io.pymatgen_adapter import to_pymatgen


class TestCLI:
    def test_doctor(self, capsys):
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "crystalfig doctor" in out

    def test_inspect(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "rocksalt.cif"
            pmg = to_pymatgen(rocksalt_structure())
            pmg.to(filename=str(cif_path))
            assert main(["inspect", str(cif_path)]) == 0
            out = capsys.readouterr().out
            assert "Formula" in out

    def test_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "rocksalt.cif"
            out_path = Path(tmpdir) / "out.svg"
            pmg = to_pymatgen(rocksalt_structure())
            pmg.to(filename=str(cif_path))
            assert main(["render", str(cif_path), "-o", str(out_path), "--fmt", "svg", "--bonds", "covalent"]) == 0
            assert out_path.exists()
