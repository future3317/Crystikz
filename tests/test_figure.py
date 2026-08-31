"""Tests for CrystalFigure fluent API."""

import tempfile
from pathlib import Path

from crystalfig.examples.presets import perovskite_structure, rocksalt_structure
from crystalfig.figure.builder import CrystalFigure


class TestCrystalFigure:
    def test_quick_export_pdf(self):
        fig = CrystalFigure(rocksalt_structure()).quick()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "rocksalt.pdf"
            result = fig.export(str(out))
            assert Path(result.path).exists()

    def test_view_direction(self):
        fig = CrystalFigure(rocksalt_structure()).view([1, 1, 0])
        assert fig.camera is not None

    def test_add_polyhedra(self):
        fig = CrystalFigure(perovskite_structure()).add_bonds("covalent").add_polyhedra("Ti")
        scene = fig.build_scene()
        polyhedra = [p for p in scene.all_primitives() if p.__class__.__name__ == "Polyhedron"]
        assert len(polyhedra) > 0

    def test_supercell(self):
        fig = CrystalFigure(rocksalt_structure()).supercell((2, 1, 1)).quick()
        scene = fig.build_scene()
        atoms = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
        assert len(atoms) == 16  # 8 atoms * 2

    def test_save_recipe(self):
        fig = CrystalFigure(rocksalt_structure()).quick()
        with tempfile.TemporaryDirectory() as tmpdir:
            recipe_path = Path(tmpdir) / "recipe.yaml"
            fig.save_recipe(str(recipe_path))
            assert recipe_path.exists()
