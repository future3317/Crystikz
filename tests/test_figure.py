"""Tests for CrystalFigure fluent API."""

import tempfile
from pathlib import Path

import numpy as np

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

    def test_view_direction_along_axis(self):
        """[100] should make the camera look along the lattice a-vector."""
        fig = CrystalFigure(rocksalt_structure()).view([1, 0, 0])
        rot = fig.camera.rotation_matrix()
        a_axis = fig.structure.lattice.matrix[0]
        a_axis = a_axis / np.linalg.norm(a_axis)
        view_dir = -rot[2]  # camera looks along negative z
        assert np.dot(view_dir, a_axis) > 0.99

    def test_view_direction_111(self):
        fig = CrystalFigure(rocksalt_structure()).view([1, 1, 1])
        rot = fig.camera.rotation_matrix()
        direction = fig.structure.lattice.frac_to_cart(np.array([1, 1, 1]))
        direction = direction / np.linalg.norm(direction)
        view_dir = -rot[2]
        assert np.dot(view_dir, direction) > 0.99

    def test_add_polyhedra(self):
        fig = CrystalFigure(perovskite_structure()).add_bonds("covalent").add_polyhedra("Ti")
        scene = fig.build_scene()
        polyhedra = [p for p in scene.all_primitives() if p.__class__.__name__ == "Polyhedron"]
        assert len(polyhedra) > 0

    def test_supercell(self):
        """Primitive rocksalt has 2 atoms; a 2x1x1 supercell has 4 atoms."""
        fig = CrystalFigure(rocksalt_structure()).supercell((2, 1, 1)).quick()
        scene = fig.build_scene()
        atoms = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
        assert len(atoms) == 4

    def test_supercell_bonds_and_polyhedra(self):
        """Bonds and polyhedra must be computed on the supercell structure."""
        fig = (
            CrystalFigure(perovskite_structure())
            .supercell((2, 1, 1))
            .add_bonds("cutoff", cutoff=2.5)
            .add_polyhedra("Ti")
        )
        scene = fig.build_scene()
        atoms = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
        bonds = [p for p in scene.all_primitives() if p.__class__.__name__ == "Bond"]
        polyhedra = [p for p in scene.all_primitives() if p.__class__.__name__ == "Polyhedron"]
        # Primitive perovskite: 5 atoms, 6 Ti-O bonds, 1 octahedron.
        # 2x1x1 supercell doubles those counts.
        assert len(atoms) == 10
        assert len(bonds) == 12
        assert len(polyhedra) == 2

        # Check that a bond endpoint is actually close to its neighbour site image.
        for bond in bonds:
            assert bond.distance < 2.5
            assert bond.distance > 0.1
            # jimage should be an integer triple
            assert len(bond.jimage) == 3
            assert all(isinstance(x, int) for x in bond.jimage)
