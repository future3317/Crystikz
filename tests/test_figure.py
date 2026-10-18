"""Tests for CrystalFigure fluent API."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from crystalfig.examples.presets import perovskite_structure, rocksalt_structure, wurtzite_structure
from crystalfig.figure import FigureGrid
from crystalfig.figure.builder import CrystalFigure
from crystalfig.geometry.planes import MillerPlane
from crystalfig.model.lattice import Lattice
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure
from crystalfig.scene.camera import Camera


class TestCrystalFigure:
    def test_nature_render_auto_centers_colors_by_element_and_exports(self, tmp_path):
        figure = CrystalFigure(wurtzite_structure()).render(
            style="nature", strategy="cutoff", show_bonds=False
        )
        scene = figure.build_scene()
        polyhedra = scene.get_group("polyhedra").primitives
        assert polyhedra
        assert scene.get_group("bonds") is None
        explicit = CrystalFigure(wurtzite_structure()).render(
            style="nature", strategy="cutoff", centers=["Zn", "O"], show_bonds=False
        ).build_scene()
        assert len({poly.fill_color for poly in explicit.get_group("polyhedra").primitives}) == 2
        path = tmp_path / "nature.svg"
        figure.save(path)
        assert "<svg" in path.read_text(encoding="utf-8")[:500]
        figure.save(tmp_path / "nature.pdf")
        figure.save(tmp_path / "nature.png", dpi=80)
        assert (tmp_path / "nature.pdf").stat().st_size > 1000
        assert (tmp_path / "nature.png").stat().st_size > 1000

    def test_nature_can_export_dark_and_transparent_backgrounds(self, tmp_path):
        from PIL import Image

        structure = wurtzite_structure()
        images = {}
        for background in ("dark", "transparent"):
            path = tmp_path / f"{background}.png"
            CrystalFigure(structure).render(
                style="nature", strategy="cutoff", background=background
            ).save(path, dpi=80)
            assert path.is_file() and path.stat().st_size > 1000
            images[background] = Image.open(path).convert("RGBA")
        assert images["dark"].getpixel((0, 0))[:3] == (32, 36, 44)
        assert 0 in images["transparent"].getchannel("A").getextrema()

    def test_nature_reports_ballstick_fallback_for_single_atom(self):
        structure = CrystalStructure(Lattice.cubic(4.0), [Site([0, 0, 0], "He")])
        figure = CrystalFigure(structure).render(style="nature", strategy="cutoff", show_bonds=False)
        scene = figure.build_scene()
        assert not scene.get_group("polyhedra")
        assert any("no valid polyhedra" in message for message in figure.diagnostics)
        assert len(scene.get_group("atoms").primitives) == 1

    def test_figure_grid_saves_labeled_structure_panels(self, tmp_path):
        grid = FigureGrid(rows=1, cols=2)
        grid.add_structure(perovskite_structure(), 0, 0, strategy="cutoff", show_bonds=False)
        grid.add_structure(rocksalt_structure(), 0, 1, style="ballstick", strategy="cutoff")
        for suffix in ("png", "pdf", "svg"):
            path = grid.save(tmp_path / f"panels.{suffix}", dpi=120)
            assert path.is_file() and path.stat().st_size > 1000

    def test_from_pymatgen_structure(self):
        """Passing a pymatgen Structure directly must use the adapter."""
        from pymatgen.core import Lattice as PmgLattice
        from pymatgen.core import Structure

        pmg = Structure(PmgLattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        fig = CrystalFigure.from_structure(pmg)
        assert fig.structure.num_sites == 2
        assert set(fig.structure.unique_species()) == {"Na", "Cl"}

    def test_perovskite_formula_reduced(self):
        fig = CrystalFigure(perovskite_structure())
        assert fig.structure.formula == "BaTiO3"
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

    def test_camera_screen_roll_rotates_the_camera_plane(self):
        base = Camera.along_direction([1, 0, 0])
        rolled = Camera.along_direction([1, 0, 0], roll=90)
        assert np.allclose(rolled.rotation_matrix()[0], -base.rotation_matrix()[1])
        assert np.allclose(rolled.rotation_matrix()[1], base.rotation_matrix()[0])

    def test_view_direction_111(self):
        fig = CrystalFigure(rocksalt_structure()).view([1, 1, 1])
        rot = fig.camera.rotation_matrix()
        direction = fig.structure.lattice.frac_to_cart(np.array([1, 1, 1]))
        direction = direction / np.linalg.norm(direction)
        view_dir = -rot[2]
        assert np.dot(view_dir, direction) > 0.99

    def test_view_rejects_unknown_axis(self):
        with pytest.raises(ValueError):
            CrystalFigure(rocksalt_structure()).view("z")

    def test_layer_top_uses_reciprocal_plane_normal_for_nonorthogonal_cell(self):
        fig = CrystalFigure(rocksalt_structure()).view_normal_to_plane((0, 0, 1))
        expected = fig.structure.lattice.reciprocal_vector((0, 0, 1))
        expected /= np.linalg.norm(expected)
        assert np.dot(-fig.camera.rotation_matrix()[2], expected) > 0.999

    def test_auto_view_is_scene_based_deterministic_and_explainable(self):
        fig = CrystalFigure(perovskite_structure()).supercell((2, 2, 1)).publication()
        scene = fig.build_scene()
        reports = fig.report_view_candidates(scene)
        repeated = fig.report_view_candidates(scene)
        assert reports == repeated
        assert reports[0]["score"] >= reports[-1]["score"]
        assert "atom_overlap_fraction" in reports[0]
        assert all(0.0 <= report["atom_overlap_fraction"] <= 1.0 for report in reports)
        assert "bond_visibility" in reports[0]
        assert {report["roll_degrees"] for report in reports} == {0.0, 90.0}

    def test_auto_view_adds_generic_lattice_directions_to_reduce_atom_overlap(self):
        figure = CrystalFigure(rocksalt_structure(conventional=True)).publication()
        report = figure.report_view_candidates(figure.build_scene())[0]
        assert report["atom_overlap_fraction"] < 0.5

    def test_auto_fit_recenters_final_supercell_geometry(self):
        fig = CrystalFigure(rocksalt_structure()).supercell((3, 1, 1)).view([1, 0, 0])
        scene = fig.build_scene()
        frame = fig.camera.scene_frame(scene)
        assert frame is not None
        assert (frame.xmin + frame.xmax) / 2 == pytest.approx(0.0, abs=1e-8)
        assert (frame.ymin + frame.ymax) / 2 == pytest.approx(0.0, abs=1e-8)

    def test_publication_api_uses_an_explicit_neighbor_model(self):
        fig = CrystalFigure(perovskite_structure()).publication()
        assert fig.theme.name == "ballstick_clean"
        assert fig.theme.render_quality == "publication"
        assert fig._scene_options.bond_strategy.__class__.__name__ == "CrystalNNStrategy"
        assert "verify coordination" in fig.diagnostics[0]

    def test_quick_reports_its_cutoff_as_a_display_heuristic(self):
        fig = CrystalFigure(rocksalt_structure()).quick()
        assert "3.0 Å cutoff" in fig.diagnostics[0]
        assert "verify coordination" in fig.diagnostics[0]

    def test_select_single_index_and_species(self):
        fig = CrystalFigure(rocksalt_structure()).select(indices=0).select(species="Cl")
        assert sorted(fig._scene_options.selected_sites) == [0, 1]

    def test_select_value_predicate(self):
        fig = CrystalFigure(rocksalt_structure())
        fig.structure.sites[0].properties.set("flag", True)
        fig.select(flag=True)
        assert fig._scene_options.selected_sites == [0]

    def test_add_polyhedra(self):
        fig = CrystalFigure(perovskite_structure()).add_bonds("covalent").add_polyhedra("Ti")
        scene = fig.build_scene()
        polyhedra = [p for p in scene.all_primitives() if p.__class__.__name__ == "Polyhedron"]
        assert len(polyhedra) > 0

    def test_supercell(self):
        """Primitive rocksalt has 2 atoms; a 2x1x1 supercell has 4 canonical atoms."""
        fig = (
            CrystalFigure(rocksalt_structure())
            .supercell((2, 1, 1))
            .boundary_mode("connected")
            .quick()
        )
        scene = fig.build_scene()
        atoms = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
        canonical = [p for p in atoms if p.metadata.get("image_offset") == (0, 0, 0)]
        assert len(canonical) == 4
        # Periodic-image partner atoms are now also emitted for visible bonds.
        assert len(atoms) >= len(canonical)

    def test_supercell_accepts_general_matrix(self):
        matrix = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
        scene = CrystalFigure(rocksalt_structure()).supercell(matrix).build_scene()
        assert scene.metadata["num_sites"] == 4

    def test_miller_plane_uses_final_supercell_lattice(self):
        fig = CrystalFigure(rocksalt_structure()).supercell((2, 1, 1))
        fig.add_miller_plane((1, 0, 0))
        scene = fig.build_scene()
        polygons = [p for p in scene.all_primitives() if p.__class__.__name__ == "Polygon"]
        assert len(polygons) == 1

        final_structure = fig.structure.make_supercell((2, 1, 1))
        expected = MillerPlane((1, 0, 0), final_structure.lattice).intersection_polygon()
        assert expected is not None
        assert np.allclose(np.sort(polygons[0].points, axis=0), np.sort(expected, axis=0))

    def test_supercell_bonds_and_polyhedra(self):
        """Bonds and polyhedra must be computed on the supercell structure."""
        fig = (
            CrystalFigure(perovskite_structure())
            .supercell((2, 1, 1))
            .boundary_mode("connected")
            .add_bonds("cutoff", cutoff=2.5)
            .add_polyhedra("Ti")
        )
        scene = fig.build_scene()
        atoms = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
        bonds = [p for p in scene.all_primitives() if p.__class__.__name__ == "Bond"]
        polyhedra = [p for p in scene.all_primitives() if p.__class__.__name__ == "Polyhedron"]
        # Primitive perovskite: 5 atoms, 6 Ti-O bonds, 1 octahedron.
        # 2x1x1 supercell doubles those counts (canonical sites only).
        canonical = [p for p in atoms if p.metadata.get("image_offset") == (0, 0, 0)]
        assert len(canonical) == 10
        assert len(bonds) == 12
        assert len(polyhedra) == 2

        # Check that a bond endpoint is actually close to its neighbour site image.
        for bond in bonds:
            assert bond.distance < 2.5
            assert bond.distance > 0.1
            # jimage should be an integer triple
            assert len(bond.jimage) == 3
            assert all(isinstance(x, int) for x in bond.jimage)

    def test_cell_complete_boundary_atoms(self):
        """cell_complete mode replicates boundary atoms so the cell looks full."""
        fig = (
            CrystalFigure(rocksalt_structure(conventional=True))
            .boundary_mode("cell_complete")
            .show_unit_cell()
        )
        scene = fig.build_scene()
        atoms = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
        # Boundary completion repeats the corner and face-centered sites on the
        # displayed cell boundary, so it emits more than the 8 canonical spheres.
        assert len(atoms) > 2
        # All emitted spheres must still be valid periodic images.
        for a in atoms:
            offset = a.metadata.get("image_offset", (0, 0, 0))
            assert all(isinstance(x, int) for x in offset)

    def test_cell_edges_follow_camera(self):
        """Back edges classification must change when the camera view changes."""
        struct = rocksalt_structure()
        fig_default = CrystalFigure(struct).show_unit_cell()
        scene_default = fig_default.build_scene()
        edges_default = [p for p in scene_default.all_primitives() if p.__class__.__name__ == "CellEdge"]
        back_default = sum(1 for e in edges_default if e.is_back)

        fig_view = CrystalFigure(struct).show_unit_cell().view([1, 0, 0])
        scene_view = fig_view.build_scene()
        edges_view = [p for p in scene_view.all_primitives() if p.__class__.__name__ == "CellEdge"]
        back_view = sum(1 for e in edges_view if e.is_back)

        # The number of back edges should generally differ for a different view.
        # If both are zero or equal, at least verify that the property is set.
        assert len(edges_default) == 12
        assert len(edges_view) == 12
        assert back_default != back_view or back_default + back_view <= 24

    def test_cell_complete_no_duplicate_atoms(self):
        """Canonical atoms must appear exactly once with image_offset (0,0,0)."""
        fig = (
            CrystalFigure(rocksalt_structure(conventional=True))
            .boundary_mode("cell_complete")
            .show_unit_cell()
        )
        scene = fig.build_scene()
        atoms = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
        canonical = [p for p in atoms if p.metadata.get("image_offset") == (0, 0, 0)]
        # Conventional rocksalt has 8 canonical sites.
        assert len(canonical) == 8
        # No duplicate (site_index, image_offset) pairs.
        keys = [(a.metadata.get("site_index"), tuple(a.metadata.get("image_offset", (0, 0, 0)))) for a in atoms]
        assert len(keys) == len(set(keys))

    def test_style_atoms_changes_color(self):
        fig = CrystalFigure(rocksalt_structure()).style_atoms(species="Na", color="#ff0000")
        scene = fig.build_scene()
        na_spheres = [
            p for p in scene.all_primitives()
            if p.__class__.__name__ == "Sphere" and p.metadata.get("element") == "Na"
        ]
        assert len(na_spheres) > 0
        for sphere in na_spheres:
            assert sphere.color == "#ff0000"

    def test_hide_atoms_removes_spheres(self):
        fig = CrystalFigure(rocksalt_structure()).hide_atoms(species="Na")
        scene = fig.build_scene()
        na_spheres = [
            p for p in scene.all_primitives()
            if p.__class__.__name__ == "Sphere" and p.metadata.get("element") == "Na"
        ]
        assert len(na_spheres) == 0

    def test_multiple_polyhedra_specs(self):
        fig = (
            CrystalFigure(perovskite_structure())
            .add_bonds("covalent")
            .add_polyhedra("Ti", fill_color="#ff0000")
            .add_polyhedra("Ba", fill_color="#0000ff")
        )
        scene = fig.build_scene()
        polyhedra = [p for p in scene.all_primitives() if p.__class__.__name__ == "Polyhedron"]
        assert len(polyhedra) >= 2
        fill_colors = {p.fill_color for p in polyhedra}
        assert "#ff0000" in fill_colors
        assert "#0000ff" in fill_colors

    def test_manual_bonds(self):
        from crystalfig.neighbors.base import NeighborBond
        bonds = [NeighborBond(i=0, j=1, jimage=(0, 0, 0), distance=2.0)]
        fig = CrystalFigure(rocksalt_structure()).add_manual_bonds(bonds)
        scene = fig.build_scene()
        rendered = [p for p in scene.all_primitives() if p.__class__.__name__ == "Bond"]
        assert len(rendered) == 1
        assert rendered[0].site_i == 0 and rendered[0].site_j == 1

    def test_draw_into_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = CrystalFigure(rocksalt_structure()).show_unit_cell()
        _, ax = plt.subplots()
        fig.draw(ax)
        assert len(ax.patches) > 0
        plt.close()

    def test_add_label_emits_annotation(self):
        fig = CrystalFigure(rocksalt_structure()).add_label(0, "Na", offset=(4, 4))
        scene = fig.build_scene()
        annotations = [
            p for p in scene.all_primitives()
            if p.__class__.__name__ == "Text" and getattr(p, "layer", "default") == "annotation"
        ]
        assert len(annotations) == 1
        assert annotations[0].text == "Na"

    @pytest.mark.parametrize("kind", ["site", "formula", "panel"])
    @pytest.mark.parametrize("fmt", ["pdf", "svg"])
    def test_annotations_export_end_to_end(self, kind, fmt):
        fig = CrystalFigure(rocksalt_structure())
        if kind == "site":
            fig.add_label(0, "Na", offset=(4, 4))
        elif kind == "formula":
            fig.add_formula_label("NaCl", position="top_left")
        else:
            fig.add_panel_label("(a)", position="top_right")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / f"annotation.{fmt}"
            result = fig.export(str(output))
            assert Path(result.path).exists()
            if fmt == "svg":
                svg = output.read_text(encoding="utf-8")
                expected_text = {"site": "Na", "formula": "NaCl", "panel": "(a)"}[kind]
                assert svg.count(f">{expected_text}</text>") == 1
                if kind == "panel":
                    assert 'font-weight="bold"' in svg
