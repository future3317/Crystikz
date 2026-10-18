"""Tests for scene and camera."""

import numpy as np
import pytest

from crystalfig.examples.presets import mos2_structure, perovskite_structure, rocksalt_structure
from crystalfig.figure.builder import CrystalFigure
from crystalfig.geometry.polyhedra import validate_mesh_winding
from crystalfig.neighbors.strategies import CrystalNNStrategy, CutoffStrategy
from crystalfig.scene.camera import Camera
from crystalfig.scene.primitives import Sphere
from crystalfig.scene.scene import Scene


class TestCamera:
    def test_orthographic_projection(self):
        cam = Camera(elevation=0, azimuth=0, projection="orthographic", scale=1.0)
        pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        uv = cam.project(pts)
        # Looking along z-axis: x->u, y->v
        assert np.allclose(uv[0], [1, 0])
        assert np.allclose(uv[1], [0, 1])

    def test_depth_ordering(self):
        cam = Camera(elevation=0, azimuth=0, projection="orthographic")
        pts = np.array([[0, 0, 1], [0, 0, -1]])
        depths = cam.depth(pts)
        assert depths[0] > depths[1]

    def test_along_direction(self):
        cam = Camera.along_direction([1, 1, 0])
        assert cam.projection == "orthographic"


class TestScene:
    def test_auto_publication_selects_structure_intent_and_complete_motif(self):
        figure = CrystalFigure(perovskite_structure(a=3.95, c=4.05)).publication()
        scene = figure.build_scene()
        assert scene.metadata["selected_intent"] == "coordination"
        assert scene.metadata["display_window"] == [2, 2, 2]
        polyhedra = scene.get_group("polyhedra").primitives
        assert len(polyhedra) == 4
        assert all(len(polyhedron.vertex_metadata) == 6 for polyhedron in polyhedra)
        atoms = scene.get_group("atoms").primitives
        visible_elements = {atom.metadata["element"] for atom in atoms}
        assert "Ba" in visible_elements
        assert sum(atom.metadata["element"] == "Ti" for atom in atoms) == len(polyhedra)
        identities = [
            (atom.metadata["site_index"], tuple(atom.metadata["image_offset"]))
            for atom in atoms
        ]
        assert len(identities) == len(set(identities))
        assert scene.metadata["camera_used"] != "isometric"
        assert scene.metadata["num_real_polyhedra"] == 4
        assert scene.metadata["num_clipped_polyhedra"] == 0

    def test_cell_clips_periodic_bonds_and_polyhedra_to_the_cell(self):
        structure = mos2_structure()
        figure = (
            CrystalFigure(structure)
            .boundary_mode("cell")
            .add_bonds("cutoff", cutoff=2.5)
            .add_polyhedra("Mo", strategy="cutoff")
        )
        scene = figure.build_scene()
        atoms = scene.get_group("atoms").primitives
        image_atoms = scene.get_group("image_atoms")
        for atom in atoms + (image_atoms.primitives if image_atoms else []):
            frac = structure.lattice.cart_to_frac(atom.position)
            assert np.all(frac >= -1e-8) and np.all(frac <= 1.0 + 1e-8)
        bonds = scene.get_group("bonds").primitives
        assert any(np.linalg.norm(bond.end - bond.start) < bond.distance - 1e-7 for bond in bonds)
        for bond in bonds:
            frac = structure.lattice.cart_to_frac(np.asarray([bond.start, bond.end]))
            assert np.all(frac >= -1e-8) and np.all(frac <= 1.0 + 1e-8)
        for poly in scene.get_group("polyhedra").primitives:
            frac = structure.lattice.cart_to_frac(np.asarray(poly.vertices))
            assert np.all(frac >= -1e-8) and np.all(frac <= 1.0 + 1e-8)
            assert poly.metadata["clipped"] is True
            assert any(vertex.get("kind") == "cut_vertex" for vertex in poly.vertex_metadata)
            assert any(face.get("kind") == "cut_face" for face in poly.face_metadata)
            assert validate_mesh_winding(poly.vertices, poly.faces)

    def test_connected_bonds_are_limited_to_one_cell_of_padding(self):
        structure = mos2_structure()
        scene = (
            CrystalFigure(structure)
            .boundary_mode("connected")
            .add_bonds("cutoff", cutoff=2.5)
            .add_polyhedra("Mo", strategy="cutoff")
            .build_scene()
        )
        for bond in scene.get_group("bonds").primitives:
            frac = structure.lattice.cart_to_frac(np.asarray([bond.start, bond.end]))
            assert np.all(frac >= -1.0 - 1e-8) and np.all(frac <= 2.0 + 1e-8)
        for poly in scene.get_group("polyhedra").primitives:
            frac = structure.lattice.cart_to_frac(np.asarray(poly.vertices))
            assert np.all(frac >= -1.0 - 1e-8) and np.all(frac <= 2.0 + 1e-8)

    def test_supercell_contains_only_requested_finite_copies(self):
        structure = rocksalt_structure()
        scene = CrystalFigure(structure).supercell((2, 1, 3)).boundary_mode("supercell").build_scene()
        assert scene.metadata["num_sites"] == structure.num_sites * 6
        atoms = scene.get_group("atoms").primitives
        assert len(atoms) == structure.num_sites * 6

    def test_repeated_cell_scene_builds_have_identical_geometry(self):
        figure = CrystalFigure(mos2_structure()).render(
            style="nature", strategy="cutoff", show_bonds=False
        )
        first = figure.build_scene().as_dict()
        second = figure.build_scene().as_dict()
        assert first == second

    def test_periodically_equivalent_coordinates_render_in_the_display_cell(self):
        structure = mos2_structure()
        original_coords = structure.frac_coords.copy()
        figure = (
            CrystalFigure(structure)
            .boundary_mode("connected")
            .add_bonds("cutoff", cutoff=2.5)
        )
        scene = figure.build_scene()

        atoms = scene.get_group("atoms").primitives
        assert np.allclose(structure.frac_coords, original_coords)
        atom_frac = np.array([structure.lattice.cart_to_frac(atom.position) for atom in atoms])
        assert np.all((atom_frac >= 0.0) & (atom_frac < 1.0))

        for bond in scene.get_group("bonds").primitives:
            assert np.linalg.norm(bond.end - bond.start) == pytest.approx(bond.distance)

        image_atoms = scene.get_group("image_atoms")
        assert image_atoms is not None
        for atom in image_atoms.primitives:
            site = structure.sites[atom.metadata["site_index"]]
            expected = site.frac_coords % 1.0 + np.asarray(atom.metadata["image_offset"])
            assert np.allclose(structure.lattice.cart_to_frac(atom.position), expected)

    def test_periodic_polyhedron_vertices_keep_neighbor_distances_after_folding(self):
        structure = mos2_structure()
        figure = CrystalFigure(structure).add_polyhedra("Mo", strategy="cutoff")
        scene = figure.build_scene()
        polyhedra = scene.get_group("polyhedra").primitives
        assert polyhedra
        bonds = CutoffStrategy(cutoff=2.5).get_bonds(structure)
        for poly in polyhedra:
            center = structure.lattice.frac_to_cart(
                structure.frac_coords[poly.center_site] % 1.0
            )
            expected = [
                bond.distance
                for bond in bonds
                if bond.i == poly.center_site or bond.j == poly.center_site
            ]
            actual = [np.linalg.norm(vertex - center) for vertex in poly.vertices]
            assert np.sort(actual) == pytest.approx(np.sort(expected))

    @pytest.mark.parametrize(
        "factory,center_element,neighbor_element,repeats",
        [
            (lambda: perovskite_structure(a=3.95, c=4.05), "Ti", "O", (2, 2, 2)),
            (mos2_structure, "Mo", "S", (2, 2, 1)),
        ],
    )
    def test_publication_keeps_complete_periodic_coordination_shells(
        self, factory, center_element, neighbor_element, repeats
    ):
        structure = factory()
        figure = CrystalFigure(structure).publication(
            intent="coordination", centers=center_element, region=repeats,
            coordination_shell="complete", strategy="crystalnn",
        )
        scene = figure.build_scene()
        supercell = structure.make_supercell(repeats)
        neighbor_bonds = CrystalNNStrategy().get_bonds(supercell)
        polyhedra = scene.get_group("polyhedra").primitives
        assert polyhedra
        assert scene.metadata["num_clipped_polyhedra"] == 0
        normal = np.cross(supercell.lattice.matrix[0], supercell.lattice.matrix[1])
        normal /= np.linalg.norm(normal)
        for poly in polyhedra:
            center = supercell.lattice.frac_to_cart(
                supercell.frac_coords[poly.center_site] % 1.0
            )
            identities = [
                (meta["site_index"], tuple(meta["image_offset"]))
                for meta in poly.vertex_metadata
            ]
            assert len(identities) == len(set(identities)) == 6
            assert all(
                supercell.sites[site_index].dominant_element == neighbor_element
                for site_index, _ in identities
            )
            distances = np.linalg.norm(np.asarray(poly.vertices) - center, axis=1)
            assert np.all(distances > 0.1)
            assert len(poly.vertices) == 6
            expected = [
                bond.distance for bond in neighbor_bonds
                if bond.i == poly.center_site or bond.j == poly.center_site
            ]
            assert len(expected) == 6
            assert np.sort(distances) == pytest.approx(np.sort(expected), abs=1e-5)
            if center_element == "Mo":
                heights = (np.asarray(poly.vertices) - center) @ normal
                assert np.any(heights > 0.1) and np.any(heights < -0.1)
        image_atoms = scene.get_group("image_atoms")
        assert image_atoms is None or all(atom is not None for atom in image_atoms.primitives)

    def test_add_and_all_primitives(self):
        scene = Scene()
        scene.add(Sphere(position=np.array([0, 0, 0]), radius=0.2))
        assert len(scene.all_primitives()) == 1

    def test_bounding_box(self):
        scene = Scene()
        scene.add(Sphere(position=np.array([0, 0, 0]), radius=0.2))
        scene.add(Sphere(position=np.array([1, 2, 3]), radius=0.2))
        bbox = scene.bounding_box()
        assert bbox is not None
        assert np.allclose(bbox[0], [0, 0, 0])
        assert np.allclose(bbox[1], [1, 2, 3])

    def test_serialization(self):
        scene = Scene()
        scene.add(Sphere(position=np.array([0, 0, 0]), radius=0.2))
        d = scene.as_dict()
        assert "primitives" in d

    def test_visible_false_skipped(self):
        scene = Scene()
        scene.add(Sphere(position=np.array([0, 0, 0]), radius=0.2, visible=True))
        scene.add(Sphere(position=np.array([1, 1, 1]), radius=0.2, visible=False))
        assert len(scene.all_primitives()) == 1

    def test_group_visible_respected(self):
        from crystalfig.scene.primitives import Group
        scene = Scene()
        scene.add(Sphere(position=np.array([0, 0, 0]), radius=0.2))
        group = Group(name="hidden", primitives=[Sphere(position=np.array([1, 1, 1]), radius=0.2)])
        group.visible = False
        scene.groups["hidden"] = group
        assert len(scene.all_primitives()) == 1
        assert scene.get_group("hidden") is group

    def test_polyhedron_specs_do_not_reuse_adjacency(self):
        class FourNeighborStrategy:
            def get_bonds(self, structure):
                center = structure.indices_of_species("Ti")[0]
                local = [
                    bond for bond in CutoffStrategy(cutoff=2.5).get_bonds(structure)
                    if bond.i == center or bond.j == center
                ]
                return local[:5]

        def vertex_counts(spec_order):
            fig = CrystalFigure(perovskite_structure())
            fig.add_bonds("cutoff", cutoff=2.5)
            for strategy in spec_order:
                fig.add_polyhedra("Ti", strategy=strategy)
            scene = fig.build_scene()
            return [len(poly.vertex_metadata) for poly in scene.get_group("polyhedra").primitives]

        assert vertex_counts([FourNeighborStrategy(), None]) == [5, 6]
        assert vertex_counts([None, FourNeighborStrategy()]) == [6, 5]

    def test_periodic_self_bonds_create_one_vertex_per_image(self):
        from crystalfig.model.lattice import Lattice
        from crystalfig.model.site import Site
        from crystalfig.model.structure import CrystalStructure

        structure = CrystalStructure(
            lattice=Lattice.cubic(3.0), sites=[Site([0, 0, 0], "Cu")]
        )
        figure = CrystalFigure(structure).add_polyhedra(
            "Cu", strategy=CutoffStrategy(cutoff=3.1)
        )
        poly = figure.build_scene().get_group("polyhedra").primitives[0]
        offsets = [tuple(meta["image_offset"]) for meta in poly.vertex_metadata]
        assert len(offsets) == 6
        assert len(set(offsets)) == 6

    def test_cell_complete_bonds_have_visible_endpoints_and_unique_identity(self):
        figure = CrystalFigure(rocksalt_structure(conventional=True)).add_bonds(
            strategy=CutoffStrategy(cutoff=3.0)
        )
        scene = figure.build_scene()
        atoms = scene.get_group("atoms").primitives + scene.get_group("image_atoms").primitives
        positions = {
            tuple(np.round(atom.position, 8))
            for atom in atoms
        }
        bonds = scene.get_group("bonds").primitives
        keys = set()
        for bond in bonds:
            assert tuple(np.round(bond.start, 8)) in positions
            assert tuple(np.round(bond.end, 8)) in positions
            key = (bond.site_i, bond.site_j, bond.jimage)
            reverse = (bond.site_j, bond.site_i, tuple(-x for x in bond.jimage))
            assert key not in keys and reverse not in keys
            keys.add(key)

    def test_failed_polyhedron_build_emits_site_diagnostic(self, monkeypatch):
        import crystalfig.scene.builder as scene_builder

        def fail_build(*args, **kwargs):
            raise ValueError("degenerate fixture")

        monkeypatch.setattr(scene_builder, "build_polyhedron", fail_build)
        fig = CrystalFigure(perovskite_structure()).add_polyhedra(
            "Ti", strategy="cutoff"
        )
        with pytest.warns(RuntimeWarning, match=r"site 1 \(Ti\) with 6 neighbors: degenerate fixture"):
            scene = fig.build_scene()
        assert scene.get_group("polyhedra") is None
