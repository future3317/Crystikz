"""Tests for scene and camera."""

import numpy as np
import pytest

from crystalfig.examples.presets import perovskite_structure, rocksalt_structure
from crystalfig.figure.builder import CrystalFigure
from crystalfig.neighbors.strategies import CutoffStrategy
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
