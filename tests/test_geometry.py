"""Tests for geometry utilities."""

import numpy as np
import pytest

from crystalfig.exceptions import GeometryError
from crystalfig.geometry.periodic import nearest_image, wrapped_frac
from crystalfig.geometry.planes import MillerPlane
from crystalfig.geometry.polyhedra import build_polyhedron
from crystalfig.model.lattice import Lattice


class TestPeriodic:
    def test_nearest_image(self):
        fi = np.array([0.9, 0.0, 0.0])
        fj = np.array([0.1, 0.0, 0.0])
        nearest = nearest_image(fi, fj)
        # Should wrap through boundary: 0.1 -> 1.1
        assert nearest[0] == pytest.approx(1.1)

    def test_wrap_frac(self):
        assert np.allclose(wrapped_frac(np.array([1.1, -0.2, 0.5])), [0.1, 0.8, 0.5])


class TestPolyhedra:
    def test_octahedron(self):
        center = np.zeros(3)
        vertices = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]
        poly = build_polyhedron(center, vertices)
        assert len(poly.faces) == 8  # octahedron has 8 triangular faces
        assert poly.volume > 0

    def test_tetrahedron(self):
        center = np.zeros(3)
        vertices = [
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ]
        poly = build_polyhedron(center, vertices)
        assert len(poly.faces) == 4

    def test_too_few_vertices_raises(self):
        with pytest.raises(GeometryError):
            build_polyhedron(np.zeros(3), [[1, 0, 0], [0, 1, 0]])


class TestMillerPlane:
    def test_intersection_with_cubic(self):
        lat = Lattice.cubic(1.0)
        plane = MillerPlane(hkl=[1, 1, 1], lattice=lat)
        poly = plane.intersection_polygon()
        assert poly is not None
        assert len(poly) >= 3
