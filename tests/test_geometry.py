"""Tests for geometry utilities."""

import numpy as np
import pytest

from crystalfig.exceptions import GeometryError
from crystalfig.geometry.periodic import PeriodicImages, nearest_image, wrapped_frac
from crystalfig.geometry.planes import MillerPlane
from crystalfig.geometry.polyhedra import build_polyhedron
from crystalfig.geometry.reciprocal import BrillouinZone
from crystalfig.model.lattice import Lattice


class TestPeriodic:
    def test_nearest_image_fractional(self):
        fi = np.array([0.9, 0.0, 0.0])
        fj = np.array([0.1, 0.0, 0.0])
        image_frac, jimage = nearest_image(fi, fj)
        # Should wrap through boundary: 0.1 -> 1.1
        assert image_frac[0] == pytest.approx(1.1)
        assert jimage == (1, 0, 0)

    def test_nearest_image_triclinic(self):
        """Minimum image must respect the actual lattice metric, not just cubic wrapping."""
        lat = Lattice.from_parameters(3.0, 4.0, 5.0, 70.0, 80.0, 85.0)
        fi = np.array([0.9, 0.1, 0.1])
        fj = np.array([0.1, 0.9, 0.9])
        image_cart, jimage = nearest_image(fi, fj, lattice=lat)
        # The returned image should be the closest periodic copy in Cartesian space.
        direct_dist = np.linalg.norm(lat.frac_to_cart(fj) - lat.frac_to_cart(fi))
        image_dist = np.linalg.norm(image_cart - lat.frac_to_cart(fi))
        assert image_dist <= direct_dist + 1e-6
        assert image_dist < 5.0  # much closer than a cell diagonal

    def test_nearest_image_hexagonal(self):
        """Hexagonal PBC: nearest image across the 120-degree angle."""
        lat = Lattice.from_parameters(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
        fi = np.array([0.9, 0.1, 0.0])
        fj = np.array([0.1, 0.9, 0.0])
        image_cart, _jimage = nearest_image(fi, fj, lattice=lat)
        ref_cart = lat.frac_to_cart(fi)
        dist = np.linalg.norm(image_cart - ref_cart)
        # The nearest image should be within a single cell dimension.
        assert dist < max(lat.lengths) + 1e-6

    def test_wrap_frac(self):
        assert np.allclose(wrapped_frac(np.array([1.1, -0.2, 0.5])), [0.1, 0.8, 0.5])

    def test_periodic_images_within_radius_shape(self):
        lat = Lattice.cubic(4.0)
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        images = PeriodicImages(lat, coords)
        image_cart, offsets = images.images_within_radius(np.array([0.0, 0.0, 0.0]), 5.0)
        assert image_cart.shape[0] == offsets.shape[0]
        assert image_cart.shape[1] == 3


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
        # Compute polygon area; must be positive and non-degenerate.
        centroid = poly.mean(axis=0)
        centered = poly - centroid
        u = centered[0]
        v = np.cross(plane.normal / np.linalg.norm(plane.normal), u)
        v = v / np.linalg.norm(v)
        angles = np.arctan2(centered @ v, centered @ u)
        ordered = poly[np.argsort(angles)]
        area = 0.0
        for i in range(1, len(ordered) - 1):
            area += 0.5 * np.linalg.norm(
                np.cross(ordered[i] - ordered[0], ordered[i + 1] - ordered[0])
            )
        assert area > 0.01

    def test_invalid_miller_indices(self):
        lat = Lattice.cubic(1.0)
        with pytest.raises(GeometryError):
            MillerPlane(hkl=[0, 0, 0], lattice=lat)


class TestBrillouinZone:
    def test_edge_indices_local(self):
        """Brillouin zone edges must refer to vertices in the returned array."""
        lat = Lattice.cubic(4.0)
        bz = BrillouinZone.from_lattice(lat)
        n_vertices = len(bz.vertices)
        assert n_vertices > 0
        for i, j in bz.edges:
            assert 0 <= i < n_vertices
            assert 0 <= j < n_vertices
            assert i != j
