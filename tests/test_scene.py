"""Tests for scene and camera."""

import numpy as np

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
