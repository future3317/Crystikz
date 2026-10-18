"""Tests for the optional PyVista renderer."""

import importlib.util

import numpy as np
import pytest

from crystalfig.exceptions import OptionalDependencyError, RenderError
from crystalfig.renderers.base import RenderOptions
from crystalfig.renderers.pyvista_renderer import PyVistaRenderer
from crystalfig.scene.camera import Camera
from crystalfig.scene.primitives import (
    Arrow,
    Axis,
    Bond,
    CellEdge,
    LegendItem,
    Line,
    Polygon,
    Polyhedron,
    Polyline,
    Sphere,
    Text,
)
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


def test_missing_dependency_has_a_clear_optional_dependency_error(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def import_without_pyvista(name, *args, **kwargs):
        if name == "pyvista":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pyvista)
    with pytest.raises(OptionalDependencyError, match="pyvista"):
        PyVistaRenderer().export(Scene(), "figure.png", FigureTheme(), RenderOptions())


def test_non_png_export_is_rejected_before_loading_optional_dependency():
    with pytest.raises(RenderError, match="PNG export only"):
        PyVistaRenderer().export(Scene(), "figure.svg", FigureTheme(), RenderOptions())


@pytest.mark.parametrize(
    "options",
    [
        RenderOptions(width=0),
        RenderOptions(width=10, height=-1),
        RenderOptions(width=0.001, height=1, dpi=1),
        RenderOptions(width=10, dpi=0),
    ],
)
def test_invalid_output_dimensions_are_rejected_before_loading_dependency(options):
    with pytest.raises(RenderError, match="PNG"):
        PyVistaRenderer().export(Scene(), "figure.png", FigureTheme(), options)


def test_transparent_background_uses_white_rgb_background():
    theme = FigureTheme()
    background, transparent = PyVistaRenderer._background(theme, RenderOptions(transparent=True))
    assert background == "white"
    assert transparent

    background, transparent = PyVistaRenderer._background(theme, RenderOptions(background="white"))
    assert background == "white"
    assert not transparent


def test_auto_height_follows_the_shared_scene_frame():
    scene = Scene([
        Sphere(position=[0, 0, 0], radius=0.2),
        Sphere(position=[4, 0, 0], radius=0.2),
    ])
    camera = Camera()
    camera.fit_to_scene(scene)
    width, height = PyVistaRenderer._window_size(
        RenderOptions(width=89, dpi=100), scene, camera
    )
    assert width / height == pytest.approx(camera.scene_frame(scene).aspect, rel=0.01)


def test_pixel_limit_rejects_oversized_renders():
    with pytest.raises(RenderError, match="pixel limit"):
        PyVistaRenderer._window_size(RenderOptions(width=1000, dpi=2000, max_pixels=1000))


@pytest.mark.optional
@pytest.mark.skipif(importlib.util.find_spec("pyvista") is None, reason="PyVista is optional")
def test_exports_scene_as_png_with_requested_pixel_size(tmp_path):
    scene = Scene([
        Sphere(position=[0, 0, 0], radius=0.35, color="#cc5544", metadata={"site_index": 0}),
        Sphere(position=[1.2, 0, 0], radius=0.25, color="#4477aa", metadata={"site_index": 1}),
        Bond(start=[0, 0, 0], end=[1.2, 0, 0], radius=0.08, site_i=0, site_j=1),
        Polyhedron(
            vertices=[[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.4, 1]],
            faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            fill_color="#66aa88",
            edge_color="#225544",
            opacity=0.25,
        ),
        CellEdge(start=[-0.3, -0.3, -0.3], end=[1.5, -0.3, -0.3], color="#555555"),
        Line(start=[0, -0.5, 0], end=[1, -0.5, 0], color="#333333", layer="foreground"),
        Polyline(points=[[0, 0, 0.5], [0.5, 0.8, 0.5], [1, 0, 0.5]], color="#5555aa"),
        Polygon(
            points=[[0, 0, -0.2], [1, 0, -0.2], [0.5, 0.8, -0.2]],
            fill_color="#aa8855",
            opacity=0.2,
            edge_color="#665533",
        ),
        Arrow(start=[0, 0, 0], direction=[0, 0, 0.8], shaft_radius=0.02, head_radius=0.06, head_length=0.2),
        Axis(start=[0, 0, 0], direction=[0.8, 0, 0], shaft_radius=0.02, head_radius=0.06, head_length=0.2, name="a"),
        Text(position=[0.7, 0.8, 0.5], text="site", fontsize=8, color="#222222"),
        Text(position=[0.05, 0.95, 0], text="panel", fontsize=8, layer="annotation", coordinate_space="screen"),
        LegendItem(text="A", color="#cc5544"),
    ])
    output = tmp_path / "structure.png"

    PyVistaRenderer().export(
        scene,
        str(output),
        FigureTheme(),
        RenderOptions(width=25.4, height=12.7, dpi=40, transparent=True),
    )

    from PIL import Image

    with Image.open(output) as image:
        assert image.size == (40, 20)
        assert image.mode == "RGBA"
        assert image.info["dpi"] == pytest.approx((40, 40), abs=0.1)
        pixels = np.asarray(image)
        assert pixels.shape == (20, 40, 4)
        assert pixels[0, 0, 3] == 0
        assert np.any(pixels[:, :, 3] == 255)
