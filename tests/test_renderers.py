"""Tests for rendering backends."""

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from crystalfig.examples.presets import rocksalt_structure
from crystalfig.exceptions import RenderError
from crystalfig.export.exporter import RenderOptions
from crystalfig.export.latex import LatexCompiler
from crystalfig.figure.builder import CrystalFigure
from crystalfig.renderers.matplotlib_3d_renderer import Matplotlib3DRenderer
from crystalfig.renderers.matplotlib_renderer import MatplotlibRenderer
from crystalfig.renderers.svg_renderer import SvgRenderer
from crystalfig.renderers.tikz_renderer import TikzRenderer
from crystalfig.scene.camera import Camera
from crystalfig.scene.primitives import Cylinder, Line, Polyline, Primitive, Sphere, Text
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


class TestMatplotlibRenderer:
    def test_export_pdf(self):
        fig = CrystalFigure(rocksalt_structure()).show_unit_cell().add_bonds("covalent")
        scene = fig.build_scene()
        renderer = MatplotlibRenderer(camera=Camera())
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.pdf"
            renderer.export(scene, str(out), fig.theme, RenderOptions(width=80.0))
            assert out.exists()
            assert out.stat().st_size > 0

    def test_export_png(self):
        fig = CrystalFigure(rocksalt_structure()).show_unit_cell()
        scene = fig.build_scene()
        renderer = MatplotlibRenderer(camera=Camera())
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.png"
            renderer.export(scene, str(out), fig.theme, RenderOptions(width=80.0, dpi=150), fmt="png")
            assert out.exists()

    def test_draw_does_not_close(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = CrystalFigure(rocksalt_structure()).show_unit_cell()
        scene = fig.build_scene()
        renderer = MatplotlibRenderer(camera=Camera())
        _, ax = plt.subplots()
        renderer.draw(ax, scene, fig.theme, RenderOptions(width=80.0))
        assert ax.figure is not None
        plt.close(ax.figure)


class TestSvgRenderer:
    def test_svg_contains_gradients_for_glossy_atoms(self):
        fig = CrystalFigure(rocksalt_structure()).show_unit_cell()
        scene = fig.build_scene()
        renderer = SvgRenderer(camera=Camera())
        svg = renderer.render(scene, fig.theme, RenderOptions(width=80.0))
        assert "<svg" in svg
        # Glossy shaded spheres emit clipPath defs for radial shading.
        assert "<clipPath" in svg

    @pytest.mark.parametrize("renderer_type", [MatplotlibRenderer, SvgRenderer, TikzRenderer])
    def test_unknown_primitive_raises_render_error(self, renderer_type):
        scene = Scene([Primitive()])
        with pytest.raises(RenderError):
            renderer_type(camera=Camera()).render(scene, FigureTheme(), RenderOptions())

    def test_layer_order_and_basic_line_primitives(self):
        scene = Scene([
            Line(start=[0, 0, 0], end=[1, 0, 0], color="#111111", layer="foreground"),
            Line(start=[0, 0, 0], end=[1, 0, 0], color="#222222", layer="background"),
            Cylinder(start=[0, 0, 0], end=[0, 1, 0], color="#333333"),
            Polyline(points=[[0, 0, 0], [0, 0, 1]], color="#444444"),
        ])
        svg = SvgRenderer(camera=Camera()).render(scene, FigureTheme(), RenderOptions())
        assert svg.index('stroke="#222222"') < svg.index('stroke="#111111"')
        assert '<polyline ' in svg

    def test_stroke_and_font_sizes_use_physical_points(self):
        scene = Scene([
            Sphere(position=[0, 0, 0], radius=1),
            Line(start=[-1, 0, 0], end=[1, 0, 0], linewidth=2),
            Text(position=[0, 0, 0], text="label", fontsize=10),
        ])
        renderer = SvgRenderer(camera=Camera())
        svg = renderer.render(scene, FigureTheme(), RenderOptions(width=100))
        dx = float(svg.split('viewBox="')[1].split()[2])
        expected = 2 * 25.4 / 72 * dx / 100
        assert f'stroke-width="{expected:.3f}"' in svg
        expected_font = 10 * 25.4 / 72 * dx / 100
        assert f'font-size="{expected_font:.4f}"' in svg

    def test_svg_preserves_matplotlib_vertical_orientation(self):
        camera = Camera(auto_fit=False)
        positions = np.array([[0, 0, 1], [1, 0, 0]])
        scene = Scene([
            Sphere(position=positions[0], radius=0.1, render_style="flat"),
            Sphere(position=positions[1], radius=0.1, render_style="flat"),
        ])
        renderer = SvgRenderer(camera=camera)
        svg = renderer.render(scene, FigureTheme(), RenderOptions(width=100))
        viewbox = re.search(r'viewBox="([^"]+)"', svg)
        circles = re.findall(r'<circle cx="([^"]+)" cy="([^"]+)"', svg)
        assert viewbox is not None and len(circles) == 2
        _, ymin, _, dy = map(float, viewbox.group(1).split())
        projected = camera.project(positions)
        expected = np.column_stack((projected[:, 0], 2 * ymin + dy - projected[:, 1]))
        actual = np.array([[float(x), float(y)] for x, y in circles])
        assert np.allclose(sorted(actual.tolist()), sorted(expected.tolist()), atol=1e-4)


class TestMatplotlib3DRenderer:
    def test_export_png(self):
        fig = CrystalFigure(rocksalt_structure(conventional=True)).show_unit_cell().add_bonds("covalent")
        scene = fig.build_scene()
        renderer = Matplotlib3DRenderer(camera=Camera())
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_3d.png"
            renderer.export(scene, str(out), fig.theme, RenderOptions(width=80.0, dpi=150), fmt="png")
            assert out.exists()
            assert out.stat().st_size > 0

    def test_export_pdf(self):
        fig = CrystalFigure(rocksalt_structure(conventional=True)).show_unit_cell()
        scene = fig.build_scene()
        renderer = Matplotlib3DRenderer(camera=Camera())
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_3d.pdf"
            renderer.export(scene, str(out), fig.theme, RenderOptions(width=80.0), fmt="pdf")
            assert out.exists()


class TestTikzRenderer:
    def test_render_standalone(self):
        fig = CrystalFigure(rocksalt_structure()).show_unit_cell()
        scene = fig.build_scene()
        renderer = TikzRenderer(camera=Camera())
        tex = renderer.render(scene, fig.theme, RenderOptions(), standalone=True)
        assert r"\documentclass" in tex
        assert r"\begin{tikzpicture}" in tex
        assert "positioning" in renderer.libraries

    @pytest.mark.latex
    def test_compile_tikz(self):
        if not LatexCompiler.detect_engine():
            pytest.skip("No LaTeX engine available")
        fig = CrystalFigure(rocksalt_structure()).show_unit_cell().add_bonds("covalent")
        scene = fig.build_scene()
        renderer = TikzRenderer(camera=Camera())
        tex = renderer.render(scene, fig.theme, RenderOptions(), standalone=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_tikz.pdf"
            compiler = LatexCompiler(engine=LatexCompiler.detect_engine())
            result = compiler.compile(tex, str(out))
            assert result.success
            assert Path(result.output_path).exists()
