"""Tests for rendering backends."""

import tempfile
from pathlib import Path

import pytest

from crystalfig.examples.presets import rocksalt_structure
from crystalfig.export.exporter import RenderOptions
from crystalfig.export.latex import LatexCompiler
from crystalfig.figure.builder import CrystalFigure
from crystalfig.renderers.matplotlib_3d_renderer import Matplotlib3DRenderer
from crystalfig.renderers.matplotlib_renderer import MatplotlibRenderer
from crystalfig.renderers.svg_renderer import SvgRenderer
from crystalfig.renderers.tikz_renderer import TikzRenderer
from crystalfig.scene.camera import Camera


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
