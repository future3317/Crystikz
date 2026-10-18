"""Tests for export/preflight utilities."""

import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from crystalfig.examples.presets import rocksalt_structure
from crystalfig.export.exporter import Exporter, RenderOptions
from crystalfig.export.preflight import preflight_export, preflight_pdf
from crystalfig.figure.builder import CrystalFigure
from crystalfig.scene.camera import Camera
from crystalfig.scene.primitives import Sphere, Text
from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


class TestPreflight:
    def test_preflight_raster_unknown_without_pdfimages(self):
        """If pdfimages is unavailable, raster_objects must remain None (unknown)."""
        fig = CrystalFigure(rocksalt_structure()).quick()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pdf"
            fig.export(str(path))
            with mock.patch("shutil.which", side_effect=lambda name: name != "pdfimages"):
                report = preflight_pdf(str(path))
            assert report.exists
            assert report.raster_objects is None

    @pytest.mark.optional
    def test_hybrid_pdf_has_measured_page_size_and_restores_font_settings(self, tmp_path):
        pytest.importorskip("pyvista")
        import matplotlib

        scene = Scene([
            Sphere(position=[0, 0, 0], radius=0.35, metadata={"site_index": 0}),
            Text(
                position=[0, 0, 0], text="editable-label", layer="annotation",
                metadata={"kind": "site_label", "offset": (0, 0)},
            ),
        ])
        exporter = Exporter(scene, FigureTheme(), Camera())
        old_fonttype = matplotlib.rcParams["pdf.fonttype"]
        path = tmp_path / "hybrid.pdf"
        result = exporter.export_hybrid_pdf(
            str(path), RenderOptions(width=89, height=55, dpi=72, transparent=True)
        )
        assert result.vector_status == "hybrid"
        assert matplotlib.rcParams["pdf.fonttype"] == old_fonttype
        report = preflight_export(
            str(path), expected_width_mm=89, expected_height_mm=55, vector_status="hybrid"
        )
        assert report["page_width_mm"] == pytest.approx(89, abs=0.3)
        assert report["page_height_mm"] == pytest.approx(55, abs=0.3)
        assert path.stat().st_size > 0
        if shutil.which("pdftotext"):
            extracted = tmp_path / "hybrid.txt"
            import subprocess

            subprocess.run(["pdftotext", str(path), str(extracted)], check=True)
            assert "editable-label" in extracted.read_text(encoding="utf-8")
