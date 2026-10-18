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
    @pytest.mark.parametrize(
        ("font_type", "has_type3", "expected_warning"),
        [("Type 3", True, True), ("Type 42", False, False)],
    )
    def test_pdffonts_type_detection(self, tmp_path, font_type, has_type3, expected_warning):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"pdf")
        header = "name                 type"
        output = f"{header}\n{'-' * len(header)}\n{'ExampleFont':<21}{font_type} Custom yes no no 1 0"
        completed = mock.Mock(stdout=output, returncode=0)
        with (
            mock.patch("shutil.which", side_effect=lambda name: "pdffonts" if name == "pdffonts" else None),
            mock.patch("subprocess.run", return_value=completed),
        ):
            report = preflight_export(str(path))

        assert report["has_type3_fonts"] is has_type3
        assert any("Type 3" in warning for warning in report["warnings"]) is expected_warning

    def test_pdffonts_nonzero_exit_adds_warning(self, tmp_path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"pdf")
        completed = mock.Mock(stdout="", returncode=1)
        with (
            mock.patch("shutil.which", side_effect=lambda name: "pdffonts" if name == "pdffonts" else None),
            mock.patch("subprocess.run", return_value=completed),
        ):
            report = preflight_pdf(str(path))

        assert "pdffonts exited with status 1." in report.warnings
        assert report.font_detection == "failed"

    def test_pdffonts_missing_is_reported_as_unknown(self, tmp_path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"pdf")
        with mock.patch("shutil.which", return_value=None):
            report = preflight_pdf(str(path))

        assert report.font_detection == "unknown"
        assert any("Type 3 font status is unknown" in warning for warning in report.warnings)

    @pytest.mark.parametrize(("fonttype", "has_type3"), [(3, True), (42, False)])
    def test_real_matplotlib_pdf_font_detection(self, tmp_path, fonttype, has_type3):
        if not shutil.which("pdffonts"):
            pytest.skip("Poppler pdffonts is not installed")
        import matplotlib
        from matplotlib.figure import Figure

        path = tmp_path / f"type-{fonttype}.pdf"
        with matplotlib.rc_context({"pdf.fonttype": fonttype}):
            figure = Figure(figsize=(2, 1))
            figure.text(0.1, 0.5, "Crystikz font inspection")
            figure.savefig(path, format="pdf")

        report = preflight_pdf(str(path))
        assert report.font_detection == "available"
        assert report.has_type3_fonts is has_type3
        assert report.fonts

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
