"""Tests for export/preflight utilities."""

import tempfile
from pathlib import Path
from unittest import mock

from crystalfig.examples.presets import rocksalt_structure
from crystalfig.export.preflight import preflight_pdf
from crystalfig.figure.builder import CrystalFigure


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
