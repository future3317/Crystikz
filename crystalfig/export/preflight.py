"""Publication preflight checks for exported figures."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreflightReport:
    """Report from a preflight check."""

    path: str
    exists: bool
    page_size: tuple[float, float] | None
    fonts: list[str]
    has_type3_fonts: bool
    raster_objects: bool | None
    warnings: list[str]
    raw: dict[str, str]


def preflight_pdf(path: str) -> PreflightReport:
    """Run preflight checks on a PDF file."""
    path_obj = Path(path)
    report = PreflightReport(
        path=str(path_obj),
        exists=path_obj.exists(),
        page_size=None,
        fonts=[],
        has_type3_fonts=False,
        raster_objects=None,
        warnings=[],
        raw={},
    )
    if not report.exists:
        report.warnings.append("File does not exist.")
        return report

    if shutil.which("pdfinfo"):
        try:
            result = subprocess.run(["pdfinfo", str(path_obj)], capture_output=True, text=True, shell=False)
            report.raw["pdfinfo"] = result.stdout
            for line in result.stdout.splitlines():
                if line.lower().startswith("page size:"):
                    parts = line.split(":", 1)[1].strip().split()
                    if len(parts) >= 2:
                        report.page_size = (float(parts[0]), float(parts[1]))
        except Exception as exc:
            report.warnings.append(f"pdfinfo failed: {exc}")

    if shutil.which("pdffonts"):
        try:
            result = subprocess.run(["pdffonts", str(path_obj)], capture_output=True, text=True, shell=False)
            report.raw["pdffonts"] = result.stdout
            lines = result.stdout.strip().splitlines()
            if len(lines) > 2:
                header = lines[1]
                name_col = header.find("fontname")
                type_col = header.find("type")
                for line in lines[2:]:
                    font_name = line[name_col:type_col].strip() if name_col >= 0 else line.split()[0]
                    font_type = line[type_col:].split()[0] if type_col >= 0 else ""
                    report.fonts.append(font_name)
                    if "Type 3" in font_type or "Type3" in font_type:
                        report.has_type3_fonts = True
        except Exception as exc:
            report.warnings.append(f"pdffonts failed: {exc}")

    # Detect embedded raster images using pdfimages if available.
    if shutil.which("pdfimages"):
        report.raster_objects = False
        try:
            result = subprocess.run(
                ["pdfimages", "-list", str(path_obj)],
                capture_output=True,
                text=True,
                shell=False,
            )
            report.raw["pdfimages"] = result.stdout
            lines = result.stdout.strip().splitlines()
            if len(lines) > 2:
                report.raster_objects = True
        except Exception as exc:
            report.warnings.append(f"pdfimages failed: {exc}")

    return report


def preflight_raster(path: str) -> dict[str, any]:
    """Check basic properties of a raster image."""
    from PIL import Image

    img = Image.open(path)
    return {
        "size": img.size,
        "mode": img.mode,
        "format": img.format,
        "has_alpha": img.mode in ("RGBA", "LA", "P"),
    }
