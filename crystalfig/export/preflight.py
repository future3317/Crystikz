"""Publication preflight checks for exported figures."""

from __future__ import annotations

import re
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
                    match = re.search(r"([\d.]+)\s+x\s+([\d.]+)", line.split(":", 1)[1])
                    if match:
                        report.page_size = (float(match.group(1)), float(match.group(2)))
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
    has_alpha = "A" in img.getbands() or "transparency" in img.info
    has_transparent_pixels = img.convert("RGBA").getchannel("A").getextrema()[0] < 255
    return {
        "size": img.size,
        "mode": img.mode,
        "format": img.format,
        "has_alpha": has_alpha,
        "has_transparent_pixels": has_transparent_pixels,
        "dpi": img.info.get("dpi"),
    }


def preflight_export(
    path: str,
    *,
    expected_width_mm: float | None = None,
    expected_height_mm: float | None = None,
    expected_dpi: int | None = None,
    transparent: bool | None = None,
    vector_status: str | None = None,
) -> dict:
    """Check measurable export properties; visual quality still needs review."""
    path_obj = Path(path)
    report = {
        "path": str(path_obj),
        "exists": path_obj.is_file(),
        "format": path_obj.suffix.lower().lstrip("."),
        "declared_vector_status": vector_status,
        "warnings": [],
        "manual_visual_review_required": True,
    }
    if not report["exists"]:
        report["warnings"].append("File does not exist.")
        return report

    if report["format"] == "pdf":
        pdf = preflight_pdf(str(path_obj))
        report.update({
            "page_size_points": pdf.page_size,
            "fonts": pdf.fonts,
            "has_type3_fonts": pdf.has_type3_fonts,
            "contains_raster_objects": pdf.raster_objects,
        })
        if pdf.page_size:
            width_mm = pdf.page_size[0] * 25.4 / 72
            height_mm = pdf.page_size[1] * 25.4 / 72
            report["page_width_mm"] = width_mm
            report["page_height_mm"] = height_mm
            if expected_width_mm is not None and abs(width_mm - expected_width_mm) > 0.3:
                report["warnings"].append(
                    f"PDF page width is {width_mm:.2f} mm; expected {expected_width_mm:.2f} mm."
                )
            if expected_height_mm is not None and abs(height_mm - expected_height_mm) > 0.3:
                report["warnings"].append(
                    f"PDF page height is {height_mm:.2f} mm; expected {expected_height_mm:.2f} mm."
                )
        if vector_status == "pure" and pdf.raster_objects:
            report["warnings"].append("PDF was declared pure but contains raster objects.")
        if vector_status == "hybrid" and pdf.raster_objects is False:
            report["warnings"].append("PDF was declared hybrid but no raster object was detected.")
        if pdf.has_type3_fonts:
            report["warnings"].append("PDF contains Type 3 fonts.")
    elif report["format"] in {"png", "tif", "tiff"}:
        raster = preflight_raster(str(path_obj))
        report.update(raster)
        dpi = raster["dpi"]
        if expected_dpi is not None and dpi:
            actual_dpi = float(dpi[0])
            if abs(actual_dpi - expected_dpi) > 1:
                report["warnings"].append(
                    f"Raster DPI is {actual_dpi:.1f}; expected {expected_dpi}."
                )
        if expected_width_mm is not None and dpi:
            width_mm = raster["size"][0] * 25.4 / float(dpi[0])
            report["physical_width_mm"] = width_mm
            if abs(width_mm - expected_width_mm) > 0.3:
                report["warnings"].append(
                    f"Raster width is {width_mm:.2f} mm; expected {expected_width_mm:.2f} mm."
                )
        if expected_height_mm is not None and dpi:
            height_mm = raster["size"][1] * 25.4 / float(dpi[1])
            report["physical_height_mm"] = height_mm
            if abs(height_mm - expected_height_mm) > 0.3:
                report["warnings"].append(
                    f"Raster height is {height_mm:.2f} mm; expected {expected_height_mm:.2f} mm."
                )
        if transparent is True and not raster["has_alpha"]:
            report["warnings"].append("Transparent output was requested but no alpha channel exists.")
        elif transparent is True and not raster["has_transparent_pixels"]:
            report["warnings"].append("Transparent output was requested but no transparent pixels were found.")
    elif report["format"] == "svg":
        import xml.etree.ElementTree as ET

        root = ET.parse(path_obj).getroot()
        report["svg_width"] = root.attrib.get("width")
        report["svg_height"] = root.attrib.get("height")
        report["contains_raster_images"] = bool(root.findall(".//{http://www.w3.org/2000/svg}image"))
        width_value = report["svg_width"] or ""
        height_value = report["svg_height"] or ""
        if width_value.endswith("mm"):
            width_mm = float(width_value[:-2])
            report["page_width_mm"] = width_mm
            if expected_width_mm is not None and abs(width_mm - expected_width_mm) > 0.3:
                report["warnings"].append(
                    f"SVG width is {width_mm:.2f} mm; expected {expected_width_mm:.2f} mm."
                )
        if height_value.endswith("mm"):
            height_mm = float(height_value[:-2])
            report["page_height_mm"] = height_mm
            if expected_height_mm is not None and abs(height_mm - expected_height_mm) > 0.3:
                report["warnings"].append(
                    f"SVG height is {height_mm:.2f} mm; expected {expected_height_mm:.2f} mm."
                )
        if vector_status == "pure" and report["contains_raster_images"]:
            report["warnings"].append("SVG was declared pure but contains raster images.")
    return report
