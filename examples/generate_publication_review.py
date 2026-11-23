"""Render the publication benchmark and compare it with the frozen v4 gallery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from generate_visual_review_v4 import DATA, ROOT, structures
from generate_visual_review_v4 import OUT as BASELINE
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter

from crystalfig import (
    CameraConfig,
    CrystalFigure,
    DisplayConfig,
    FigureConfig,
    FigureGrid,
    RenderConfig,
)
from crystalfig.export.exporter import Exporter, RenderOptions
from crystalfig.export.preflight import preflight_export
from crystalfig.io.loader import load_structure
from crystalfig.scene.primitives import Bond, Polyhedron, Sphere

OUT = ROOT / "gallery" / "publication_review"
WIDTH_MM = 89
DPI = 600
THUMBNAIL = (344, 268)


def thumbnail(path):
    with Image.open(path) as source:
        image = Image.new("RGBA", source.size, "white")
        image.alpha_composite(source.convert("RGBA"))
        return ImageOps.pad(image.convert("RGB"), THUMBNAIL, color="white")


def ssim(first, second):
    """Windowed grayscale SSIM, after fitting both images to one white canvas."""
    a, b = (np.asarray(image.convert("L"), dtype=float) / 255 for image in (first, second))
    mean_a, mean_b = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    var_a = gaussian_filter(a * a, 1.5) - mean_a**2
    var_b = gaussian_filter(b * b, 1.5) - mean_b**2
    cov = gaussian_filter(a * b, 1.5) - mean_a * mean_b
    return float(np.mean(
        ((2 * mean_a * mean_b + 0.01**2) * (2 * cov + 0.03**2))
        / ((mean_a**2 + mean_b**2 + 0.01**2) * (var_a + var_b + 0.03**2))
    ))


def image_metrics(path):
    with Image.open(path) as image:
        dpi = image.info["dpi"]
        if image.width != round(WIDTH_MM * DPI / 25.4) or abs(dpi[0] - DPI) > 1:
            raise RuntimeError(f"Unexpected export size or DPI for {path.name}")
        pixels = np.asarray(image.convert("RGB"))
        mask = np.min(pixels, axis=2) < 240
        rows, cols = np.nonzero(mask)
        return {
            "pixels": image.size,
            "dpi": dpi,
            "occupancy_width": float(np.ptp(cols) / image.width),
            "occupancy_height": float(np.ptp(rows) / image.height),
            "foreground_fraction": float(mask.mean()),
        }


def make_sheet(rows, path):
    """Assemble images with integral thumbnails and aligned row headings."""
    width, height = THUMBNAIL
    image = Image.new("RGB", (len(rows[0]) * (width + 24), len(rows) * (height + 36)), "#edf1f5")
    draw = ImageDraw.Draw(image)
    for row, panels in enumerate(rows):
        for col, (caption, panel) in enumerate(panels):
            x, y = col * (width + 24), row * (height + 36)
            draw.text((x + 12, y + 8), caption, fill="#263344")
            if panel is not None:
                image.paste(panel, (x + 12, y + 26))
    image.save(path)


def read_scene(figure, scene):
    primitives = scene.all_primitives()
    polyhedra = [item for item in primitives if isinstance(item, Polyhedron)]
    return {
        "display": figure._display_plan.intent,
        "repeat": figure._display_plan.repeat,
        "atoms": sum(isinstance(item, Sphere) for item in primitives),
        "bonds": sum(isinstance(item, Bond) for item in primitives),
        "polyhedra": len(polyhedra),
        "clipped_polyhedra": sum(bool(item.metadata.get("clipped")) for item in polyhedra),
        "ligands_per_shell": dict(Counter(len(poly.vertex_metadata) for poly in polyhedra)),
        "camera": {
            "candidate": scene.metadata.get("camera_used"),
            "score": scene.metadata.get("camera_score_components"),
            "rotation": figure.camera.rotation_matrix().tolist(),
            "target": figure.camera.target.tolist(),
            "padding": figure.camera.padding,
            "projection": figure.camera.projection,
        },
    }


def new_figure(structure, style, *, display=None, camera=None):
    return CrystalFigure.from_structure(structure).configure(FigureConfig(
        style=style,
        display=display or DisplayConfig(),
        camera=camera or CameraConfig(),
        render=RenderConfig(width_mm=WIDTH_MM, dpi=DPI),
    ))


def main():
    import pyvista as pv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    materials = structures() | {
        name: load_structure(DATA / f"{name}.cif") for name in ("Eu2FeGe2OS6", "Ce6Cd23Te")
    }
    manifest = {
        "render": {"dpi": DPI, "width_mm": WIDTH_MM, "pyvista": pv.__version__, "vtk": list(pv.vtk_version_info)},
        "structures": {},
    }
    rows = []
    options = RenderOptions(width=WIDTH_MM, dpi=DPI)
    for name, structure in materials.items():
        old_path = BASELINE / f"{name}_journal_vivid.png"
        old = thumbnail(old_path) if old_path.is_file() else None
        row = [(f"{name} / v4" if old is not None else f"{name} / new CIF", old)]
        manifest["structures"][name] = {}
        for style in ("journal_clean", "journal_vivid"):
            figure = new_figure(structure, style)
            scene = figure.build_scene()
            path = out / f"{name}_{style}.png"
            result = Exporter(scene, figure.theme, camera=figure.camera).export(
                path, options=options, backend="pyvista",
            )
            panel = thumbnail(path)
            row.append((style.replace("_", " ").title(), panel))
            details = read_scene(figure, scene) | image_metrics(path) | {
                "path": path.name,
                "depth_peeling": result.metadata.get("depth_peeling_used"),
            }
            if details["polyhedra"] and not details["depth_peeling"]:
                raise RuntimeError(f"VTK did not depth-peel {path.name}")
            if old is not None:
                details["ssim_vs_v4"] = ssim(old, panel)
            manifest["structures"][name][style] = details
        rows.append(row)
    make_sheet(rows, out / "comparison_contact_sheet.png")

    # Use one fixed view and scene across renderers so shading differences are visible.
    comparisons = []
    manifest["fixed_cases"] = {}
    for name, style, intent in (
        ("BaTiO3", "vesta_coordination", "coordination"),
        ("Si", "vesta_ballstick", "network"),
    ):
        figure = new_figure(
            materials[name], style, display=DisplayConfig(intent=intent, region="cell"),
            camera=CameraConfig(mode="fixed", elevation=32, azimuth=32),
        )
        scene = figure.build_scene()
        exporter = Exporter(scene, figure.theme, camera=figure.camera)
        panel = []
        for backend in ("pyvista", "matplotlib"):
            path = out / f"{name}_fixed_{backend}.png"
            exporter.export(path, options=options, backend=backend)
            panel.append((f"{name} / {backend}", thumbnail(path)))
        exporter.export(out / f"{name}_fixed.svg", options=options, backend="svg")
        manifest["fixed_cases"][name] = read_scene(figure, scene) | {
            "ssim_pyvista_vs_matplotlib": ssim(panel[0][1], panel[1][1]),
        }
        comparisons.append(panel)
    make_sheet(comparisons, out / "backend_comparison.png")

    grid = FigureGrid(
        rows=1, cols=4, backend="hybrid", width_mm=183, height_mm=54,
        panel_gap_mm=2.2, legend_mode="panel", font_size_pt=7,
    )
    for col, (name, title, style, display) in enumerate((
        ("BaTiO3", "BaTiO₃ · TiO₆", "vesta_coordination", DisplayConfig(intent="coordination")),
        ("MoS2", "MoS₂ · S–Mo–S", "layered_crystal", DisplayConfig(intent="layered")),
        ("Eu2FeGe2OS6", "Eu₂FeGe₂OS₆", "vesta_coordination",
         DisplayConfig(intent="coordination", region="cell", centers=("Fe", "Ge"), show_unit_cell=True)),
        ("Ce6Cd23Te", "Ce₆Cd₂₃Te", "vesta_ballstick", DisplayConfig(intent="network", region="cell")),
    )):
        grid.add_structure(new_figure(materials[name], style, display=display), 0, col, title=title)
    grid.save(out / "paper_four_panel.png", dpi=DPI)
    pdf = grid.save(out / "paper_four_panel.pdf", dpi=DPI)
    pdf_report = preflight_export(pdf, expected_width_mm=183, expected_height_mm=54, vector_status="hybrid")
    manifest["four_panel_pdf"] = pdf_report | {"path": pdf.name}
    (out / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if pdf_report["warnings"] or pdf_report["page_size_points"] is None or pdf_report["has_type3_fonts"] is not False:
        raise RuntimeError(f"Four-panel PDF preflight failed: {pdf_report}")


if __name__ == "__main__":
    main()
