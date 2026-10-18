"""Generate the v4 material-agnostic visual review gallery."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystalfig.examples.presets import (
    diamond_structure,
    mos2_structure,
    perovskite_structure,
    rocksalt_structure,
)
from crystalfig.export.exporter import Exporter, RenderOptions
from crystalfig.figure.builder import CrystalFigure
from crystalfig.figure.grid import FigureGrid
from crystalfig.io.loader import load_structure

ROOT = Path(__file__).parent.parent
OUT = ROOT / "gallery" / "visual_review_v4"
DATA = Path(__file__).parent / "data" / "visual_regression"


def structures():
    return {
        "BaTiO3": perovskite_structure(),
        "MoS2": mos2_structure(),
        "Si": diamond_structure(conventional=True),
        "NaCl": rocksalt_structure(conventional=True),
        "MgAl2O4": load_structure(DATA / "mg_al2o4_normal_spinel.cif"),
        "LiFePO4": load_structure(DATA / "li_fepo4_mp_workshop.cif"),
        "alpha_SiO2": load_structure(DATA / "alpha_quartz_cod5000035.cif"),
    }


def main():
    from PIL import Image, ImageDraw

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"render": {"backend": "pyvista", "width_mm": 89, "dpi": 450}, "structures": {}}
    for name, structure in structures().items():
        item = {"formula": structure.formula, "styles": {}, "candidate_views": []}
        reference_figure = CrystalFigure(structure).publication(intent="auto")
        reference_scene = reference_figure.build_scene()
        reference_view = {
            "label": reference_scene.metadata.get("camera_used"),
            "score": reference_scene.metadata.get("camera_score"),
            "score_components": reference_scene.metadata.get("camera_score_components"),
            "direction": reference_scene.metadata.get("view_dir_cartesian"),
        }
        for style in ("old", "journal_clean", "journal_vivid"):
            if style == "old":
                figure, scene = reference_figure, reference_scene
            else:
                figure = CrystalFigure(structure).publication(
                    intent="auto", style=style, camera=reference_figure.camera,
                )
                scene = figure.build_scene()
            image = OUT / f"{name}_{'old' if style == 'nature' else style}.png"
            figure.export(image, backend="pyvista", width=89, dpi=450, transparent=False)
            polyhedra = scene.get_group("polyhedra")
            display = structure.make_supercell(figure._display_plan.repeat)
            environments = Counter(
                f"{display.sites[poly.center_site].dominant_element}{len(poly.vertex_metadata)}"
                for poly in (polyhedra.primitives if polyhedra else [])
            )
            item["styles"][style] = {
                "path": image.name,
                "intent": figure._display_plan.intent,
                "repeat": list(figure._display_plan.repeat),
                "camera": reference_view["label"],
                "camera_score": reference_view["score"],
                "camera_score_components": reference_view["score_components"],
                "camera_parameters": {
                    "target": figure.camera.target.tolist(),
                    "distance": figure.camera.distance,
                    "elevation": figure.camera.elevation,
                    "azimuth": figure.camera.azimuth,
                    "roll": figure.camera.roll,
                    "projection": figure.camera.projection,
                    "padding": figure.camera.padding,
                },
                "camera_direction": reference_view["direction"],
                "display_budget": figure._display_budget,
                "coordination_environments": dict(environments),
                "polyhedron_count": sum(environments.values()),
                "render_quality": figure.theme.render_quality,
            }
            if style == "journal_vivid":
                candidates = figure.report_view_candidates(scene)
                for rank, candidate in enumerate(candidates[:3], start=1):
                    camera = figure._scored_view_candidates(scene)[rank - 1][1]
                    view_path = OUT / f"{name}_candidate_{rank}.png"
                    Exporter(scene, figure.theme, camera=camera).export(
                        view_path, fmt="png", backend="pyvista",
                        options=RenderOptions(width=89, dpi=300, transparent=False),
                    )
                    item["candidate_views"].append({**candidate, "path": view_path.name})
        if name == "MoS2":
            for view_name in ("layered_side", "layered_oblique"):
                figure = CrystalFigure(structure).publication(intent="layered", style="journal_vivid")
                scene = figure.build_scene()
                candidates = figure._scored_view_candidates(scene)
                selected = next(camera for report, camera in candidates if report["label"].startswith(view_name))
                view_figure = CrystalFigure(structure).publication(
                    intent="layered", style="journal_vivid", camera=selected,
                    display_budget={"max_atoms": 24, "max_bonds": 120},
                )
                path = OUT / f"MoS2_{view_name}.png"
                view_figure.export(path, backend="pyvista", width=89, dpi=450, transparent=False)
                item[view_name] = {"path": path.name, "camera": view_name}
        manifest["structures"][name] = item

    panels = FigureGrid(
        rows=1, cols=4, backend="hybrid", width_mm=183, height_mm=55,
        panel_gap_mm=2.2, background="white", legend_mode="panel", caption=None,
        font_size_pt=7,
    )
    selected = structures()
    for col, (name, title) in enumerate((
        ("BaTiO3", "BaTiO₃ · TiO₆ coordination"),
        ("MoS2", "MoS₂ · layered S–Mo–S"),
        ("Si", "Si · covalent network"),
        ("NaCl", "NaCl · ionic framework"),
    )):
        panels.add_structure(
            selected[name], 0, col, style="journal_vivid", title=title, intent="auto",
        )
    panels.save(OUT / "paper_four_panel_v4.png", dpi=600)
    panels.save(OUT / "paper_four_panel_v4.pdf", dpi=600)

    for name, intent in (("BaTiO3", "coordination"), ("Si", "network")):
        CrystalFigure(selected[name]).publication(
            intent=intent, style="journal_vivid",
        ).export(OUT / f"{name}_89mm_600dpi.png", backend="pyvista", width=89, dpi=600)

    (OUT / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # A compact, high-resolution review sheet for quick visual comparison.
    files = [OUT / f"{name}_{variant}.png" for name in structures() for variant in ("old", "journal_clean", "journal_vivid")]
    thumbs = []
    for path in files:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((390, 300))
            panel = Image.new("RGB", (410, 330), "white")
            panel.paste(image, ((410 - image.width) // 2, 5))
            ImageDraw.Draw(panel).text((10, 308), path.stem, fill="#25313a")
            thumbs.append(panel)
    sheet = Image.new("RGB", (410 * 3, 330 * len(structures())), "#edf0f2")
    for index, panel in enumerate(thumbs):
        sheet.paste(panel, ((index % 3) * 410, (index // 3) * 330))
    sheet.save(OUT / "comparison_contact_sheet.png")


if __name__ == "__main__":
    main()
