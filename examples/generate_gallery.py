"""Generate gallery examples for crystalfig."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystalfig.diagrams.equivariant import EquivariantGNNDiagram
from crystalfig.examples.presets import (
    cesium_chloride_structure,
    diamond_structure,
    graphite_structure,
    mos2_structure,
    perovskite_structure,
    rocksalt_structure,
    rutile_structure,
    triclinic_structure,
    wurtzite_structure,
    zincblende_zns_structure,
)
from crystalfig.figure.builder import CrystalFigure
from crystalfig.geometry.reciprocal import BrillouinZone
from crystalfig.io.loader import load_structure
from crystalfig.scene.camera import Camera

GALLERY_DIR = Path(__file__).parent.parent / "gallery"
BASELINE_DIR = GALLERY_DIR / "baseline"


def generate_baseline():
    """Render fixed-camera PNG/SVG references and a small reproducibility manifest."""
    import numpy as np
    from PIL import Image, ImageDraw

    from crystalfig.neighbors.strategies import CutoffStrategy
    from crystalfig.styles.palette import get_palette

    structures = [
        ("nacl", "NaCl rock salt", rocksalt_structure(), "Na", "Cl", 3.0),
        ("si", "Si diamond", diamond_structure(), "Si", "Si", 2.6),
        ("batio3", "BaTiO3 perovskite", perovskite_structure(a=3.95, c=4.05), "Ti", "O", 2.5),
        ("rutile_tio2", "Rutile TiO2", rutile_structure(), "Ti", "O", 2.1),
        ("zno", "Wurtzite ZnO", wurtzite_structure(), "Zn", "O", 2.2),
        ("mos2", "2H-MoS2", mos2_structure(), "Mo", "S", 2.8),
        ("triclinic_ab", "Triclinic P1 AB", triclinic_structure(), None, None, None),
    ]
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    palette = get_palette("publication")
    manifest = {
        "width_mm": 89.0,
        "dpi": 300,
        "transparent": True,
        "backend_by_extension": {"png": "MatplotlibRenderer", "svg": "SvgRenderer"},
        "camera": {"elevation": 25.0, "azimuth": 45.0, "padding": 0.12},
        "display_boundary": "cell_complete",
        "structures": {},
    }
    panels = []

    for name, label, structure, center_element, neighbor_element, cutoff in structures:
        camera = Camera(elevation=25.0, azimuth=45.0, auto_fit=True, padding=0.12)
        figure = CrystalFigure(structure)
        figure.camera = camera
        figure.show_unit_cell().boundary_mode("connected").add_bonds("cutoff", cutoff=cutoff or 2.5)
        figure.export(BASELINE_DIR / f"{name}.png", width=89.0, transparent=True)
        figure.export(BASELINE_DIR / f"{name}.svg", width=89.0, transparent=True)

        image = Image.open(BASELINE_DIR / f"{name}.png").convert("RGBA")
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        occupancy = 0.0 if bbox is None else max(
            (bbox[2] - bbox[0]) / image.width,
            (bbox[3] - bbox[1]) / image.height,
        )
        pixels = np.asarray(image, dtype=np.uint8)
        foreground = pixels[:, :, 3] > 0
        luminance = (
            0.2126 * pixels[:, :, 0]
            + 0.7152 * pixels[:, :, 1]
            + 0.0722 * pixels[:, :, 2]
        )[foreground]

        coordination = None
        if center_element:
            center = structure.indices_of_species(center_element)[0]
            bonds = CutoffStrategy(cutoff=cutoff).get_bonds(structure)
            coordination = sum(
                1 for bond in bonds if bond.i == center or bond.j == center
            )
        else:
            bonds = CutoffStrategy(cutoff=cutoff or 2.5).get_bonds(structure)
        manifest["structures"][name] = {
            "label": label,
            "formula": structure.formula,
            "lattice_lengths_angstrom": [round(float(x), 5) for x in structure.lattice.lengths],
            "lattice_angles_degrees": [round(float(x), 5) for x in structure.lattice.angles],
            "num_sites": structure.num_sites,
            "reference_center": center_element,
            "reference_neighbor": neighbor_element,
            "cutoff_angstrom": cutoff if cutoff is not None else 2.5,
            "bond_count": len(bonds),
            "periodic_bond_count": sum(bool(bond.jimage != (0, 0, 0)) for bond in bonds),
            "reference_coordination": coordination,
            "png_pixels": list(image.size),
            "png_size_mm": [round(image.width * 25.4 / 300, 3), round(image.height * 25.4 / 300, 3)],
            "rendered_content_occupancy_max_axis": round(occupancy, 4),
            "touches_image_edge": bbox is not None and (
                bbox[0] == 0 or bbox[1] == 0 or bbox[2] >= image.width or bbox[3] >= image.height
            ),
            "foreground_luminance_std": round(float(luminance.std()), 3),
            "alpha_bbox": bbox,
            "element_colors": {
                site.dominant_element: palette.hex(site.dominant_element)
                for site in structure.sites
            },
        }
        thumb = image.copy()
        thumb.thumbnail((360, 300))
        panel = Image.new("RGBA", (380, 340), "white")
        panel.alpha_composite(thumb, ((380 - thumb.width) // 2, 8))
        ImageDraw.Draw(panel).text((10, 315), label, fill="black")
        panels.append(panel.convert("RGB"))

    sheet = Image.new("RGB", (760, 1020), "#eeeeee")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 380, (index // 2) * 340))
    sheet.save(BASELINE_DIR / "contact_sheet.png")
    (BASELINE_DIR / "metrics.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def generate_pyvista_baseline():
    """Render a material-agnostic 10-structure publication showcase."""
    from PIL import Image, ImageDraw

    from crystalfig.export.preflight import preflight_export

    output_dir = GALLERY_DIR / "showcase"
    output_dir.mkdir(parents=True, exist_ok=True)
    structures = [
        ("rocksalt", rocksalt_structure(conventional=True)),
        ("diamond", diamond_structure(conventional=True)),
        ("perovskite", perovskite_structure(a=3.95, c=4.05)),
        ("rutile", rutile_structure()),
        ("wurtzite", wurtzite_structure()),
        ("layered_mos2", mos2_structure()),
        ("triclinic_p1", triclinic_structure()),
        ("cesium_chloride", cesium_chloride_structure()),
        ("zincblende_zns", zincblende_zns_structure()),
        ("graphite", graphite_structure()),
        (
            "monoclinic_tenorite",
            load_structure(Path(__file__).parent / "data" / "tenorite_amcsd_18822.cif"),
        ),
    ]
    manifest = {
        "width_mm": 89,
        "dpi": 300,
        "quality": "publication",
        "structure_sources": {
            "monoclinic_tenorite": {
                "record": "American Mineralogist Crystal Structure Database 18822",
                "record_url": "https://larixite.seescience.org/cifs/18822",
                "citation": "S. Åsbrink and A. Waśkowska, J. Phys.: Condens. Matter 3, 8173–8180 (1991), doi:10.1088/0953-8984/3/42/012",
            }
        },
        "structures": {},
    }
    panels = []
    for name, structure in structures:
        figure = CrystalFigure.from_structure(structure).publication().quality("publication")
        outputs = {
            "png": figure.export(
                output_dir / f"{name}.png", backend="pyvista", width=89, dpi=300, transparent=True
            ),
            "svg": figure.export(output_dir / f"{name}.svg", backend="svg", width=89),
            "pdf": figure.export(output_dir / f"{name}.pdf", backend="matplotlib", width=89),
        }
        image_path = output_dir / f"{name}.png"
        with Image.open(image_path) as source:
            panel_image = source.convert("RGBA")
            panel_image.thumbnail((420, 300))
            panel = Image.new("RGBA", (450, 340), "white")
            panel.alpha_composite(panel_image, ((450 - panel_image.width) // 2, 5))
            ImageDraw.Draw(panel).text((12, 315), f"{name} · {structure.formula}", fill="#263442")
            panels.append(panel.convert("RGB"))
        manifest["structures"][name] = {
            "formula": structure.formula,
            "sites": structure.num_sites,
            "lattice_lengths_angstrom": [round(float(value), 5) for value in structure.lattice.lengths],
            "selected_view": figure.report_view_candidates(figure.build_scene())[0],
            "diagnostics": figure.diagnostics,
            "exports": {
                suffix: preflight_export(
                    str(output_dir / f"{name}.{suffix}"),
                    expected_width_mm=89,
                    expected_dpi=300 if suffix == "png" else None,
                    transparent=True if suffix == "png" else None,
                    vector_status=result.vector_status,
                )
                for suffix, result in outputs.items()
            },
        }
    rows = (len(panels) + 1) // 2
    contact = Image.new("RGB", (900, rows * 340), "#edf1f4")
    for index, panel in enumerate(panels):
        row = index // 2
        x = 225 if len(panels) % 2 and index == len(panels) - 1 else (index % 2) * 450
        contact.paste(panel, (x, row * 340))
    contact.save(output_dir / "contact_sheet.png")
    (output_dir / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate reproducible crystal structure galleries")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--baseline-only", action="store_true")
    modes.add_argument("--showcase-only", action="store_true")
    modes.add_argument("--reference-only", action="store_true")
    args = parser.parse_args(argv)
    GALLERY_DIR.mkdir(exist_ok=True)
    if args.baseline_only:
        generate_baseline()
        return
    if args.showcase_only:
        generate_pyvista_baseline()
        return
    if args.reference_only:
        generate_reference_gallery()
        return
    generate_baseline()
    generate_pyvista_baseline()
    generate_reference_gallery()


def generate_reference_gallery():

    # Publication defaults: no legend, no axes unless needed.
    def _pub(fig):
        return fig.theme.show_legend is False and fig.theme.show_axes is False

    # 1. NaCl rocksalt ball-and-stick (conventional cell is more intuitive)
    fig = (
        CrystalFigure(rocksalt_structure(conventional=True))
        .show_unit_cell()
        .add_bonds("crystalnn")
    )
    fig.export(GALLERY_DIR / "01_rocksalt_ballstick.pdf")
    fig.export(GALLERY_DIR / "01_rocksalt_ballstick.svg")
    fig.export(GALLERY_DIR / "01_rocksalt_ballstick.png", width=80.0, transparent=True)

    # 2. Diamond Si with tetrahedral bonds (conventional cell)
    fig = (
        CrystalFigure(diamond_structure(conventional=True))
        .show_unit_cell()
        .add_bonds("covalent")
    )
    fig.view([1, 1, 1])
    fig.export(GALLERY_DIR / "02_diamond_tetrahedral.pdf")
    fig.export(GALLERY_DIR / "02_diamond_tetrahedral.png", width=90.0, transparent=True)

    # 3. BaTiO3 perovskite with TiO6 octahedron and polarization
    # Use the polyhedron preset so A-site atoms do not overwhelm the cage.
    struct = perovskite_structure(a=3.95, c=4.05)
    fig = (
        CrystalFigure(struct, theme="publication_polyhedra")
        .show_unit_cell()
        .add_polyhedra("Ti", strategy="cutoff", opacity=0.22)
        .add_vector(1, [0, 0, 0.4], color="amber", scale=1.0)
    )
    fig.camera = Camera(elevation=22.0, azimuth=35.0)
    fig.export(GALLERY_DIR / "03_perovskite_octahedron.pdf")
    fig.export(GALLERY_DIR / "03_perovskite_octahedron.png", width=90.0, transparent=True)
    fig.export_tikz_pdf(GALLERY_DIR / "03_perovskite_octahedron_tikz.pdf")

    # 4. Rutile TiO2 distorted octahedra
    fig = (
        CrystalFigure(rutile_structure(), theme="publication_polyhedra")
        .show_unit_cell()
        .add_polyhedra("Ti", strategy="cutoff", opacity=0.20)
    )
    fig.view([2, 2, 1])
    fig.export(GALLERY_DIR / "04_rutile_distorted.pdf")
    fig.export(GALLERY_DIR / "04_rutile_distorted.png", width=90.0, transparent=True)

    # 5. Wurtzite ZnO hexagonal: expand in the a-b plane to show the tetrahedral network
    fig = (
        CrystalFigure(wurtzite_structure())
        .supercell((2, 2, 1))
        .show_unit_cell()
        .add_bonds("cutoff", cutoff=2.6)
    )
    fig.view([1, 1, 1])
    fig.export(GALLERY_DIR / "05_wurtzite_hexagonal.pdf")
    fig.export(GALLERY_DIR / "05_wurtzite_hexagonal.png", width=90.0, transparent=True)

    # 6. MoS2 layered: modest side view of a (2,2,1) supercell so both the
    # in-plane honeycomb and the S-Mo-S layer stacking are readable.
    fig = (
        CrystalFigure(mos2_structure())
        .supercell((2, 2, 1))
        .show_unit_cell()
        .add_bonds("cutoff", cutoff=2.8)
    )
    fig.theme.atom_radius_scale = 0.22
    fig.theme.bond_width = 0.045
    fig.camera = Camera(elevation=38.0, azimuth=-25.0)
    fig.export(GALLERY_DIR / "06_mos2_layered.pdf")
    fig.export(GALLERY_DIR / "06_mos2_layered.png", width=90.0, transparent=True)

    # 7. Equivariant GNN diagram
    diagram = EquivariantGNNDiagram()
    tex = diagram.to_tikz(standalone=True)
    out_tex = GALLERY_DIR / "07_equivariant_gnn.tex"
    out_tex.write_text(tex, encoding="utf-8")
    from crystalfig.export.latex import LatexCompiler
    latex_engine = LatexCompiler.detect_engine()
    if latex_engine:
        compiler = LatexCompiler(engine=latex_engine)
        compiler.compile(tex, str(GALLERY_DIR / "07_equivariant_gnn.pdf"))
    else:
        print("Skipping optional TikZ PDF: no LaTeX engine is installed.")

    # 8. Diamond Si rendered with true Matplotlib 3D projection
    fig = (
        CrystalFigure(diamond_structure(conventional=True))
        .show_unit_cell()
        .add_bonds("covalent")
    )
    fig.export_3d(GALLERY_DIR / "08_diamond_3d.png", width=120.0, transparent=True)

    # 9. Brillouin zone
    bz = BrillouinZone.from_lattice(rocksalt_structure().lattice)
    print(f"Brillouin zone vertices: {len(bz.vertices)}")

    print(f"Gallery generated in {GALLERY_DIR}")


if __name__ == "__main__":
    main()
