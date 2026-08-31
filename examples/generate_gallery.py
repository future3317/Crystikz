"""Generate gallery examples for crystalfig."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crystalfig.diagrams.equivariant import EquivariantGNNDiagram
from crystalfig.examples.presets import (
    diamond_structure,
    mos2_structure,
    perovskite_structure,
    rocksalt_structure,
    rutile_structure,
    wurtzite_structure,
)
from crystalfig.figure.builder import CrystalFigure
from crystalfig.geometry.reciprocal import BrillouinZone

GALLERY_DIR = Path(__file__).parent.parent / "gallery"


def main():
    GALLERY_DIR.mkdir(exist_ok=True)

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
    fig.view([1, 2, 3])
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

    # 6. MoS2 layered: modest in-plane expansion with a near-c-axis tilt so the
    # honeycomb in-plane network and layer stacking are both readable.
    fig = (
        CrystalFigure(mos2_structure())
        .supercell((2, 2, 2))
        .show_unit_cell()
        .add_bonds("cutoff", cutoff=2.8)
    )
    fig.view([2, 2, 5])
    fig.export(GALLERY_DIR / "06_mos2_layered.pdf")
    fig.export(GALLERY_DIR / "06_mos2_layered.png", width=90.0, transparent=True)

    # 7. Equivariant GNN diagram
    diagram = EquivariantGNNDiagram()
    tex = diagram.to_tikz(standalone=True)
    out_tex = GALLERY_DIR / "07_equivariant_gnn.tex"
    out_tex.write_text(tex, encoding="utf-8")
    from crystalfig.export.latex import LatexCompiler
    compiler = LatexCompiler(engine=LatexCompiler.detect_engine() or "pdflatex")
    compiler.compile(tex, str(GALLERY_DIR / "07_equivariant_gnn.pdf"))

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
