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

    # 1. NaCl rocksalt ball-and-stick
    fig = CrystalFigure(rocksalt_structure()).quick()
    fig.export(GALLERY_DIR / "01_rocksalt_ballstick.pdf")
    fig.export(GALLERY_DIR / "01_rocksalt_ballstick.svg")
    fig.export(GALLERY_DIR / "01_rocksalt_ballstick.png", width=80.0, transparent=True)

    # 2. Diamond Si with tetrahedral bonds
    fig = CrystalFigure(diamond_structure()).add_bonds("covalent").show_unit_cell().show_axes()
    fig.view([1, 1, 1])
    fig.export(GALLERY_DIR / "02_diamond_tetrahedral.pdf")

    # 3. BaTiO3 perovskite with TiO6 octahedron and polarization
    struct = perovskite_structure(a=3.95, c=4.05)
    fig = (
        CrystalFigure(struct)
        .show_unit_cell()
        .show_axes()
        .add_bonds("covalent")
        .add_polyhedra("Ti", opacity=0.25)
        .add_vector(8, [0, 0, 0.4], color="amber", scale=1.0)
    )
    fig.export(GALLERY_DIR / "03_perovskite_octahedron.pdf")
    fig.export_tikz_pdf(GALLERY_DIR / "03_perovskite_octahedron_tikz.pdf")

    # 4. Rutile TiO2 distorted octahedra
    fig = (
        CrystalFigure(rutile_structure())
        .show_unit_cell()
        .add_bonds("covalent")
        .add_polyhedra("Ti")
    )
    fig.export(GALLERY_DIR / "04_rutile_distorted.pdf")

    # 5. Wurtzite ZnO hexagonal
    fig = CrystalFigure(wurtzite_structure()).quick().view([1, 0, 0])
    fig.export(GALLERY_DIR / "05_wurtzite_hexagonal.pdf")

    # 6. MoS2 layered
    fig = CrystalFigure(mos2_structure()).quick().view([0, 0, 1])
    fig.export(GALLERY_DIR / "06_mos2_layered.pdf")

    # 7. Equivariant GNN diagram
    diagram = EquivariantGNNDiagram()
    tex = diagram.to_tikz(standalone=True)
    out_tex = GALLERY_DIR / "07_equivariant_gnn.tex"
    out_tex.write_text(tex, encoding="utf-8")
    from crystalfig.export.latex import LatexCompiler
    compiler = LatexCompiler(engine=LatexCompiler.detect_engine() or "pdflatex")
    compiler.compile(tex, str(GALLERY_DIR / "07_equivariant_gnn.pdf"))

    # 8. Brillouin zone
    bz = BrillouinZone.from_lattice(rocksalt_structure().lattice)
    print(f"Brillouin zone vertices: {len(bz.vertices)}")

    print(f"Gallery generated in {GALLERY_DIR}")


if __name__ == "__main__":
    main()
