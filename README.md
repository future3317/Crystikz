# crystalfig

Publication-grade crystal structure visualization toolkit for materials science and equivariant geometric deep learning.

`crystalfig` converts crystallographic data (CIF, POSCAR, pymatgen `Structure`, ASE `Atoms`) into publication-ready figures via multiple backends:

- **MatplotlibRenderer** — vector PDF/SVG/EPS/PGF and high-DPI PNG/TIFF.
- **SvgRenderer** — pure SVG with vector glossy shading, gradient bonds, and translucent polyhedra.
- **TikzRenderer** — pure LaTeX TikZ for font-uniform papers.

The core pipeline is:

```text
Crystal data (CIF / POSCAR / pymatgen / ASE)
    → canonical internal model (CrystalStructure)
    → analysis + geometry construction
    → backend-independent Scene
    → renderer
    → publication-ready PDF/SVG/TEX/PNG
```

## Quick start

### Install

```bash
pip install crystalfig
```

For development:

```bash
git clone https://github.com/future3317/Crystikz
cd Crystikz
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install crystalfig[ase,reciprocal,latex]
```

### One-line: CIF → PDF

```python
from crystalfig import plot_structure

fig = plot_structure("BaTiO3.cif")
fig.export("BaTiO3.pdf")
```

### Common paper workflow

```python
from crystalfig import CrystalFigure

fig = (
    CrystalFigure.from_file("BaTiO3.cif")
    .conventional_cell()
    .view(direction=[1, 1, 0])
    .show_unit_cell()
    .add_bonds(strategy="crystalnn")
    .add_polyhedra(centers="Ti")
    .style("publication_polyhedra")
)

fig.export("BaTiO3.pdf")
fig.export("BaTiO3.svg")
```

### Embed in a normal Matplotlib figure

```python
import matplotlib.pyplot as plt
from crystalfig import CrystalFigure
from crystalfig.examples.presets import rocksalt_structure

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

CrystalFigure(rocksalt_structure()).quick().draw(axes[0])
axes[1].plot([0, 1, 2], [0, 1, 0])

plt.tight_layout()
plt.savefig("combined.pdf")
```

### CLI

```bash
# Render a structure
crystalfig render BaTiO3.cif -o BaTiO3.pdf --bonds crystalnn --polyhedra Ti

# Inspect structure metadata
crystalfig inspect BaTiO3.cif

# Check environment
crystalfig doctor
```

## Features

### Structure input

- CIF, POSCAR (with or without extension), CSSR, JSON, YAML
- pymatgen `Structure`
- ASE `Atoms` (optional)

### Structure analysis

- Conventional / primitive cells via spglib
- Supercells (diagonal integer tuple or general 3×3 integer matrix)
- Reciprocal lattice and first Brillouin zone (basic Voronoi construction)

### Visual elements

- Atoms: glossy shaded spheres, flat circles, wireframe
- Bonds: CrystalNN, cutoff, covalent-radii, ASE natural cutoffs; split-color or uniform
- Coordination polyhedra: automatic ConvexHull faces
- Unit cell / supercell edges with front/back styling
- Vectors (polarization, force, magnetic moment, phonon arrows)
- Miller plane intersection polygons
- Manual site/species styling overrides

### Backends

- **MatplotlibRenderer**: true vector PDF/SVG/EPS/PGF, high-DPI PNG/TIFF
- **SvgRenderer**: pure SVG publication output
- **TikzRenderer**: standalone TikZ with centralized library management
- **Matplotlib3DRenderer**: true 3D projection preview (hybrid vector/raster)

### Export

- `.pdf`, `.svg`, `.eps`, `.png`, `.tiff`, `.pgf`, `.tex`
- Figure size in mm
- Transparent backgrounds

## Current limitations

The following items are **not implemented** in the current release:

- Volumetric isosurfaces (CHGCAR, cube)
- Trajectory / multi-frame export
- Magnetic symmetry (spglib magnetic groups)
- Advanced automatic label placement / collision avoidance
- Multi-panel compositor (`FigureGrid`); use `CrystalFigure.draw(ax=...)` with normal Matplotlib subplots instead
- PyVista/Fresnel/POV-Ray advanced 3D rendering is not included
- Brillouin-zone k-paths require the optional `seekpath` dependency

## Scientific conventions

- Lattice matrix stores `a, b, c` as rows (pymatgen/ASE convention).
- Fractional coordinates are column vectors: `cart = frac @ lattice_matrix`.
- Reciprocal vectors include the `2π` factor.
- `[uvw]` denotes a direct-space direction; `(hkl)` denotes a reciprocal-space normal.
- Bond detection is model-dependent; no universal chemical-correctness is claimed.

## Project layout

```text
crystalfig/
    model/          # canonical structure, lattice, site
    io/             # loaders and adapters
    geometry/       # periodic images, polyhedra, Miller planes, BZ
    neighbors/      # bond strategies
    scene/          # backend-independent scene graph and camera
    renderers/      # Matplotlib, SVG, TikZ, Matplotlib3D
    styles/         # palettes, radii, publication themes
    figure/         # high-level CrystalFigure fluent API
    export/         # compiler, preflight, unified exporter
    diagrams/       # equivariant GNN architecture diagrams
    cli/            # command-line interface
```

## Development

```bash
# Run tests
python -m pytest tests -q

# Lint
python -m ruff check .

# Build wheel
python -m build --wheel

# Generate gallery
python examples/generate_gallery.py
python examples/make_contact_sheet.py
```

## License

MIT
