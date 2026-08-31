# crystalfig

Publication-grade crystal structure visualization toolkit for materials science and equivariant geometric deep learning.

`crystalfig` converts crystallographic data (CIF, POSCAR, pymatgen `Structure`, ASE `Atoms`) into publication-ready figures via multiple backends: Matplotlib (PDF/SVG/PNG/EPS/PGF), TikZ/PGF (pure LaTeX vector), and optional PyVista/VTK for advanced 3D rendering.

The core philosophy is:

```text
Crystal data / Structure
→ canonical internal model
→ analysis + geometry construction
→ backend-independent Scene
→ multiple rendering backends
→ publication-ready PDF/SVG/PGF/TikZ/PNG/TIFF
```

## Quick start

### Install

```bash
pip install dist/crystalfig-0.1.0-py3-none-any.whl
```

For optional dependencies:

```bash
pip install crystalfig[ase,reciprocal,pyvista]
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
    .add_lattice_axes()
    .style("publication_muted")
)

fig.export("BaTiO3.pdf")
fig.export("BaTiO3.svg")
fig.save_recipe("BaTiO3.figure.yaml")
```

### CLI

```bash
# Render a structure
crystalfig render BaTiO3.cif -o BaTiO3.pdf --bonds crystalnn --polyhedra Ti

# Inspect structure metadata
crystalfig inspect BaTiO3.cif

# Check environment
crystalfig doctor

# Preflight an exported PDF
crystalfig preflight BaTiO3.pdf

# Batch from YAML
crystalfig batch figures.yaml
```

## Features

### Structure input

- CIF, POSCAR, CSSR, JSON, YAML
- pymatgen `Structure`
- ASE `Atoms` (optional)

### Structure analysis

- Conventional / primitive cells via spglib
- Supercells (diagonal or 3×3 matrix)
- Space group, Wyckoff positions, symmetry-equivalent sites
- Reciprocal lattice and first Brillouin zone

### Visual elements

- Atoms: shaded spheres, flat circles, wireframe, space-filling
- Bonds: CrystalNN, cutoff, covalent-radii, ASE natural cutoffs
- Coordination polyhedra: automatic ConvexHull faces, no hand-written face lists
- Unit cell / supercell edges with front/back styling
- Crystallographic axes
- Miller planes and `[uvw]` directions
- Polarization, force, magnetic-moment, phonon arrows
- Site-property colormaps
- Defect / vacancy highlights

### Backends

- **MatplotlibRenderer**: true vector PDF/SVG/EPS/PGF, high-DPI PNG/TIFF
- **TikzRenderer**: pure LaTeX TikZ with centralized library management
- **PyVistaRenderer**: optional 3D meshes / isosurfaces (placeholder)

### Export

- `.pdf`, `.svg`, `.eps`, `.png`, `.tiff`, `.pgf`, `.tex`
- Figure size in mm/cm/in/pt
- Transparent backgrounds
- Recipe YAML/JSON for full reproducibility

## Scientific conventions

- Lattice matrix stores `a, b, c` as rows (pymatgen/ASE convention).
- Fractional coordinates are column vectors: `cart = frac @ lattice_matrix`.
- Reciprocal vectors include the `2π` factor.
- `[uvw]` denotes a direct-space direction; `(hkl)` denotes a reciprocal-space normal.
- Bond detection is model-dependent; no universal chemical-correctness is claimed.

## Project layout

```text
crystalfig/
    model/          # canonical structure, lattice, site, properties
    io/             # loaders and adapters
    geometry/       # periodic images, polyhedra, Miller planes, BZ
    neighbors/      # bond strategies (CrystalNN, cutoff, covalent, ASE)
    scene/          # backend-independent scene graph and camera
    renderers/      # Matplotlib, TikZ, optional PyVista
    styles/         # palettes, radii, publication themes
    figure/         # high-level CrystalFigure fluent API
    export/         # compiler, preflight, unified exporter
    diagrams/       # equivariant GNN architecture diagrams
    cli/            # command-line interface
    compat.py       # legacy crystal_tikz compatibility aliases
```

## Development

```bash
# Run tests
python -m pytest tests -v

# Lint
python -m ruff check .

# Build wheel
python -m build --wheel

# Generate gallery
python examples/generate_gallery.py
```

## License

MIT
