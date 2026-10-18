# crystalfig

**Publication-oriented crystal structure figures for Python.** `crystalfig` turns CIF files and
crystal structures into reusable ball-and-stick, coordination-polyhedron, layered, and unit-cell
visualizations. A shared structure and scene model feeds editable vector renderers and an optional
depth-buffered 3D renderer.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Install

```bash
python -m pip install crystalfig
```

Install optional integrations only when needed:

```bash
python -m pip install 'crystalfig[render3d]'  # PyVista/VTK 3D PNG rendering
python -m pip install 'crystalfig[ase]'       # ASE Atoms input and neighbor strategy
python -m pip install 'crystalfig[reciprocal]' # seekpath reciprocal-space tools
python -m pip install 'crystalfig[latex]'     # PDF-to-image helpers
python -m pip install 'crystalfig[all]'       # all optional runtime integrations
```

The base installation includes CIF/POSCAR loading, pymatgen structures, Matplotlib and SVG
rendering. PyVista and VTK are intentionally optional because they add a large 3D runtime.

## Quick start

Render a CIF as an editable vector PDF:

```python
from crystalfig import CrystalFigure

figure = (
    CrystalFigure.from_file("BaTiO3.cif", theme="polyhedra_clean")
    .add_polyhedra(centers="Ti", strategy="crystalnn")
    .show_unit_cell()
)
result = figure.export("BaTiO3.pdf", width=89)
print(result.vector_status)  # "pure"
```

Use the same API with a pymatgen `Structure` or (with the `ase` extra) ASE `Atoms`:

```python
from pymatgen.core import Structure
from crystalfig import CrystalFigure

structure = Structure.from_file("BaTiO3.cif")
figure = CrystalFigure.from_structure(structure).quick()
figure.export("BaTiO3.svg", width=89)
```

`CrystalFigure` is reusable and backend-independent: adjust the cell, view, bonds, polyhedra,
style and annotations before exporting. For example, `view([1, 1, 0])` sets a crystallographic
view direction, while `supercell((2, 2, 1))` expands the displayed structure.

## Choose an output backend

| Backend | Formats | Output | Use it for |
| --- | --- | --- | --- |
| `matplotlib` (default) | PDF, EPS, PGF, PNG, TIFF | Vector for PDF/EPS/PGF; raster for PNG/TIFF | Conventional paper figures and Matplotlib integration |
| `svg` | SVG | Editable vector | Editing paths and labels in a vector graphics editor |
| `tikz` | TEX/TIKZ | LaTeX drawing commands | Figures that should use the paper's LaTeX fonts |
| `pyvista` (optional) | PNG | High-resolution raster with 3D depth buffering | Complex 3D structures and mesh occlusion |
| `matplotlib3d` | PDF, SVG, EPS, PGF, PNG, TIFF | Projected 3D preview; hybrid for vector formats | Previewing the legacy Matplotlib 3D projection |

Select a backend explicitly when needed. The default backend is chosen from the output extension.

```python
figure.export("BaTiO3.png", backend="pyvista", width=89, dpi=600)
figure.export("BaTiO3.svg", backend="svg", width=89)
hybrid = figure.export_hybrid_pdf("BaTiO3-hybrid.pdf", width=89, dpi=600)
print(hybrid.vector_status)  # "hybrid": 3D image plus vector annotations
```

The SVG, TikZ, and Matplotlib vector paths are useful for editable figures. PyVista exports PNG;
`export_hybrid_pdf()` embeds its high-resolution image in a PDF with vector annotations. A hybrid
or raster result is not a pure-vector figure. PyVista requires a working VTK off-screen rendering
environment; unsupported transparency/depth-peeling capabilities are reported by the renderer.

## Styles and views

Built-in semantic themes provide starting points for common figure types:

```python
figure = (
    CrystalFigure.from_file("MoS2.cif", theme="layered_clean")
    .add_bonds(strategy="crystalnn")
    .view_preset("layer_oblique")
)
figure.export("MoS2.png", backend="pyvista", width=89, dpi=600)
```

Themes: `ballstick_clean`, `polyhedra_clean`, `layered_clean`, `overview_minimal`, plus the
`publication` family. View presets include `auto`, `isometric`, `along_a`, `along_b`, `along_c`,
`layer_top`, `layer_oblique`, and `polyhedron_focus`. Presets are editable defaults; the final
figure still depends on the structure, selected neighbors, view, labels and export size.

## Command line

```bash
crystalfig render BaTiO3.cif -o BaTiO3.pdf --bonds crystalnn --polyhedra Ti --width 89
crystalfig inspect BaTiO3.cif
crystalfig doctor
crystalfig preflight BaTiO3.pdf
```

`crystalfig render --help` lists rendering options. The Python API also supports YAML batch
rendering with `crystalfig batch figures.yaml`.

## Capabilities

- Read CIF, POSCAR, CSSR, JSON and YAML structures; accept pymatgen `Structure` and optional ASE
  `Atoms` objects.
- Build conventional/primitive cells, supercells, periodic images, bonds, coordination polyhedra,
  Miller-plane intersections and basic Brillouin-zone geometry.
- Style atoms, bonds and sites; add vectors and labels; draw into an existing Matplotlib axes.
- Export at figure widths specified in millimetres, with configurable DPI and transparent
  backgrounds where supported by the selected renderer.
- Inspect the generated examples and reproducible structure baseline in the
  [visual gallery](gallery/) and [baseline contact sheet](gallery/baseline/contact_sheet.png).

## Scientific and rendering notes

Bond detection depends on the selected strategy and input structure; it is not a universal
chemical-bond assignment. Check coordination and periodic images for the material and use case.
Vector renderers use projected drawing order and cannot provide full per-pixel 3D occlusion for
intersecting transparent geometry. Use PyVista for depth-buffered 3D output when that distinction
matters. PyVista output is raster; use SVG, TikZ or Matplotlib when editable vector geometry is
required.

`crystalfig` complements crystallographic editors such as VESTA, atomistic analysis tools such as
OVITO, and general 3D software such as Blender. It focuses on repeatable Python figure generation
from crystal structure data.

## Development

```bash
git clone https://github.com/future3317/Crystikz.git
cd Crystikz
python -m pip install -e '.[dev]'
python -m pytest tests -q
python -m ruff check crystalfig tests examples
python -m build --wheel
```

## License

MIT. See [LICENSE](LICENSE).
