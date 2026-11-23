# crystalfig

**Publication-oriented crystal structure figures for Python.** `crystalfig` turns CIF files and
crystal structures into reusable ball-and-stick, coordination-polyhedron, layered, and unit-cell
visualizations. A shared structure and scene model feeds editable vector renderers and an optional
depth-buffered 3D renderer.

![Four crystals rendered in PyVista with vector titles and element keys](gallery/publication_review/paper_four_panel.png)

[Journal and VESTA benchmarks](gallery/publication_review/comparison_contact_sheet.png) · [Bond and framing comparisons](gallery/publication_review/coordination_comparison.png) · [Same scene in two backends](gallery/publication_review/backend_comparison.png) · [14-structure diagnostic gallery](gallery/showcase/contact_sheet.png)

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/future3317/Crystikz/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/future3317/Crystikz/actions/workflows/ci.yml)

## Install

```bash
python -m pip install "crystalfig @ git+https://github.com/future3317/Crystikz.git"
```

Version 0.2.0 is prepared but not yet published to PyPI. When the project is published there,
the installation command will become `python -m pip install crystalfig`.

Install optional integrations only when needed:

```bash
python -m pip install "crystalfig[render3d] @ git+https://github.com/future3317/Crystikz.git"  # PyVista/VTK
python -m pip install "crystalfig[ase] @ git+https://github.com/future3317/Crystikz.git"       # ASE
python -m pip install "crystalfig[reciprocal] @ git+https://github.com/future3317/Crystikz.git" # seekpath
python -m pip install "crystalfig[latex] @ git+https://github.com/future3317/Crystikz.git"     # PDF helpers
python -m pip install "crystalfig[all] @ git+https://github.com/future3317/Crystikz.git"       # all extras
```

The base installation includes CIF/POSCAR loading, pymatgen structures, Matplotlib and SVG
rendering. PyVista and VTK are intentionally optional because they add a large 3D runtime.

## Quick start

Render a CIF as a high-resolution 3D PNG or as a PDF with a 3D image and editable annotations.
Install the `render3d` extra for these outputs.

```python
from crystalfig import CrystalFigure, DisplayConfig, FigureConfig

figure = CrystalFigure.from_file("structure.cif").configure(FigureConfig(
    display=DisplayConfig(intent="auto", region="auto"),
    style="journal_clean",
))
figure.export("structure.png")  # 89 mm wide, 600 DPI
figure.export("structure.pdf")  # Hybrid PDF
```

Use the vector backend for editable geometry or a base installation.

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

The display planner selects a network, coordination, layered, ionic or minimal view. It chooses a
finite periodic region, keeps coordination shells complete by default and fits the camera to the
scene. For an explicit coordination motif, choose the center species or site indices.

```python
figure = CrystalFigure.from_file("BaTiO3.cif").configure(FigureConfig(
    display=DisplayConfig(intent="coordination", region="motif", centers=("Ti",)),
    style="vesta_coordination",
))
figure.export("motif.png", transparent=True)
```

`DisplayConfig` supports `region="cell"` for one cell, `region="motif"` for a local view and a
tuple such as `region=(2, 2, 1)` for a finite supercell. `coordination_shell="complete"` retains
periodic ligands across the cell boundary. `show_partial` explicitly clips the hull and marks
artificial faces. `CrystalNN` is a candidate neighbor model. Choose a cutoff or manual neighbors
when the scientific interpretation requires a different shell.

Use `framing="unit_cell"` for a complete cell, `framing="local_coordination"` for a single
periodic neighborhood, and `framing="motif"` for a representative connected fragment.
`framing_site` optionally specifies a zero-based site index for local coordination. For dense
networks, automatic framing can select a motif when the full cell exceeds the display budget.

Set `bond_display_policy` in `DisplayConfig` or `publication()` to control chemical bond lines.
The coordination default, `center_ligand_only`, keeps the sticks for the displayed shells and
draws them thinner and lighter. `all_bonds` draws all bonds in the displayed region.
`polyhedron_edges_only` uses the existing polyhedron outlines without sticks, while `hidden`
hides chemical bonds in any view. Polyhedron appearance is controlled by the selected style.

Create a polyhedron-led view when coordination is the intended scientific message:

```python
from crystalfig import CrystalFigure, FigureGrid

figure = CrystalFigure.from_file("structure.cif").publication(
    intent="coordination",
    coordination_shell="complete",
    palette="publication",
    camera="optimize",
    background="white",
)
figure.save("structure.svg")  # PNG and PDF are also supported
```

Automatic intent and center selection are geometry/chemistry heuristics, not a chemical assignment;
the chosen strategy and any fallback are recorded in diagnostics and scene metadata. The scene
metadata distinguishes the camera actually used from the recommended candidates. The legacy
`boundary="cell"` option remains available for explicitly clipped views; the new complete-shell
policy does not clip a coordination polyhedron to the cell.

Compose multiple structures with a shared Matplotlib figure:

```python
grid = FigureGrid(
    rows=1, cols=2, panel_labels=True, backend="hybrid",
    width_mm=183, height_mm=70, legend_mode="panel",
)
grid.add_structure(CrystalFigure.from_file("phase-a.cif"), 0, 0, title="Phase A", intent="auto")
grid.add_structure(CrystalFigure.from_file("phase-b.cif"), 0, 1, title="Phase B", intent="auto")
grid.save("phases.pdf", dpi=300)
```

`FigureGrid` supports `vector`, `pyvista`, and `hybrid` structure panels. Hybrid panels rasterize
the depth-buffered PyVista scene and keep panel labels, titles, legends, and captions as vector
overlays in the fixed-size output. `grid.last_export_metadata["vector_status"]` reports whether
the saved document is pure vector, hybrid, or raster.

## Choose an output backend

| Backend | Formats | Output | Use it for |
| --- | --- | --- | --- |
| `matplotlib` (default) | PDF, EPS, PGF, PNG, TIFF | Vector for PDF/EPS/PGF; raster for PNG/TIFF | Conventional paper figures and Matplotlib integration |
| `svg` | SVG | Editable vector | Editing paths and labels in a vector graphics editor |
| `tikz` | TEX/TIKZ | LaTeX drawing commands | Figures that should use the paper's LaTeX fonts |
| `pyvista` (optional) | PNG, hybrid PDF | High-resolution 3D image plus vector annotations in PDF | Complex 3D structures and mesh occlusion |
| `matplotlib3d` | PDF, SVG, EPS, PGF, PNG, TIFF | Projected 3D preview; hybrid for vector formats | Previewing the legacy Matplotlib 3D projection |

With `FigureConfig`, PyVista handles PNG and PDF by default. SVG and TikZ use their vector
renderers. The legacy fluent API continues to choose a backend from the output extension.

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

Publication presets pair atom and bond dimensions with separate element and coordination colors.

```python
figure = CrystalFigure.from_file("MoS2.cif").configure(FigureConfig(
    display=DisplayConfig(intent="layered"),
    style="layered_crystal",
))
figure.export("MoS2.png", backend="pyvista", width=89, dpi=600)
```

Choose `vesta_ballstick`, `vesta_coordination`, `journal_clean`, `journal_vivid` or
`layered_crystal`. The VESTA presets use larger ceramic spheres, stronger highlights and
thin cylindrical bonds. For custom appearance, use `crystalfig.styles.get_style()` and
`dataclasses.replace()` to adjust atom, bond, polyhedron, cell or lighting settings independently.
`CameraConfig(mode="fixed", elevation=30, azimuth=35)` records an explicit view. Existing
`FigureTheme` presets and crystallographic view presets are available through the fluent API.

## Command line

```bash
crystalfig render BaTiO3.cif -o BaTiO3.pdf --bonds crystalnn --polyhedra Ti --width 89
crystalfig render BaTiO3.cif -o motif.png --intent coordination --region motif --centers Ti --style vesta_coordination
crystalfig render dense.cif -o local.png --framing local_coordination --framing-site 0 --style vesta_coordination
crystalfig inspect BaTiO3.cif
crystalfig doctor
crystalfig preflight BaTiO3.pdf
```

`crystalfig render --help` lists rendering options. The Python API also supports YAML batch
rendering with `crystalfig batch figures.yaml`. Batch paths are relative to the YAML file:

```yaml
figures:
  - input: structures/sample.cif
    output: figures/sample.pdf
    style: journal_vivid
    display:
      framing: local_coordination
      neighbor_strategy: crystalnn
      bond_display_policy: center_ligand_only
    camera:
      mode: auto
    render:
      backend: pyvista
      width_mm: 89
      dpi: 600
```

## Capabilities

- Read CIF, POSCAR, CSSR, JSON and YAML structures; accept pymatgen `Structure` and optional ASE
  `Atoms` objects.
- Build conventional/primitive cells, supercells, periodic images, bonds, coordination polyhedra,
  Miller-plane intersections and basic Brillouin-zone geometry.
- Style atoms, bonds and sites; add vectors and labels; draw into an existing Matplotlib axes.
- Export at figure widths specified in millimetres, with configurable DPI and transparent
  backgrounds where supported by the selected renderer.
- Inspect the curated [four-panel hybrid figure](gallery/showcase/paper_four_panel.png), the
  [14-structure diagnostic showcase](gallery/showcase/contact_sheet.png), the
  [baseline diagnostic sheet](gallery/baseline/contact_sheet.png), and the
  [rendered format comparisons](gallery/showcase/).

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

Use Python 3.11 for development and validation.

```bash
git clone https://github.com/future3317/Crystikz.git
cd Crystikz
python -m pip install -e '.[dev]'
python -m pytest tests -q
python -m ruff check crystalfig tests examples
python -m build --wheel
python examples/generate_publication_review.py  # requires render3d and Poppler
```

For the release candidate wheel install, CLI, and export smoke checks, run
`python scripts/check_release.py`. See [PUBLISHING.md](PUBLISHING.md) for the v0.2.0 release
checklist and current PyPI status.

## License

MIT. See [LICENSE](LICENSE).
