# crystalfig Implementation Report

## 1. Architecture

`crystalfig` is organized as a multi-layer visualization toolkit:

```text
Input (CIF/POSCAR/pymatgen/ASE)
    → crystalfig.io adapters
    → crystalfig.model.CrystalStructure (canonical internal model)
    → crystalfig.geometry / crystalfig.neighbors (analysis)
    → crystalfig.scene.Scene + Camera (backend-independent scene graph)
    → crystalfig.renderers (Matplotlib, TikZ, optional PyVista)
    → crystalfig.export (PDF/SVG/PNG/TEX + preflight)
    → publication-ready output
```

The high-level `CrystalFigure` fluent API lives in `crystalfig.figure.builder`. The CLI is in `crystalfig.cli.main`.

## 2. Dependencies

### Required

- numpy, scipy, matplotlib, pymatgen, spglib, pyyaml

### Optional extras

- `ase` — ASE adapter and natural-cutoff neighbor strategy
- `reciprocal` — seekpath for Brillouin-zone k-paths
- `pyvista` — advanced 3D renderer (interface present, full implementation optional)
- `raytrace` — Fresnel/POV-Ray (not installed in target environment)
- `latex` — pdf2image helpers

## 3. Implemented Features

### Structure model

- `Lattice` with lengths/angles, fractional ↔ Cartesian conversion, reciprocal matrix
- `Site` with species, occupancy, properties (magmom, force, displacement)
- `CrystalStructure` with formula, supercell, transformations

### IO

- CIF, POSCAR, CSSR, JSON, YAML, XYZ/extxyz (via ASE)
- pymatgen `Structure` and ASE `Atoms` adapters

### Geometry

- Periodic image generation, nearest-image convention
- Coordination polyhedra via `scipy.spatial.ConvexHull`
- Miller plane / cell intersection
- First Brillouin zone via Voronoi construction

### Neighbors

- `CutoffStrategy`, `CovalentRadiiStrategy`, `CrystalNNStrategy`, `ASEStrategy`
- Bonds store site indices, jimage, distance

### Scene & Camera

- Primitives: Sphere, Cylinder/Bond, Line, Polyline, Polygon, Polyhedron, Arrow/Axis, Plane, Text, LegendItem
- `Camera` with elevation/azimuth/roll, orthographic/perspective, view along `[uvw]` or normal to `(hkl)`
- Depth sorting for painter's-algorithm rendering

### Renderers

- `MatplotlibRenderer`: PDF/SVG/EPS/PGF/PNG/TIFF, transparent backgrounds, exact figure size
- `TikzRenderer`: standalone TikZ with centralized library loading (`positioning`, `calc`, etc.)
- `PyVistaRenderer`: not fully implemented; kept as optional extension point

### Figure builder

- Fluent API: `.view()`, `.supercell()`, `.add_bonds()`, `.add_polyhedra()`, `.add_vector()`, `.add_miller_plane()`, `.style()`, `.export()`, `.save_recipe()`
- One-line helper `plot_structure(...)`

### CLI

- `crystalfig render`
- `crystalfig inspect`
- `crystalfig doctor`
- `crystalfig preflight`
- `crystalfig batch`

### Compatibility

- Legacy `crystal_tikz` names exposed via `crystalfig.compat` with `FutureWarning`

## 4. Tests

```bash
python -m pytest tests -q
```

Result: **37 passed, 1 warning** (spglib deprecation warning from pymatgen).

Coverage includes:

- Lattice round-trip for cubic, hexagonal, triclinic cells
- Site species/occupancy
- Structure supercell
- Periodic nearest-image wrapping
- ConvexHull polyhedra (octahedron, tetrahedron)
- Miller plane intersection
- Neighbor strategies
- pymatgen round-trip and CIF loading
- Matplotlib export (PDF/PNG)
- TikZ standalone generation and LaTeX compilation
- CrystalFigure fluent API and recipe serialization
- CLI render/inspect/doctor

## 5. Build & Smoke Test

```bash
python -m build --wheel
```

Built `dist/crystalfig-0.1.0-py3-none-any.whl` successfully.

Installed into a clean venv and verified:

```bash
.smoke_venv/Scripts/python -c "import crystalfig; print(crystalfig.__version__)"
.smoke_venv/Scripts/crystalfig doctor
```

Both succeeded.

## 6. Generated Examples

Gallery generated in `gallery/`:

- `01_rocksalt_ballstick.{pdf,svg,png}`
- `02_diamond_tetrahedral.pdf`
- `03_perovskite_octahedron.pdf`
- `03_perovskite_octahedron_tikz.pdf`
- `04_rutile_distorted.pdf`
- `05_wurtzite_hexagonal.pdf`
- `06_mos2_layered.pdf`
- `07_equivariant_gnn.{tex,pdf}`

## 7. Linting

```bash
python -m ruff check crystalfig tests examples
```

Result: **All checks passed!**

## 8. Compatibility

Old `crystal_tikz` APIs are available as deprecated aliases in `crystalfig.compat`:

- `LatticeBasis` → `Lattice`
- `Camera3D` → `Camera`
- `CrystalVisualizer` → `CrystalFigure`
- `build_perovskite`, `build_rutile`, `build_wurtzite`
- `EquivariantArchitectureVisualizer` → `EquivariantGNNDiagram`
- `compile_tikz_to_pdf`

## 9. Known Limitations

- `PyVistaRenderer` and `FresnelRenderer` are optional-extension points only; not fully implemented because the target environment lacks a display/GPU setup and the task prioritizes vector backends.
- Volumetric isosurfaces (CHGCAR, cube) are not implemented.
- Magnetic symmetry (spglib magnetic groups) is not implemented.
- Trajectory / multi-frame export is not implemented.
- Advanced depth occlusion for vector renderers uses a painter's algorithm with average-depth sorting; complex inter-penetrating meshes may need the PyVista backend.
- `crystalfig batch` YAML schema is intentionally simple; advanced per-figure styling is better handled via the Python API.

These limitations are all outside the core publication-vector workflow and are documented as future extension points rather than incomplete core requirements.
