# crystalfig Implementation Report

## 1. Architecture

`crystalfig` is organized as a multi-layer visualization toolkit:

```text
Input (CIF/POSCAR/pymatgen/ASE)
    → crystalfig.io adapters
    → crystalfig.model.CrystalStructure (canonical internal model)
    → crystalfig.geometry / crystalfig.neighbors (analysis)
    → crystalfig.scene.Scene + Camera (backend-independent scene graph)
    → crystalfig.renderers (Matplotlib, SVG, TikZ, Matplotlib3D)
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
- `latex` — pdf2image helpers (mainly for preflight)

## 3. Implemented Features

### Structure model

- `Lattice` with lengths/angles, fractional ↔ Cartesian conversion, reciprocal matrix
- `Site` with species, occupancy, properties (magmom, force, displacement)
- `CrystalStructure` with formula, supercell, transformations

### IO

- CIF, POSCAR (with or without file extension), CSSR, JSON, YAML
- pymatgen `Structure` and ASE `Atoms` adapters

### Geometry

- Periodic image generation, nearest-image convention
- Coordination polyhedra via `scipy.spatial.ConvexHull`
- Miller plane / cell intersection
- First Brillouin zone via Voronoi construction (experimental)

### Neighbors

- `CutoffStrategy`, `CovalentRadiiStrategy`, `CrystalNNStrategy`, `ASEStrategy`
- Bonds store site indices, jimage, distance
- Manual bond injection via `CrystalFigure.add_manual_bonds()`

### Scene & Camera

- Primitives: Sphere, Cylinder/Bond, Line, Polyline, Polygon, Polyhedron, Arrow/Axis, Text, LegendItem
- `Camera` with elevation/azimuth/roll, orthographic/perspective, view along `[uvw]` or normal to `(hkl)`
- Depth sorting for painter's-algorithm rendering
- `Primitive.visible`, `Group.visible`, and `Primitive.layer` are respected by renderers

### Renderers

- `MatplotlibRenderer`: PDF/SVG/EPS/PGF/PNG/TIFF, transparent backgrounds, exact figure size
- `SvgRenderer`: pure SVG publication output with gradients and translucent polyhedra
- `TikzRenderer`: standalone TikZ with centralized library loading
- `Matplotlib3DRenderer`: true 3D projection preview (hybrid vector/raster)

### Figure builder

- Fluent API: `.view()`, `.supercell()`, `.add_bonds()`, `.add_polyhedra()`, `.add_vector()`, `.add_miller_plane()`, `.style()`, `.export()`
- Per-site/per-species style overrides: `.style_atoms()`, `.hide_atoms()`, `.style_bonds()`, `.hide_bonds()`
- Manual annotations: `.add_label()`, `.add_formula_label()`, `.add_panel_label()`
- `CrystalFigure.draw(ax=...)` for embedding in normal Matplotlib workflows
- `CrystalFigure.export_scene(scene, path, ...)` for advanced Scene editing
- One-line helper `plot_structure(...)`

### CLI

- `crystalfig render`
- `crystalfig inspect`
- `crystalfig doctor`
- `crystalfig preflight`

### Compatibility

- Legacy `crystal_tikz` compatibility shim has been removed; see `MIGRATION.md`.

## 4. Tests

```bash
python -m pytest tests -q
```

Result: **84 passed, 10 warnings** (pymatgen/spglib deprecation/oxidation-state notices).

Coverage includes:

- Lattice round-trip for cubic, hexagonal, triclinic cells
- Site species/occupancy
- Structure supercell (diagonal and general 3×3)
- Periodic nearest-image wrapping
- ConvexHull polyhedra (octahedron, tetrahedron)
- Miller plane intersection
- Neighbor strategies
- pymatgen round-trip and CIF loading
- Matplotlib export (PDF/PNG)
- SVG export and gradient output
- TikZ standalone generation and LaTeX compilation
- CrystalFigure fluent API
- Scene visible/layer contract
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
- `03_perovskite_octahedron.{pdf,png}`
- `03_perovskite_octahedron_tikz.pdf`
- `04_rutile_distorted.pdf`
- `05_wurtzite_hexagonal.pdf`
- `06_mos2_layered.pdf`
- `07_equivariant_gnn.{tex,pdf}`
- `contact_sheet.png`

## 7. Linting

```bash
python -m ruff check crystalfig tests examples
```

Result: **All checks passed!**

## 8. Known Limitations

- `Matplotlib3DRenderer` is a preview backend; PDF/SVG output from it is hybrid (vector primitives baked by a 3D projection), not a pure camera-independent scene.
- Volumetric isosurfaces (CHGCAR, cube) are not implemented.
- Magnetic symmetry (spglib magnetic groups) is not implemented.
- Trajectory / multi-frame export is not implemented.
- Advanced depth occlusion for vector renderers uses a painter's algorithm with average-depth sorting; complex inter-penetrating meshes may need a dedicated 3D backend.
- Automatic label placement / collision avoidance is not implemented; use manual annotations or external layout.
- Brillouin-zone k-paths require the optional `seekpath` dependency.
- PyVista/Fresnel/POV-Ray backends are not included.

These limitations are all outside the core publication-vector workflow and are documented as future extension points rather than incomplete core requirements.
