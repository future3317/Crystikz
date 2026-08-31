# Migration Guide: from `crystal_tikz` to `crystalfig`

## What changed

The old `crystal_tikz` module mixed TikZ string generation with crystallographic logic. `crystalfig` separates these concerns:

- `CrystalStructure` / `Lattice` / `Site` form a backend-independent model.
- `Scene` holds visual primitives.
- `Renderer` backends (Matplotlib, TikZ, ...) consume the scene.
- `CrystalFigure` provides a high-level fluent API.

## Old → New mapping

| Old API | New API | Notes |
|---|---|---|
| `CrystalVisualizer` | `CrystalFigure` | `CrystalFigure.from_file(...)`, `.show_unit_cell()`, `.add_bonds(...)` |
| `LatticeBasis` | `Lattice` | `Lattice.from_parameters(a, b, c, alpha, beta, gamma)` |
| `Camera3D` | `Camera` | `Camera(elevation=..., azimuth=..., projection="orthographic")` |
| `build_perovskite` | `crystalfig.examples.presets.perovskite_structure` | Returns `CrystalStructure` |
| `build_rutile` | `crystalfig.examples.presets.rutile_structure` | Returns `CrystalStructure` |
| `build_wurtzite` | `crystalfig.examples.presets.wurtzite_structure` | Returns `CrystalStructure` |
| `EquivariantArchitectureVisualizer` | `EquivariantGNNDiagram` | Same TikZ output, data-driven API |
| `compile_tikz_to_pdf` | `crystalfig.export.compile_tikz_to_pdf` | Also see `LatexCompiler` |

## Compatibility layer

The `crystalfig.compat` shim has been removed in this stabilization pass.
Old names are no longer importable; migrate to the new APIs listed above.

## Example migration

### Before

```python
from crystal_tikz import LatticeTikZ
lat = LatticeTikZ(a=3.95, b=3.95, c=4.05, camera_elevation=25, camera_azimuth=40)
lat.add_atom("Ba", [0, 0, 0], color="primary")
lat.add_atom("Ti", [0.5, 0.5, 0.5], color="secondary")
lat.add_bond(0, 1)
tex = lat.generate_tikz()
```

### After

```python
from crystalfig import CrystalFigure
from crystalfig.examples.presets import perovskite_structure

fig = CrystalFigure(perovskite_structure())
fig.add_bonds("covalent").add_polyhedra("Ti")
fig.export("BaTiO3.pdf")
fig.export("BaTiO3.tex")  # pure TikZ
```
