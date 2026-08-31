# Visual Quality Refactor Report

## Scope

This refactor targets the **visual output** of `crystalfig`: the same crystallographic
correctness layer (`CrystalStructure`, PBC, neighbor search, polyhedra) is preserved,
but the `SceneBuilder → Renderer` pipeline and the default publication styling were
reworked so that gallery figures read like materials-science journal illustrations
rather than Matplotlib demos.

## Largest visual problems before the refactor

1. **Flat atoms and weak bonds.** Atoms were plain `Circle` patches; bonds were thin
   `ax.plot()` lines. At 89 mm single-column width the figure looked like a 2D diagram.
2. **Polyhedron clutter.** Triangulated octahedra rendered every internal diagonal as a
   dark edge; the translucent cage was often lost behind the atoms.
3. **A-site atoms swallowed the cage.** In perovskite/polyhedron mode Ba (and similar
   large cations) were drawn at the same large radius as the transition-metal center,
   hiding the TiO₆ octahedron.
4. **Heavy cell box.** Unit-cell edges were nearly black and drew attention away from
   the structure.
5. **No real publication palette.** The "muted" palette still used #ff0d0d for O,
   bright yellow for S, and neon green for Ba.
6. **Bonds rendered automatically with polyhedra.** Calling `add_polyhedra()` always
   turned on bond rendering, creating redundant gray spokes in polyhedron figures.
7. **Extreme aspect ratios.** Very elongated supercells (e.g. MoS₂ 3×3×1 viewed along a)
   produced tall canvases with large empty margins.

## What was changed

### Color palette (`crystalfig/styles/palette.py`)

- Added a hand-tuned `publication` / `journal_clean` palette.
- Keeps element identity (O red, S yellow, Ba green, Ti gray, etc.) but removes
  pure-RGB neon values.
- Made it the default; `muted`/`publication_muted` and `jmol` remain available.

### Themes (`crystalfig/styles/theme.py`)

- Default theme now uses the `publication` palette.
- Added three targeted presets:
  - `publication_structure` – ball-and-stick defaults.
  - `publication_polyhedra` – small central cation, medium vertex anions,
    low polyhedron opacity, thin edges.
  - `publication_overview` – smaller atoms and bonds for dense/multi-site figures.
- Added `bond_color_mode` (default `"split"`) and polyhedron-specific radius scales.

### Scene construction (`crystalfig/scene/builder.py`)

- Bonds are now computed whenever bonds **or** polyhedra are requested, but bond
  rendering respects the user's explicit choice; `add_polyhedra()` no longer forces
  bonds on.
- Polyhedron centers use a smaller radius than vertex anions so the cage remains
  readable.
- Cell edges use the gray accent color and reduced opacity.

### Matplotlib renderer (`crystalfig/renderers/matplotlib_renderer.py`)

- **Atoms:** `shaded` style now draws a base fill, dark rim, lower-right crescent
  shade, and upper-left highlight/spot to create a glossy, vector-safe sphere.
- **Bonds:** tube-like thick lines with round caps; split-color mode draws half of
  each bond in the color of each endpoint element; bonds are clipped so they stop at
  atom surfaces.
- **Polyhedra:** back-face transparency + silhouette-only edges; internal
  triangulation diagonals are no longer drawn.
- **Framing:** figure height auto-derives from projected aspect ratio, clamped to
  avoid extreme canvases.

### Gallery (`examples/generate_gallery.py`)

- NaCl now uses `crystalnn` for the full rock-salt network.
- BaTiO₃ and rutile use `publication_polyhedra` + cutoff-based TiO₆ octahedra.
- Wurtzite uses a (2,2,1) supercell with a [111] tilt.
- MoS₂ uses a (2,2,2) supercell with a near-c-axis tilt so both honeycomb in-plane
  structure and layer stacking are visible.
- Added `examples/make_contact_sheet.py` to produce `gallery/contact_sheet.png`.

## Gallery outputs

All outputs were regenerated and inspected visually:

- `gallery/01_rocksalt_ballstick.{pdf,svg,png}`
- `gallery/02_diamond_tetrahedral.{pdf,png}`
- `gallery/03_perovskite_octahedron.{pdf,png}` + TikZ PDF
- `gallery/04_rutile_distorted.{pdf,png}`
- `gallery/05_wurtzite_hexagonal.{pdf,png}`
- `gallery/06_mos2_layered.{pdf,png}`
- `gallery/07_equivariant_gnn.{tex,pdf}`
- `gallery/08_diamond_3d.png`
- `gallery/contact_sheet.png`

## Verification

```bash
python -m ruff check .
python -m pytest tests/ -q
python examples/generate_gallery.py
python examples/make_contact_sheet.py
```

- `ruff check .` → all checks passed.
- `pytest tests/ -q` → 63 passed.
- Gallery and contact sheet regenerated successfully.

## Current renderer capabilities

| Feature | Matplotlib | TikZ | Matplotlib3D |
|--------|------------|------|--------------|
| Glossy atoms | yes (vector circles) | yes (ball shading) | flat 3D scatter |
| Tube / split-color bonds | yes | no (plain line) | yes |
| Bond clipping at atom surface | yes | no | no |
| Clean polyhedron fill + silhouette edges | yes | partial (no silhouette logic) | no |
| Light cell box | yes | yes | yes |
| Auto aspect-ratio framing | yes | N/A | N/A |

## Remaining visual limitations

1. **TikZ backend** still uses simple lines for bonds and full face-edge drawing for
   polyhedra; it does not yet implement split-color bonds, clipping, or silhouette
   edges.
2. **SVG dedicated renderer** was not added; publication vector output currently goes
   through Matplotlib (PDF/SVG) or TikZ.
3. **Screen-space annotations** (corner axis triad, formula labels, panel labels,
   manual labels) are not implemented.
4. **Multi-panel compositor (`FigureGrid`)** is not implemented.
5. **Depth compositing** is painter's-algorithm only; there is no true CSG occlusion
   for intersecting spheres/bonds.
6. Atom glossy shading is built from stacked vector circles rather than true radial
   gradients, so it is slightly less smooth than a dedicated SVG gradient backend.
