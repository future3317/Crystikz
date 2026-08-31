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

## Publication capability stabilization (this pass)

After the visual refactor, the next priority was controllability: making it easy to
override colors, radii, bonds, polyhedra, and annotations for real paper figures
without adding heavy new abstractions.

### Element coverage

- `crystalfig/styles/palette.py`: `_PUBLICATION_BASE` now covers H through Bi (Z=1-83).
  Existing hand-tuned colors are preserved; missing elements use muted Jmol-derived
  colors.
- `crystalfig/styles/radii.py`: Covalent radii now cover H through Bi. Unknown
  elements fall back to 1.5 Å with a one-time warning instead of the previous 0.2 Å.

### Correctness fixes

- `cell_complete` boundary mode no longer draws bonds whose periodic partner lies
  outside the displayed `[0,1]^3` cell, eliminating dangling bonds.
- `CrystalStructure.make_supercell()` now supports arbitrary integer 3×3
  transformation matrices via pymatgen.
- `CrystalFigure.view()` rejects unknown string axes instead of silently defaulting
  to `[1,1,0]`.
- `load_structure()` recognizes `POSCAR`/`CONTCAR` files with no extension by
  basename.
- Species selectors like `"Ti"` now match the dominant element if an exact species
  match fails.
- Disordered sites emit a warning when only the dominant species is rendered.

### User-control API

- `CrystalFigure.palette` is now a per-figure copy; modifying it does not pollute
  the global palette.
- `style_atoms(species=..., indices=..., color=..., scale=..., opacity=...,
  radius=..., render_style=...)` and `hide_atoms(...)` for per-species/per-site
  atom overrides.
- `style_bonds(pair=..., indices=..., width=..., color=..., opacity=...)` and
  `hide_bonds(...)` for bond overrides.
- `add_bonds()` accepts a strategy name, a `NeighborStrategy` instance, or
  `pair_cutoffs` for cutoff strategies.
- `add_manual_bonds(bonds)` for explicit bond lists.
- `add_polyhedra()` appends a `PolyhedraSpec` instead of overwriting, so multiple
  coordination environments can be drawn with independent colors and strategies.

### Annotations and embedding

- `add_label(site_index, text, offset=...)` for manual site labels.
- `add_formula_label(text, position=...)` and `add_panel_label(text, position=...)`
  for screen-space figure labels.
- `CrystalFigure.draw(ax)` embeds a crystal figure into an existing Matplotlib Axes,
  enabling normal multi-panel publication workflows without a custom compositor.
- `CrystalFigure.export_scene(scene, path, ...)` lets advanced users modify a Scene
  directly and export it.

### Scene contract

- `Scene.all_primitives()` respects `Primitive.visible=False` and `Group.visible=False`.
- `Scene.get_group(name)` helper added.
- Unused `Plane` primitive removed.

## Verification (this pass)

```bash
python -m ruff check .
python -m pytest tests/ -q
python examples/generate_gallery.py
python examples/make_contact_sheet.py
python -m build --wheel
```

- `ruff check .` → all checks passed.
- `pytest tests/ -q` → 84 passed.
- Gallery and `gallery/contact_sheet.png` regenerated.
- Wheel built successfully and smoke-installed.

## Current renderer capabilities

| Feature | Matplotlib | SVG | TikZ | Matplotlib3D |
|--------|------------|-----|------|--------------|
| Glossy atoms | yes (vector circles) | yes (radial-gradient-like) | yes (ball shading) | flat 3D scatter |
| Tube / split-color bonds | yes | yes (gradient) | no (plain line) | yes |
| Bond clipping at atom surface | yes | yes | no | no |
| Clean polyhedron fill + silhouette edges | yes | yes | partial (no silhouette logic) | no |
| Light cell box | yes | yes | yes | yes |
| Auto aspect-ratio framing | yes | yes | N/A | N/A |
| Screen-space annotations | yes | yes | partial | no |

## Remaining visual limitations

1. **TikZ backend** still uses simple lines for bonds and full face-edge drawing for
   polyhedra; it does not yet implement split-color bonds, clipping, or silhouette
   edges.
2. **Depth compositing** is painter's-algorithm only; there is no true CSG occlusion
   for intersecting spheres/bonds.
3. **Multi-panel compositor (`FigureGrid`)** is not implemented; use
   `CrystalFigure.draw(ax=...)` with normal Matplotlib subplots instead.
4. **Automatic label placement / collision avoidance** is not implemented.
5. Atom glossy shading in Matplotlib is built from stacked vector circles rather than
   true radial gradients; the SVG backend is now available for smoother gradients.
