# Changelog

## 0.2.0 — release candidate

### Added

- `FigureConfig`, `DisplayConfig`, `CameraConfig`, and `RenderConfig` for reproducible Python and YAML figures.
- Composable 3D styles, five publication presets, and separate element and coordination colors.
- Independent bond display policies for all bonds, center–ligand sticks, polyhedron edges, or hidden bonds.
- Full-cell, local coordination, and connected motif framing with projection crowding in camera ranking.
- Explicit cell, motif and supercell views, connected coordination shells, and depth-aware camera ranking.
- Nine-structure publication benchmark with two experimental Mat2Seq reference CIFs and same-scene backend comparisons.
- Optional PyVista/VTK mesh rendering and hybrid PDF export.
- Shared scene fitting, deterministic structure-derived view selection, and three render-quality levels.
- A generic `CrystalFigure.publication()` entry point and matching Python, CLI, and YAML batch options.
- Eleven-structure rendering showcase, including a monoclinic oxide CIF input, and Linux headless PyVista CI coverage.
- Export preflight reporting for vector/raster properties, dimensions, DPI, transparency, and Type 3 fonts.

### Compatibility

- Existing `CrystalFigure.export(path, fmt, width, transparent, *, backend, dpi, height)` arguments remain supported.
- Existing Matplotlib, SVG, TikZ, and Matplotlib3D renderers remain available.

### Known limits

- Bond strategies infer candidate neighbors; users must verify coordination for their scientific interpretation.
- PyVista writes raster PNG and hybrid PDF. Hybrid PDFs contain a raster scene with vector annotations.
- Automatic label collision avoidance and volumetric data rendering are not implemented.
- PyPI publication is pending; install from the repository until a release is uploaded.
