# Changelog

## 0.2.0 — release candidate

### Added

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
- PyVista produces raster PNG. Hybrid PDFs contain a raster scene with vector annotations.
- Automatic label collision avoidance and volumetric data rendering are not implemented.
- PyPI publication is pending; install from the repository until a release is uploaded.
