# v0.2.0 release checks

Current candidate: `0.2.0`. This is a release candidate and has not been uploaded to PyPI.

The PyPI project page returned 404 during the release audit, so the README uses a Git installation
command. Recheck the package name and owner immediately before any upload. Do not publish until the
repository owner has prepared the project metadata, account, and trusted publishing configuration.

## Local checks

Use a Git checkout and a Python 3.11 development environment with the `dev` and `render3d` extras.

```bash
python -m pytest tests -q
python -m ruff check .
python scripts/check_release.py
python examples/generate_gallery.py --baseline-only
python examples/generate_gallery.py --showcase-only
python examples/generate_publication_review.py
```

`check_release.py` builds the wheel, installs it and its declared dependencies into a temporary
virtual environment, checks the installed version and CLI, renders a CIF through the installed
CLI, and checks PDF/SVG/PNG export metadata. PyVista's real renderer is validated separately by
the Linux visual-gallery job using the official PyVista headless-display action. The publication
benchmark also requires Poppler to inspect PDF dimensions, fonts, and image objects.

## CI acceptance

For a release, verify the Python 3.11 test job, the wheel-install smoke job, and the Linux
`visual-gallery` job. The visual artifact includes the baseline and diagnostic galleries and the
nine-structure publication benchmark. Inspect the 89 mm, 600 DPI exports across the Journal and
VESTA presets, the bond and framing comparisons, and the hybrid PDF at its intended width.
Verify periodic neighbors, coordination geometry, and camera composition before accepting a
visual change. Keep the run URL with the release notes.

## Version and uploads

Keep `pyproject.toml` and `crystalfig/_version.py` synchronized. Build and inspect the wheel before
upload. This repository does not include credentials or an automatic publish action; uploading
to PyPI is an explicit release operation and is not part of this audit.
