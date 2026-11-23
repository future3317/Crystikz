# v0.2.0 release checks

Current candidate: `0.2.0`. This is a release candidate and has not been uploaded to PyPI.

The PyPI project page returned 404 during the release audit, so the README uses a Git installation
command. Recheck the package name and owner immediately before any upload. Do not publish until the
repository owner has prepared the project metadata, account, and trusted publishing configuration.

## Local checks

Use the repository's project environment and run:

```bash
python -m pytest tests -q
python -m ruff check .
python -m build --no-isolation --wheel
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

The `CI` workflow must pass all four supported Python test jobs, the wheel-install smoke job, and
the Linux `visual-gallery` job. The visual artifact must include the baseline diagnostic sheet,
the diagnostic gallery, and the nine-structure publication benchmark at 89 mm and 600 DPI. Compare
the seven repeated structures to the frozen v4 renders, check the two sourced Mat2Seq structures,
and inspect the hybrid PDF at its intended width. SSIM flags changed images for review. Verify
periodic neighbors, coordination geometry, and camera composition before accepting a visual change.
Keep the job URL with the release notes.

## Version and uploads

Keep `pyproject.toml` and `crystalfig/_version.py` synchronized. Build and inspect the wheel before
upload. This repository does not include credentials or an automatic publish action; uploading
to PyPI is an explicit release operation and is not part of this audit.
