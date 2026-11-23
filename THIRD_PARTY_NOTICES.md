# Third-Party Notices

`crystalfig` builds on top of the following open-source libraries. It does not copy their source code; it calls their public APIs.

## Required dependencies

- **numpy** — BSD-3-Clause
- **scipy** — BSD-3-Clause
- **matplotlib** — PSF-based license
- **pymatgen / pymatgen-core** — MIT
- **spglib** — BSD-3-Clause
- **PyYAML** — MIT

## Optional dependencies

- **ASE** — LGPL-2.1+
- **seekpath** — MIT
- **PyVista / VTK** — MIT / BSD
- **Fresnel** — BSD-3-Clause

## External tools

- **TeX Live / MiKTeX** — used for TikZ → PDF compilation
- **pdfinfo / pdffonts** — used by `crystalfig preflight`

## Design influences

- Crystal Toolkit scene-graph philosophy (BSD-3-Clause)

## Publication benchmark data

`examples/data/visual_regression/Eu2FeGe2OS6.cif` and `Ce6Cd23Te.cif` are
unchanged copies from the CrystaLLM challenge set, published by Antunes,
Butler and Grau-Crespo as
[supporting data for Crystal Structure Generation with Autoregressive Large Language Modeling](https://doi.org/10.5281/zenodo.10642388)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The corresponding ground-truth structures are shown in Fig. 4 of
[Mat2Seq](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e23133d34964a0a09f6d076fc4b922a4-Abstract-Conference.html).
