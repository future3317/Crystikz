# Crystikz v4 visual QA

## Comparison protocol

`examples/generate_visual_review_v4.py` rendered seven structures with matching 89 mm, 450 dpi canvases in the legacy `nature` theme, `journal_clean`, and `journal_vivid`. The three styles share the same planned intent, display window, and actual selected camera. `comparison_contact_sheet.png` shows all 21 renders; the folder also contains three actual camera candidate renders per structure, MoS₂ side and oblique views, a four-panel PNG/PDF, and 89 mm, 600 dpi BaTiO₃ and Si examples.

The supplied target reference uses a white canvas, connected periodic motifs, visibly distinct coordination colors, translucent but saturated coordination volumes, element-colored atoms, thin physical edges, and compact keys. This is a visual comparison, not a claim of journal certification. At the displayed single-column size, the revised vivid theme uses a front-face alpha of 0.448 (`0.64 × 0.70`) and a back-face alpha of 0.102 (`0.64 × 0.16`); the clean theme is more restrained. Environment colors are stable by coordination role rather than atom identity: the vivid role sequence starts with blue, orange, green, and purple, while atom colors remain element-based. Automatically cropped coordination/hybrid scenes omit the distant full-cell frame so the motif fills the canvas; callers can restore it with `show_unit_cell=True`.

## Structure-by-structure review

| Structure | Visual result | Actual selected view / expression | Remaining limitation |
| --- | --- | --- | --- |
| BaTiO₃ | Two complete TiO₆ cages form a readable connected motif; the center and oxygen roles are visible, and vivid separates the faces more clearly than the legacy theme. | `lattice_+++_roll90`; hybrid coordination, repeat 2×2×1; 2 TiO₆. | Ba context atoms remain sparse in this coordination-focused crop. |
| MoS₂ | The selected oblique view shows the S–Mo–S trilayer and avoids interlayer bonds. Side and oblique candidate images are both included. | `tilt_direct_a_plane_b`; layered network, repeat 2×4×1. | The side view compresses in-plane topology; the oblique view is the clearer overview. |
| Si | A continuous diamond network is shown without the former dense projected columns. | `lattice_++-`; ball-and-stick network, repeat 3×2×1. | Several bonds still cross in projection because the network is three-dimensional; alternate scored views are included. |
| NaCl | The connected ionic framework reads as a periodic lattice without unnecessary coordination volumes. | `lattice_++-`; ball-and-stick framework, repeat 2×2×1. | The view emphasizes framework continuity over one ion-pair motif. |
| MgAl₂O₄ | Two complete shells retain both MgO₄ and AlO₆; the center selection and lower vivid alpha reduce the previous pile-up. | `lattice_+-+_roll90`; hybrid coordination, repeat 1×1×1; 1 MgO₄ and 1 AlO₆. | The shared ligand is the visible junction; the view is a representative local motif, not a full-cell summary. |
| LiFePO₄ | Three complete shells include FeO₆, PO₄, and LiO₆, inferred from the same periodic neighbor graph. Their colors distinguish local roles. | `lattice_0+-_roll90`; hybrid coordination, repeat 1×1×1; one shell of each environment. | CrystalNN-derived LiO₆ is retained as a coordination candidate; users should review whether it belongs in their scientific story. |
| α-SiO₂ | Two connected SiO₄ tetrahedra make the local tetrahedral coordination legible. | `plane_b_roll90`; hybrid coordination, repeat 3×2×1; 2 SiO₄. | This close motif communicates local coordination better than the extended quartz topology. |

## Four-panel review

`paper_four_panel_v4.png` improves the occupied area, aligns panel letters and titles, and places compact element keys on a consistent bottom baseline. Diagnostic display text is removed. The PDF combines PyVista raster scenes with Matplotlib vector titles and keys. It is usable as a paper-figure draft, but the four equal panels still have different visual density: MoS₂ is a thin layer view and Si is an open network. A manuscript may need hand-selected crops or panel sizes to emphasize a particular scientific claim.

## Reproducible inspection

```powershell
& 'D:\Anaconda\envs\piepaper\python.exe' examples\generate_visual_review_v4.py
& 'D:\Anaconda\envs\piepaper\python.exe' -m pytest tests/test_visual_semantics.py -q
```

`metrics.json` records each scene's selected camera and score components, display budget and repeat, coordination counts, renderer, and output path. The automatic neighbor graph is a coordination heuristic; users should review it for structures whose oxidation states or bonding are not represented in the input.
