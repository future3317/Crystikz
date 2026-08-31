"""
crystal_tikz: Publication-Grade TikZ Generator for 3D Crystal Structures & Equivariant GNNs
Designed for Nature / Science / ICLR / Physical Review style figures.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

# --- Color Palette Presets (Nature / Science Style: Muted, Low-Saturation, Elegant) ---
PALETTES = {
    "nature_classic": {
        "primary": "RGB(44, 95, 138)",      # Slate Blue
        "secondary": "RGB(196, 90, 74)",    # Muted Coral / Crimson
        "accent": "RGB(67, 147, 108)",      # Forest Teal
        "amber": "RGB(217, 131, 36)",       # Muted Amber
        "purple": "RGB(117, 112, 179)",     # Soft Violet
        "gray": "RGB(140, 150, 160)",       # Cool Gray
        "dark": "RGB(40, 44, 52)",          # Deep Charcoal
        "light_bg": "RGB(248, 249, 250)",   # Clean Off-white
        "poly_fill": "RGB(67, 147, 108)",   # Soft Green Polyhedron
        "poly_fill2": "RGB(196, 90, 74)",   # Soft Coral Polyhedron
    },
    "scipost_vibrant": {
        "primary": "RGB(31, 119, 180)",
        "secondary": "RGB(255, 127, 14)",
        "accent": "RGB(44, 160, 44)",
        "amber": "RGB(214, 39, 40)",
        "purple": "RGB(148, 103, 189)",
        "gray": "RGB(127, 127, 127)",
        "dark": "RGB(30, 30, 30)",
        "light_bg": "RGB(255, 255, 255)",
        "poly_fill": "RGB(31, 119, 180)",
        "poly_fill2": "RGB(255, 127, 14)",
    }
}

@dataclass
class Atom:
    symbol: str
    frac_coords: np.ndarray
    color_name: str
    radius: float = 0.18  # in visual units
    label: Optional[str] = None
    charge: Optional[str] = None
    draw_halo: bool = False
    halo_style: str = "dashed"

@dataclass
class Bond:
    idx_i: int
    idx_j: int
    color_name: str = "dark!50"
    width: str = "thick"
    opacity: float = 0.85
    dashed: bool = False

@dataclass
class Polyhedron:
    center_idx: int
    vertex_indices: List[int]
    faces: List[List[int]]  # list of vertex index triangles/quads
    fill_color: str = "poly_fill"
    fill_opacity: float = 0.22
    draw_color: str = "dark!60"
    draw_opacity: float = 0.60
    line_width: str = "thin"

class LatticeTikZ:
    """
    Renders 3D Crystal Unit Cells, Supercells, Wyckoff sites, and Coordination Polyhedra
    into pure LaTeX TikZ with depth sorting (Painter's Algorithm).
    """
    def __init__(
        self,
        a: float = 4.0,
        b: float = 4.0,
        c: float = 4.0,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        camera_elevation: float = 22.0,
        camera_azimuth: float = 48.0,
        scale: float = 1.0,
        palette: str = "nature_classic"
    ):
        self.a, self.b, self.c = a, b, c
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.elevation = np.radians(camera_elevation)
        self.azimuth = np.radians(camera_azimuth)
        self.scale = scale
        self.palette = PALETTES.get(palette, PALETTES["nature_classic"])
        
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.polyhedra: List[Polyhedron] = []
        self.vectors: List[Dict] = []  # Custom vectors (e.g. polarization, magnetic moment, flow arrow)
        
        self._build_lattice_matrix()

    def _build_lattice_matrix(self):
        """Constructs 3x3 Cartesian lattice basis matrix [a, b, c]."""
        al, be, ga = np.radians(self.alpha), np.radians(self.beta), np.radians(self.gamma)
        val = (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        val = np.clip(val, -1.0, 1.0)
        
        c_x = self.c * np.cos(be)
        c_y = self.c * val
        c_z = np.sqrt(max(0.0, self.c**2 - c_x**2 - c_y**2))
        
        self.lattice_matrix = np.array([
            [self.a, self.b * np.cos(ga), c_x],
            [0.0,    self.b * np.sin(ga), c_y],
            [0.0,    0.0,                 c_z]
        ])

    def add_atom(self, symbol: str, frac: Union[List[float], np.ndarray], color: str, radius: float = 0.22, label: Optional[str] = None, charge: Optional[str] = None, draw_halo: bool = False):
        self.atoms.append(Atom(
            symbol=symbol,
            frac_coords=np.array(frac, dtype=float),
            color_name=color,
            radius=radius,
            label=label or symbol,
            charge=charge,
            draw_halo=draw_halo
        ))
        return len(self.atoms) - 1

    def add_bond(self, i: int, j: int, color: str = "gray!60", width: str = "semithick", opacity: float = 0.8, dashed: bool = False):
        self.bonds.append(Bond(idx_i=i, idx_j=j, color_name=color, width=width, opacity=opacity, dashed=dashed))

    def add_polyhedron(self, center_idx: int, vertex_indices: List[int], faces: List[List[int]], fill_color: str = "poly_fill", fill_opacity: float = 0.22):
        self.polyhedra.append(Polyhedron(
            center_idx=center_idx,
            vertex_indices=vertex_indices,
            faces=faces,
            fill_color=fill_color,
            fill_opacity=fill_opacity
        ))

    def add_vector(self, start_frac: Union[List[float], np.ndarray], end_frac: Union[List[float], np.ndarray], color: str = "secondary", label: str = "", style: str = "-{Stealth[length=3mm,width=2mm]}", width: str = "very thick"):
        self.vectors.append({
            "start": np.array(start_frac, dtype=float),
            "end": np.array(end_frac, dtype=float),
            "color": color,
            "label": label,
            "style": style,
            "width": width
        })

    def _project_3d(self, cart_pos: np.ndarray, center: np.ndarray) -> Tuple[float, float, float]:
        """Projects 3D point to 2D screen coordinate (u, v) and returns depth d."""
        p = cart_pos - center
        
        # Azimuth rotation around Z
        R_z = np.array([
            [np.cos(self.azimuth), -np.sin(self.azimuth), 0],
            [np.sin(self.azimuth),  np.cos(self.azimuth), 0],
            [0,                    0,                   1]
        ])
        # Elevation rotation around X
        R_x = np.array([
            [1, 0,                   0                  ],
            [0, np.cos(self.elevation), -np.sin(self.elevation)],
            [0, np.sin(self.elevation),  np.cos(self.elevation)]
        ])
        
        p_cam = R_x @ (R_z @ p)
        # Scale to standard TikZ cm units
        u = p_cam[0] * self.scale
        v = p_cam[1] * self.scale
        d = p_cam[2]  # depth along line of sight (ascending = deeper / farther away)
        return u, v, d

    def generate_tikz(self, include_preamble: bool = False, draw_axes: bool = True, draw_cell_box: bool = True, title: str = "") -> str:
        """Generates depth-sorted Nature-grade TikZ code."""
        # Calculate cell origin and corners
        L = self.lattice_matrix
        corners_frac = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ])
        corners_cart = np.array([L @ cf for cf in corners_frac])
        cell_center = np.mean(corners_cart, axis=0)

        # 12 Unit cell edges (pairs of corner indices)
        edges = [
            (0,1), (1,2), (2,3), (3,0), # bottom face
            (4,5), (5,6), (6,7), (7,4), # top face
            (0,4), (1,5), (2,6), (3,7)  # vertical struts
        ]

        # Project corners
        proj_corners = [self._project_3d(c, cell_center) for c in corners_cart]

        # Depth-sorted render queue
        render_queue = []

        # 1. Cell Edges
        for (i1, i2) in edges:
            p1 = proj_corners[i1]
            p2 = proj_corners[i2]
            avg_depth = (p1[2] + p2[2]) / 2.0
            render_queue.append({
                "type": "cell_edge",
                "depth": avg_depth,
                "p1": p1,
                "p2": p2,
                "is_back": avg_depth < 0.0  # simple depth heuristic
            })

        # 2. Coordination Polyhedra Faces
        for poly in self.polyhedra:
            vert_cart = [L @ self.atoms[vi].frac_coords for vi in poly.vertex_indices]
            vert_proj = [self._project_3d(vc, cell_center) for vc in vert_cart]
            for face in poly.faces:
                f_proj = [vert_proj[idx] for idx in face]
                avg_depth = np.mean([pt[2] for pt in f_proj])
                render_queue.append({
                    "type": "poly_face",
                    "depth": avg_depth,
                    "points": f_proj,
                    "fill_color": poly.fill_color,
                    "fill_opacity": poly.fill_opacity,
                    "draw_color": poly.draw_color,
                    "draw_opacity": poly.draw_opacity,
                    "width": poly.line_width
                })

        # 3. Bonds
        for bond in self.bonds:
            c1 = L @ self.atoms[bond.idx_i].frac_coords
            c2 = L @ self.atoms[bond.idx_j].frac_coords
            p1 = self._project_3d(c1, cell_center)
            p2 = self._project_3d(c2, cell_center)
            avg_depth = (p1[2] + p2[2]) / 2.0
            render_queue.append({
                "type": "bond",
                "depth": avg_depth,
                "p1": p1,
                "p2": p2,
                "color": bond.color_name,
                "width": bond.width,
                "opacity": bond.opacity,
                "dashed": bond.dashed
            })

        # 4. Atoms
        for atom in self.atoms:
            c_pos = L @ atom.frac_coords
            u, v, d = self._project_3d(c_pos, cell_center)
            render_queue.append({
                "type": "atom",
                "depth": d,
                "pos": (u, v),
                "atom": atom
            })

        # 5. Vectors
        for vec in self.vectors:
            c1 = L @ vec["start"]
            c2 = L @ vec["end"]
            p1 = self._project_3d(c1, cell_center)
            p2 = self._project_3d(c2, cell_center)
            avg_depth = (p1[2] + p2[2]) / 2.0 + 0.1 # slightly higher priority
            render_queue.append({
                "type": "vector",
                "depth": avg_depth,
                "p1": p1,
                "p2": p2,
                "color": vec["color"],
                "label": vec["label"],
                "style": vec["style"],
                "width": vec["width"]
            })

        # Sort queue strictly by depth ascending (farthest rendered first)
        render_queue.sort(key=lambda item: item["depth"])

        # Construct LaTeX TikZ commands
        lines = []
        if include_preamble:
            lines.append(r"\documentclass[tikz,border=8pt]{standalone}")
            lines.append(r"\usepackage{tikz}")
            lines.append(r"\usepackage{amsmath,amssymb}")
            lines.append(r"\usetikzlibrary{shapes,arrows.meta,calc,shadings,shadows,decorations.markings}")
            lines.append(r"\begin{document}")

        lines.append(r"% === Nature-Style Crystal Unit Cell TikZ Code ===")
        # Palette definitions
        for col_name, col_val in self.palette.items():
            lines.append(f"\\definecolor{{{col_name}}}{{RGB}}{{{col_val[4:-1].replace(' ', '')}}}")

        lines.append(r"\begin{tikzpicture}[scale=1.0, >=Stealth,")
        lines.append(r"    atom_label/.style={font=\sffamily\bfseries\tiny, text=white},")
        lines.append(r"    vec_label/.style={font=\sffamily\bfseries\footnotesize, inner sep=2pt}")
        lines.append(r"]")

        if title:
            lines.append(f"  % Title\n  \\node[above, font=\\sffamily\\bfseries\\large, text=dark] at (0, 3.8) {{{title}}};")

        # Render items in sorted order
        for item in render_queue:
            itype = item["type"]
            if itype == "cell_edge" and draw_cell_box:
                p1, p2 = item["p1"], item["p2"]
                if item["is_back"]:
                    lines.append(f"  \\draw[dark!35, thin, dashed] ({p1[0]:.3f}, {p1[1]:.3f}) -- ({p2[0]:.3f}, {p2[1]:.3f});")
                else:
                    lines.append(f"  \\draw[dark!70, semithick] ({p1[0]:.3f}, {p1[1]:.3f}) -- ({p2[0]:.3f}, {p2[1]:.3f});")
            
            elif itype == "poly_face":
                pts = item["points"]
                pts_str = " -- ".join([f"({p[0]:.3f}, {p[1]:.3f})" for p in pts])
                col = item["fill_color"]
                f_op = item["fill_opacity"]
                d_col = item["draw_color"]
                d_op = item["draw_opacity"]
                w = item["width"]
                lines.append(f"  \\filldraw[{w}, fill={col}, fill opacity={f_op}, draw={d_col}, draw opacity={d_op}] {pts_str} -- cycle;")

            elif itype == "bond":
                p1, p2 = item["p1"], item["p2"]
                col = item["color"]
                w = item["width"]
                op = item["opacity"]
                dash = ", dashed" if item["dashed"] else ""
                lines.append(f"  \\draw[{w}, color={col}, opacity={op}{dash}] ({p1[0]:.3f}, {p1[1]:.3f}) -- ({p2[0]:.3f}, {p2[1]:.3f});")

            elif itype == "atom":
                u, v = item["pos"]
                atm = item["atom"]
                r = atm.radius * self.scale
                # Shading / 3D Ball appearance
                lines.append(f"  % Atom {atm.symbol}")
                if atm.draw_halo:
                    lines.append(f"  \\draw[{atm.halo_style}, {atm.color_name}!80, thin] ({u:.3f}, {v:.3f}) circle ({r*1.35:.3f});")
                
                # Ball with radial highlight for Nature look
                lines.append(f"  \\shade[shading=ball, ball color={atm.color_name}] ({u:.3f}, {v:.3f}) circle ({r:.3f});")
                if atm.label:
                    lines.append(f"  \\node[atom_label] at ({u:.3f}, {v:.3f}) {{{atm.label}}};")

            elif itype == "vector":
                p1, p2 = item["p1"], item["p2"]
                col = item["color"]
                lbl = item["label"]
                sty = item["style"]
                w = item["width"]
                lbl_code = f" node[midway, above right, vec_label, text={col}] {{{lbl}}}" if lbl else ""
                lines.append(f"  \\draw[{w}, {col}, {sty}] ({p1[0]:.3f}, {p1[1]:.3f}) -- ({p2[0]:.3f}, {p2[1]:.3f}){lbl_code};")

        # Lattice basis vector arrows (Origin -> a1, a2, a3)
        if draw_axes:
            p0 = proj_corners[0]
            pa = proj_corners[1]
            pb = proj_corners[3]
            pc = proj_corners[4]
            # Offsets for origin axis display
            lines.append("  % Lattice Basis Arrows")
            lines.append(f"  \\draw[very thick, ->, secondary] ({p0[0]:.3f}, {p0[1]:.3f}) -- ({p0[0] + (pa[0]-p0[0])*0.4:.3f}, {p0[1] + (pa[1]-p0[1])*0.4:.3f}) node[below, font=\\sffamily\\bfseries\\scriptsize, text=secondary] {{$\\mathbf{{a}}_1$}};")
            lines.append(f"  \\draw[very thick, ->, accent] ({p0[0]:.3f}, {p0[1]:.3f}) -- ({p0[0] + (pb[0]-p0[0])*0.4:.3f}, {p0[1] + (pb[1]-p0[1])*0.4:.3f}) node[below right, font=\\sffamily\\bfseries\\scriptsize, text=accent] {{$\\mathbf{{a}}_2$}};")
            lines.append(f"  \\draw[very thick, ->, primary] ({p0[0]:.3f}, {p0[1]:.3f}) -- ({p0[0] + (pc[0]-p0[0])*0.4:.3f}, {p0[1] + (pc[1]-p0[1])*0.4:.3f}) node[left, font=\\sffamily\\bfseries\\scriptsize, text=primary] {{$\\mathbf{{a}}_3$}};")

        lines.append(r"\end{tikzpicture}")
        if include_preamble:
            lines.append(r"\end{document}")

        return "\n".join(lines)


class EquivariantGNNTikZ:
    """
    Renders Modular Architecture Diagrams for Equivariant Crystal Graph Neural Networks,
    Periodic Message Passing, Irrep Tensor Products (Clebsch-Gordan), and Flow Matching.
    """
    def __init__(self, palette: str = "nature_classic"):
        self.palette = PALETTES.get(palette, PALETTES["nature_classic"])

    def generate_message_passing_pipeline(self, include_preamble: bool = False) -> str:
        """Generates a high-impact Nature/ICLR style Periodic Equivariant Message Passing Diagram."""
        lines = []
        if include_preamble:
            lines.append(r"\documentclass[tikz,border=10pt]{standalone}")
            lines.append(r"\usepackage{tikz}")
            lines.append(r"\usepackage{amsmath,amssymb}")
            lines.append(r"\usetikzlibrary{shapes.geometric,arrows.meta,calc,positioning,shadows.blur,backgrounds,fit}")
            lines.append(r"\begin{document}")

        for col_name, col_val in self.palette.items():
            lines.append(f"\\definecolor{{{col_name}}}{{RGB}}{{{col_val[4:-1].replace(' ', '')}}}")

        lines.append(r"""
\begin{tikzpicture}[
    >=Stealth,
    node distance=1.4cm and 1.8cm,
    font=\sffamily,
    box_base/.style={
        rectangle, rounded corners=6pt, align=center, inner sep=8pt,
        line width=1pt, blur shadow={shadow blur steps=4, shadow xshift=0.5mm, shadow yshift=-0.5mm, shadow opacity=15}
    },
    crystal_box/.style={
        box_base, fill=primary!8, draw=primary!80, text=dark, font=\sffamily\bfseries\small
    },
    irreps_box/.style={
        box_base, fill=purple!8, draw=purple!80, text=dark, font=\sffamily\bfseries\small
    },
    tp_box/.style={
        box_base, fill=secondary!8, draw=secondary!80, text=dark, font=\sffamily\bfseries\small
    },
    flow_box/.style={
        box_base, fill=accent!8, draw=accent!80, text=dark, font=\sffamily\bfseries\small
    },
    tag_node/.style={
        fill=white, draw=dark!30, line width=0.6pt, rounded corners=3pt,
        font=\sffamily\scriptsize\bfseries, inner sep=3pt
    },
    math_txt/.style={font=\small, text=dark},
    flow_arrow/.style={->, very thick, dark!75, line width=1.5pt},
    sub_arrow/.style={->, thick, dark!50, dashed}
]

% 1. Input Periodic Lattice Node
\node[crystal_box, minimum width=4.2cm, minimum height=2.4cm] (input_cell) {
    \textbf{Periodic Crystal Structure} $\mathcal{C}$\\
    \vspace{2pt}
    \scriptsize Fractional Coords: $\mathbf{s}_i \in [0, 1)^3$\\
    \scriptsize Cartesian Positions: $\mathbf{x}_i = \mathbf{L}\mathbf{s}_i$\\
    \scriptsize Lattice Basis: $\mathbf{L} = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3]$
};

% 2. Periodic Graph & Relative Displacements
\node[crystal_box, right=2.0cm of input_cell, minimum width=4.6cm, minimum height=2.4cm] (graph_build) {
    \textbf{Periodic Edge Displacements}\\
    \vspace{3pt}
    $\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + \mathbf{L}\mathbf{k}_{ij}$\\
    \vspace{2pt}
    \scriptsize Distance: $d_{ij} = \|\mathbf{r}_{ij}\|_2 \le r_{\text{cut}}$\\
    \scriptsize Spherical Unit: $\hat{\mathbf{r}}_{ij} = \mathbf{r}_{ij} / d_{ij}$
};

% 3. Irrep Geometric Embeddings
\node[irreps_box, below=1.6cm of input_cell, minimum width=4.2cm, minimum height=2.8cm] (irreps_embed) {
    \textbf{Equivariant Irrep Embedding}\\
    \vspace{4pt}
    \scriptsize Node Features ($l=0, 1, 2$):\\
    $\mathbf{h}_i^{(0)} = \bigoplus_c \mathbf{f}_{i, c} \in \mathbb{R}^{\sum (2l+1)C_l}$\\
    \vspace{3pt}
    \scriptsize Spherical Harmonics ($l \le l_{\max}$):\\
    $\mathbf{Y}(\hat{\mathbf{r}}_{ij}) = [Y_0^0, Y_1^{-1, 0, 1}, Y_2^{\dots}]^T$
};

% 4. Equivariant Tensor Product Interaction Block
\node[tp_box, right=2.0cm of irreps_embed, minimum width=4.6cm, minimum height=2.8cm] (tp_block) {
    \textbf{Equivariant Tensor Product}\\
    \vspace{4pt}
    $\mathbf{m}_{ij} = \left( \mathbf{h}_j \otimes_{\mathrm{cg}} \mathbf{Y}(\hat{\mathbf{r}}_{ij}) \right) \cdot \mathcal{R}_{\text{Bessel}}(d_{ij})$\\
    \vspace{3pt}
    \scriptsize Clebsch-Gordan Coefficients: $C_{(l_1, m_1)(l_2, m_2)}^{(l_3, m_3)}$\\
    \scriptsize Gated Non-Linearity: $\sigma(\mathbf{s}) \oplus \phi(\|\mathbf{v}\|)\frac{\mathbf{v}}{\|\mathbf{v}\|}$
};

% 5. Equivariant Update & Flow Vector Field
\node[flow_box, right=2.0cm of tp_block, minimum width=4.5cm, minimum height=2.8cm] (flow_field) {
    \textbf{Equivariant Flow Prediction}\\
    \vspace{4pt}
    \scriptsize Coordinate Velocity Field ($l=1$):\\
    $\mathbf{v}_i = \sum_{j \in \mathcal{N}_i} \mathbf{m}_{ij}^{(l=1)}$\\
    \vspace{2pt}
    \scriptsize Lattice Strain Drift ($l=2 \oplus 0$):\\
    $\mathbf{\dot{L}} = \frac{1}{N}\sum_i \mathbf{x}_i \otimes \mathbf{v}_i + \mathbf{\Sigma}_{\text{cell}}$
};

% 6. Physical Property Readout / Target
\node[flow_box, right=2.0cm of graph_build, minimum width=4.5cm, minimum height=2.4cm] (prop_readout) {
    \textbf{Physical Tensor Output}\\
    \vspace{3pt}
    \scriptsize Elastic Moduli: $\mathbf{C} \in \mathbb{R}^{6 \times 6}$\\
    \scriptsize Piezoelectric: $\mathbf{e} \in \mathbb{R}^{3 \times 6}$\\
    \scriptsize Invariant Energy: $E \in \mathbb{R}$
};

% Arrows and Connections
\draw[flow_arrow] (input_cell) -- node[above, font=\scriptsize\bfseries, text=dark!70] {Periodic $\mathrm{k}$-d Tree} (graph_build);
\draw[flow_arrow] (input_cell) -- (irreps_embed);
\draw[flow_arrow] (graph_build) |- (tp_block);
\draw[flow_arrow] (irreps_embed) -- node[above, font=\scriptsize\bfseries, text=purple] {$\mathrm{O}(3)$-Equivariant} (tp_block);
\draw[flow_arrow] (tp_block) -- node[above, font=\scriptsize\bfseries, text=secondary] {Message Aggr.} (flow_field);
\draw[flow_arrow] (tp_block) -- (prop_readout);
\draw[flow_arrow] (flow_field) -- (prop_readout);

% Background Group Frames
\begin{scope}[on background layer]
    \node[draw=primary!40, dashed, rounded corners=10pt, fill=primary!3, fit=(input_cell)(graph_build), inner sep=12pt, label={[primary, font=\sffamily\bfseries\footnotesize]above:Phase I: Periodic Boundary Graph Formulation}] {};
    \node[draw=purple!40, dashed, rounded corners=10pt, fill=purple!3, fit=(irreps_embed)(tp_block), inner sep=12pt, label={[purple, font=\sffamily\bfseries\footnotesize]below:Phase II: Equivariant Spherical Message Passing ($\mathrm{SE}(3) \times \mathbb{Z}^3$ Invariance)}] {};
    \node[draw=accent!40, dashed, rounded corners=10pt, fill=accent!3, fit=(flow_field)(prop_readout), inner sep=12pt, label={[accent, font=\sffamily\bfseries\footnotesize]above:Phase III: Physical Response & Generative Drift}] {};
\end{scope}

\end{tikzpicture}
""")
        if include_preamble:
            lines.append(r"\end{document}")
        return "\n".join(lines)
