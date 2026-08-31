"""Data-driven equivariant GNN diagram generator."""

from __future__ import annotations

from dataclasses import dataclass, field

from crystalfig.styles.palette import get_palette


@dataclass
class EquivariantGNNDiagram:
    """Generate TikZ diagrams of equivariant message-passing architectures."""

    title: str = "Equivariant Message Passing & Flow Matching Architecture"
    palette_name: str = "muted"
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    def add_node(self, node_id: str, label: str, style: str = "crystal", position: tuple | None = None):
        self.nodes.append({"id": node_id, "label": label, "style": style, "position": position})
        return self

    def add_edge(self, from_id: str, to_id: str, label: str = ""):
        self.edges.append({"from": from_id, "to": to_id, "label": label})
        return self

    def to_tikz(self, standalone: bool = False) -> str:
        palette = get_palette(self.palette_name)
        lines = []
        if standalone:
            lines.append(r"\documentclass[tikz,border=10pt]{standalone}")
            lines.append(r"\usepackage{tikz}")
            lines.append(r"\usepackage{amsmath,amssymb}")
            lines.append(r"\usetikzlibrary{shapes.geometric,arrows.meta,calc,positioning,shadows.blur,backgrounds,fit}")
            lines.append(r"\begin{document}")

        for name, rgb in palette.accents.items():
            lines.append(f"\\definecolor{{{name}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")

        lines.append(r"""
\begin{tikzpicture}[
    >=Stealth,
    node distance=1.4cm and 1.8cm,
    font=\sffamily,
    box_base/.style={rectangle, rounded corners=6pt, align=center, inner sep=8pt, line width=1pt,
        blur shadow={shadow blur steps=4, shadow xshift=0.5mm, shadow yshift=-0.5mm, shadow opacity=15}},
    crystal_box/.style={box_base, fill=primary!8, draw=primary!80, text=dark, font=\sffamily\bfseries\small},
    irreps_box/.style={box_base, fill=purple!8, draw=purple!80, text=dark, font=\sffamily\bfseries\small},
    tp_box/.style={box_base, fill=secondary!8, draw=secondary!80, text=dark, font=\sffamily\bfseries\small},
    flow_box/.style={box_base, fill=accent!8, draw=accent!80, text=dark, font=\sffamily\bfseries\small},
    flow_arrow/.style={->, very thick, dark!75, line width=1.5pt}
]

\node[crystal_box, minimum width=4.2cm, minimum height=2.2cm] (input_cell) {
    \textbf{Periodic Crystal} $\mathcal{C}$\\
    \vspace{2pt}
    \scriptsize Basis $\mathbf{L} = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3]$\\
    \scriptsize Positions $\mathbf{x}_i = \mathbf{L}\mathbf{s}_i \in \mathbb{R}^3$
};

\node[crystal_box, right=2.0cm of input_cell, minimum width=4.4cm, minimum height=2.2cm] (graph_build) {
    \textbf{Periodic Graph}\\
    \vspace{3pt}
    $\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + \mathbf{L}\mathbf{k}$\\
    \scriptsize $d_{ij} = \|\mathbf{r}_{ij}\| \le r_{\text{cut}}$
};

\node[irreps_box, below=1.6cm of input_cell, minimum width=4.2cm, minimum height=2.6cm] (irreps_embed) {
    \textbf{Irrep Embeddings}\\
    \vspace{3pt}
    \scriptsize Node Feats ($l=0,1,2$):\\
    $\mathbf{h}_i^{(0)} = \bigoplus_l \mathbf{f}_i^{(l)}$\\
    \vspace{2pt}
    \scriptsize $Y_l(\hat{\mathbf{r}}_{ij}) \in \mathbb{R}^{2l+1}$
};

\node[tp_box, right=2.0cm of irreps_embed, minimum width=4.4cm, minimum height=2.6cm] (tp_block) {
    \textbf{Clebsch-Gordan TP}\\
    \vspace{3pt}
    $\mathbf{m}_{ij} = (\mathbf{h}_j \otimes_{\mathrm{cg}} \mathbf{Y}(\hat{\mathbf{r}})) \cdot \mathcal{R}(d)$\\
    \vspace{2pt}
    \scriptsize Equivariant Interaction
};

\node[flow_box, right=2.0cm of tp_block, minimum width=4.0cm, minimum height=2.6cm] (flow_field) {
    \textbf{Equivariant Flow Drift}\\
    \vspace{3pt}
    \scriptsize Vector Field ($l=1$):\\
    $\mathbf{v}_i = \mathbf{h}_i^{(l=1)}$\\
    \vspace{2pt}
    \scriptsize Lattice Strain Rate
};

\node[flow_box, right=2.0cm of graph_build, minimum width=4.0cm, minimum height=2.2cm] (prop_readout) {
    \textbf{Response Tensors}\\
    \vspace{2pt}
    \scriptsize $\mathbf{C}_{ijkl}$, $e_{ijk}$, $E$
};

\draw[flow_arrow] (input_cell) -- node[above, font=\scriptsize\bfseries, text=dark!70] {Periodic Graph} (graph_build);
\draw[flow_arrow] (input_cell) -- (irreps_embed);
\draw[flow_arrow] (graph_build) |- (tp_block);
\draw[flow_arrow] (irreps_embed) -- node[above, font=\scriptsize\bfseries, text=purple] {$\mathrm{O}(3)$} (tp_block);
\draw[flow_arrow] (tp_block) -- node[above, font=\scriptsize\bfseries, text=secondary] {Aggr.} (flow_field);
\draw[flow_arrow] (tp_block) -- (prop_readout);
\draw[flow_arrow] (flow_field) -- (prop_readout);

\begin{scope}[on background layer]
    \node[draw=primary!40, dashed, rounded corners=10pt, fill=primary!3, fit=(input_cell)(graph_build), inner sep=12pt,
        label={[primary, font=\sffamily\bfseries\footnotesize]above:Stage 1}] {};
    \node[draw=purple!40, dashed, rounded corners=10pt, fill=purple!3, fit=(irreps_embed)(tp_block), inner sep=12pt,
        label={[purple, font=\sffamily\bfseries\footnotesize]below:Stage 2}] {};
    \node[draw=accent!40, dashed, rounded corners=10pt, fill=accent!3, fit=(flow_field)(prop_readout), inner sep=12pt,
        label={[accent, font=\sffamily\bfseries\footnotesize]above:Stage 3}] {};
\end{scope}

\end{tikzpicture}
""")
        if standalone:
            lines.append(r"\end{document}")
        return "\n".join(lines)
