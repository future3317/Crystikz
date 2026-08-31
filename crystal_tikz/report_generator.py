import numpy as np
import subprocess
from crystal_tikz import LatticeTikZ, EquivariantGNNTikZ


def build_perovskite(a=3.95, c=4.05):
    """Build a BaTiO3 tetragonal perovskite visualizer."""
    vis = LatticeTikZ(a=a, b=a, c=c, alpha=90, beta=90, gamma=90,
                      camera_elevation=24, camera_azimuth=42, scale=1.0)
    corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]
    for pt in corners:
        vis.add_atom("Ba", pt, color="primary", radius=0.22, label="Ba")
    ti_idx = vis.add_atom("Ti", [0.5, 0.5, 0.5], color="secondary", radius=0.19, label="Ti")
    o_faces = [
        [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
        [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],
        [0.0, 0.5, 0.5], [1.0, 0.5, 0.5]
    ]
    o_indices = []
    for pt in o_faces:
        idx = vis.add_atom("O", pt, color="accent", radius=0.15, label="O")
        o_indices.append(idx)
        vis.add_bond(ti_idx, idx, color="secondary!70", width="thick")
    octa_faces = [
        [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
        [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5]
    ]
    vis.add_polyhedron(center_idx=ti_idx, vertex_indices=o_indices,
                       faces=octa_faces, fill_color="accent", fill_opacity=0.25)
    vis.add_vector([0.5, 0.5, 0.5], [0.5, 0.5, 0.92], color="amber",
                   label=r"$\mathbf{P}_{\mathrm{piezo}}$", width="very thick")
    return vis


def build_rutile(a=4.59, c=2.96):
    """Build a TiO2 rutile visualizer."""
    vis = LatticeTikZ(a=a, b=a, c=c, alpha=90, beta=90, gamma=90,
                      camera_elevation=26, camera_azimuth=48, scale=1.0)
    corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]
    for pt in corners:
        vis.add_atom("Ti", pt, color="secondary", radius=0.20, label="Ti")
    vis.add_atom("Ti", [0.5, 0.5, 0.5], color="secondary", radius=0.20, label="Ti")
    u = 0.305
    o_positions = [
        [u, u, 0.0], [1 - u, 1 - u, 0.0],
        [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5]
    ]
    for pt in o_positions:
        vis.add_atom("O", pt, color="accent", radius=0.15, label="O")
    return vis


# Generate visualizers
p_vis = build_perovskite(a=3.95, c=4.05)
r_vis = build_rutile(a=4.59, c=2.96)
gnn_vis = EquivariantGNNTikZ()

p_tikz = p_vis.generate_tikz(title=r"$\mathrm{BaTiO}_3$ ($P4mm$)")
r_tikz = r_vis.generate_tikz(title=r"$\mathrm{TiO}_2$ Rutile ($P4_2/mnm$)")
gnn_tikz = gnn_vis.generate_message_passing_pipeline()

report_tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage{tikz}
\usepackage{amsmath,amssymb}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{listings}
\usepackage{fancyhdr}
\usepackage{hyperref}

\pagestyle{fancy}
\fancyhf{}
\rhead{\textcolor{gray}{\footnotesize AI4Science Publication Toolkit: \texttt{crystal\_tikz}}}
\lhead{\textcolor{gray}{\footnotesize High-Fidelity Crystal & Equivariant GNN Vector Graphics}}
\cfoot{\thepage}

\usetikzlibrary{shapes,shapes.geometric,arrows.meta,calc,shadings,shadows.blur,backgrounds,fit,positioning}

% Define Nature-style colors
\definecolor{primary}{RGB}{44,95,138}
\definecolor{secondary}{RGB}{196,90,74}
\definecolor{accent}{RGB}{67,147,108}
\definecolor{amber}{RGB}{217,131,36}
\definecolor{purple}{RGB}{117,112,179}
\definecolor{gray}{RGB}{140,150,160}
\definecolor{dark}{RGB}{40,44,52}
\definecolor{poly_fill}{RGB}{67,147,108}
\definecolor{poly_fill2}{RGB}{196,90,74}

\lstset{
    backgroundcolor=\color{black!3},
    basicstyle=\ttfamily\scriptsize,
    keywordstyle=\color{primary}\bfseries,
    commentstyle=\color{gray}\itshape,
    stringstyle=\color{accent},
    frame=single,
    rulecolor=\color{gray!30},
    breaklines=true,
    numbers=none,
    tabsize=2
}

\title{\vspace{-0.8cm}\textbf{\LARGE \texttt{crystal\_tikz}: Modular Nature-Grade Vector Visualizer}\\[4pt]
\large Automated 3D Crystal Lattices, Coordination Polyhedra & Equivariant AI Pipelines}
\author{\textbf{Generative Geometric Deep Learning for Periodic Materials}}
\date{\today}

\begin{document}
\maketitle

\vspace{-0.3cm}
\begin{abstract}
\noindent We introduce \texttt{crystal\_tikz}, a Python framework designed to automatically generate publication-grade vector graphics (LaTeX TikZ) for solid-state crystallography and $\mathrm{E}(3) \times \mathbb{Z}^3$-equivariant geometric deep learning models. By combining fractional-to-Cartesian projective geometry, depth-sorted Painter's rendering, translucent coordination cages, and modular equivariant flow diagram builders, \texttt{crystal\_tikz} bridges crystal structures and generative neural architectures into single-source LaTeX figures suitable for \textit{Nature}, \textit{Science}, \textit{PRL}, and top-tier AI venues (\textit{ICLR}, \textit{NeurIPS}).
\end{abstract}

\vspace{-0.2cm}
\section{Visual Showcase: 3D Unit Cells & Equivariant Message Passing}

\begin{figure}[htbp]
\centering
\begin{subfigure}[b]{0.32\textwidth}
\centering
""" + p_tikz + r"""
\caption{Tetragonal Perovskite ($\mathrm{TiO}_6$ Octahedron + $\mathbf{P}_{\mathrm{piezo}}$)}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
\centering
""" + r_tikz + r"""
\caption{Rutile Unit Cell with Wyckoff $4f$ Sites}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.34\textwidth}
\centering
\begin{tikzpicture}[scale=0.9, >=Stealth, font=\sffamily]
  \node[draw=primary, fill=primary!6, rounded corners=5pt, inner sep=6pt, text width=4.5cm, align=left] (feat) {
    \textbf{1. Irrep State Decomposition:}\\
    $\mathbf{h}_i = \mathbf{f}_i^{(l=0)} \oplus \mathbf{f}_i^{(l=1)} \oplus \mathbf{f}_i^{(l=2)}$\\[3pt]
    \textbf{2. Periodic Metric:}\\
    $\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + \mathbf{L}\mathbf{k}_{ij}$\\[3pt]
    \textbf{3. Clebsch-Gordan TP:}\\
    $\mathbf{m}_{ij} = (\mathbf{h}_j \otimes_{\mathrm{cg}} Y_l(\hat{\mathbf{r}}_{ij})) \cdot \mathcal{R}(d_{ij})$
  };
  \node[draw=purple, fill=purple!6, rounded corners=5pt, inner sep=6pt, text width=4.5cm, align=left, below=0.4cm of feat] (resp) {
    \textbf{4. Response Tensors & Drift:}\\
    $\mathbf{v}_i = \sum_j \mathbf{m}_{ij}^{(l=1)}$ \hfill \scriptsize (Velocity)\\
    $\mathbf{e}_{ijk} = \sum_i \mathbf{h}_{i, j}^{(l=1)} \otimes \mathbf{h}_{i, k}^{(l=2)}$ \hfill \scriptsize (Piezo)\\
    $\dot{\mathbf{L}} = \frac{1}{N}\sum_i \mathbf{x}_i \otimes \mathbf{v}_i$ \hfill \scriptsize (Strain)
  };
  \draw[->, very thick, dark!70] (feat) -- (resp);
\end{tikzpicture}
\caption{Equivariant Tensor Formulations}
\end{subfigure}

\vspace{0.3cm}
\begin{subfigure}[b]{0.98\textwidth}
\centering
\resizebox{0.96\textwidth}{!}{
""" + gnn_tikz + r"""
}
\caption{Full Architecture: Periodic Boundary Equivariant Tensor Product Message Passing & Generative Flow Matching.}
\end{subfigure}

\caption{\textbf{End-to-End Vector Visualizations Generated Automatically by \texttt{crystal\_tikz}}. (a-b) 3D unit cells rendered with depth sorting, low-saturation Nature color palette, and vector field overlays. (c-d) Periodic Equivariant GNN interaction architecture with spherical harmonics and physical tensor prediction modules.}
\label{fig:full_overview}
\end{figure}

\vspace{-0.2cm}
\section{Package Architecture & Python API Usage}

\begin{lstlisting}[language=Python]
from crystal_tikz import LatticeTikZ, EquivariantGNNTikZ
import subprocess

# 1. Instantiate 3D Crystal Visualizer
vis = LatticeTikZ(a=3.95, b=3.95, c=4.05, alpha=90, beta=90, gamma=90,
                  camera_elevation=25, camera_azimuth=40)

# Add Atoms (A-site, B-site, Anion) & Coordination Polyhedra
vis.add_atom("Ba", [0, 0, 0], color="primary", radius=0.22)
b_idx = vis.add_atom("Ti", [0.5, 0.5, 0.5], color="secondary", radius=0.19)
vis.add_atom("O", [0.5, 0.5, 0.0], color="accent", radius=0.15)
vis.add_vector([0.5, 0.5, 0.5], [0.5, 0.5, 0.95], color="amber", label="P_piezo")

# 2. Export pure LaTeX TikZ string or auto-compile to publication PDF
tikz_code = vis.generate_tikz(title="BaTiO3 Unit Cell", include_preamble=True)
with open("figure.tex", "w") as f:
    f.write(tikz_code)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "figure.tex"])
\end{lstlisting}

\end{document}
"""

with open("AI4Crystal_TikZ_Toolkit_Report.tex", "w") as f:
    f.write(report_tex)

res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "AI4Crystal_TikZ_Toolkit_Report.tex"],
                     capture_output=True, text=True)
print("Report compilation return code:", res.returncode)
if res.returncode != 0:
    print("Error:", res.stdout[-1500:])
else:
    print("AI4Crystal_TikZ_Toolkit_Report.pdf generated successfully!")
