import numpy as np
import subprocess
from crystal_tikz import LatticeTikZ, EquivariantGNNTikZ

# 1. Generate Perovskite Unit Cell (e.g. BaTiO3 / SrTiO3)
# Sr at 8 corners (0,0,0)
# Ti at 1 body center (0.5, 0.5, 0.5)
# O at 6 face centers (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), etc.

lat = LatticeTikZ(a=3.95, b=3.95, c=4.05, alpha=90, beta=90, gamma=90, camera_elevation=25, camera_azimuth=40, scale=1.0)

# Add 8 Corner A-site atoms (Ba / Sr)
corner_coords = [
    [0,0,0], [1,0,0], [1,1,0], [0,1,0],
    [0,0,1], [1,0,1], [1,1,1], [0,1,1]
]
a_indices = []
for pt in corner_coords:
    idx = lat.add_atom("Ba", pt, color="primary", radius=0.24, label="Ba")
    a_indices.append(idx)

# Add 1 B-site atom (Ti) at center
ti_idx = lat.add_atom("Ti", [0.5, 0.5, 0.5], color="secondary", radius=0.20, label="Ti")

# Add 6 O atoms at face centers
o_faces = [
    [0.5, 0.5, 0.0], # bottom
    [0.5, 0.5, 1.0], # top
    [0.5, 0.0, 0.5], # front
    [0.5, 1.0, 0.5], # back
    [0.0, 0.5, 0.5], # left
    [1.0, 0.5, 0.5]  # right
]
o_indices = []
for pt in o_faces:
    idx = lat.add_atom("O", pt, color="accent", radius=0.16, label="O")
    o_indices.append(idx)
    # add Ti-O bond
    lat.add_bond(ti_idx, idx, color="secondary!70", width="thick")

# Define Octahedron faces using vertex indices in o_indices:
# o_indices: 0=bottom, 1=top, 2=front, 3=back, 4=left, 5=right
octa_faces = [
    [0, 2, 4], # bottom-front-left
    [0, 2, 5], # bottom-front-right
    [0, 3, 4], # bottom-back-left
    [0, 3, 5], # bottom-back-right
    [1, 2, 4], # top-front-left
    [1, 2, 5], # top-front-right
    [1, 3, 4], # top-back-left
    [1, 3, 5], # top-back-right
]

lat.add_polyhedron(center_idx=ti_idx, vertex_indices=o_indices, faces=octa_faces, fill_color="accent", fill_opacity=0.25)

# Add Polarization / Equivariant Vector Field Arrow on Ti
lat.add_vector([0.5, 0.5, 0.5], [0.5, 0.5, 0.9], color="purple", label=r"$\mathbf{P}_{\mathrm{piezo}}$", width="line width=2pt")

unit_cell_tikz = lat.generate_tikz(include_preamble=False, title=r"$\mathrm{BaTiO}_3$ Tetragonal Unit Cell")

print("Generated unit cell TikZ length:", len(unit_cell_tikz))

# 2. Compile a standalone presentation / showcase PDF containing both figures
gnn_gen = EquivariantGNNTikZ()
gnn_tikz = gnn_gen.generate_message_passing_pipeline(include_preamble=False)

full_doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{tikz}
\usepackage{amsmath,amssymb}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{microtype}
\usetikzlibrary{shapes.geometric,arrows.meta,calc,positioning,shadows.blur,backgrounds,fit}

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

\title{\textbf{\LARGE Automated Publication-Quality TikZ Generation for AI4Materials:}\\[4pt]
\large 3D Crystal Lattices, Coordination Polyhedra & Equivariant Flow Architectures}
\author{\textbf{AI for Crystal Generation & Equivariant Geometric Deep Learning Toolbox}}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
\noindent High-impact scientific publications in computational materials science and generative geometric deep learning require precise, scalable, and publication-ready vector visualizations. Here we present a modular Python architecture designed for the automatic synthesis of LaTeX TikZ representations encompassing (i) 3D periodic crystal unit cells with Wyckoff fractional positioning, depth-sorted coordination polyhedra, and physical vector fields, and (ii) rigorous $\mathrm{E}(3) \times \mathbb{Z}^3$ equivariant message-passing architectures incorporating spherical harmonics and Clebsch-Gordan tensor product operators.
\end{abstract}

\vspace{0.5cm}

\section*{Figure 1: 3D Crystal Structure & Equivariant Neural Pipeline}

\begin{figure}[htbp]
\centering
\begin{subfigure}[b]{0.40\textwidth}
\centering
""" + unit_cell_tikz + r"""
\caption{3D Perovskite Unit Cell ($\mathrm{BaTiO}_3$) with $\mathrm{TiO}_6$ Octahedral Cage & Polarization $\mathbf{P}$.}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.56\textwidth}
\centering
\resizebox{\textwidth}{!}{
""" + gnn_tikz + r"""
}
\caption{Periodic Equivariant Message Passing and Flow-Matching Drift Prediction Architecture.}
\end{subfigure}
\caption{\textbf{High-fidelity modular TikZ illustrations generated directly via Python for AI4Science publications.} \textbf{(a)} Real-space depth-sorted crystal cell with fractional coordinate projection, coordination octahedra, and lattice vectors. \textbf{(b)} Equivariant tensor product graph neural network architecture mapping periodic graphs to physical tensors and generative flow fields.}
\label{fig:main_showcase}
\end{figure}

\end{document}
"""

with open("showcase.tex", "w") as f:
    f.write(full_doc)

res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "showcase.tex"], capture_output=True, text=True)
print("pdflatex Return code:", res.returncode)
if res.returncode != 0:
    print("STDOUT error snippet:\n", res.stdout[-1500:])
else:
    print("Success! showcase.pdf generated successfully!")
