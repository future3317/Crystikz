"""Compatibility facades for legacy crystal_tikz APIs.

These aliases allow scripts using the old standalone module to keep working
while the package migrates to the new CrystalFigure / Scene architecture.
"""

from __future__ import annotations

import warnings

from crystalfig.export.latex import compile_tikz_to_pdf
from crystalfig.figure.builder import CrystalFigure
from crystalfig.model.lattice import Lattice as LatticeBasis
from crystalfig.scene.camera import Camera as Camera3D


def _warn_renamed(old: str, new: str):
    warnings.warn(f"{old} is deprecated; use {new} instead.", FutureWarning, stacklevel=3)


class CrystalVisualizer:
    """Legacy facade that delegates to CrystalFigure."""

    def __init__(
        self,
        a: float = 4.0,
        b: float = 4.0,
        c: float = 4.0,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        camera_elevation: float = 22.0,
        camera_azimuth: float = 42.0,
        scale: float = 1.0,
        palette_name: str = "nature_classic",
    ):
        _warn_renamed("CrystalVisualizer", "CrystalFigure")
        self.lattice = LatticeBasis.from_parameters(a, b, c, alpha, beta, gamma)
        self.camera = Camera3D(camera_elevation, camera_azimuth, scale)
        self._atoms = []
        self._bonds = []

    def add_atom(self, symbol, frac_coords, color="primary", radius=0.22, label=None, **kwargs):
        self._atoms.append((symbol, frac_coords, color, radius, label))
        return len(self._atoms) - 1

    def add_bond(self, i, j, color="gray", width="thick", opacity=0.85, dashed=False):
        self._bonds.append((i, j, color, width, opacity, dashed))

    def to_tikz(self, **kwargs):
        raise NotImplementedError("Legacy CrystalVisualizer.to_tikz is not implemented; use CrystalFigure instead.")


class EquivariantArchitectureVisualizer:
    """Legacy facade for equivariant GNN diagrams."""

    def __init__(self, title: str = ""):
        _warn_renamed("EquivariantArchitectureVisualizer", "EquivariantGNNDiagram")
        self.title = title

    def to_tikz(self, standalone: bool = False):
        from crystalfig.diagrams.equivariant import EquivariantGNNDiagram
        return EquivariantGNNDiagram(title=self.title).to_tikz(standalone=standalone)


def build_perovskite(*args, **kwargs):
    _warn_renamed("build_perovskite", "CrystalFigure examples")
    from crystalfig.examples.presets import perovskite_structure
    return CrystalFigure(perovskite_structure(*args, **kwargs))


def build_rutile(*args, **kwargs):
    _warn_renamed("build_rutile", "CrystalFigure examples")
    from crystalfig.examples.presets import rutile_structure
    return CrystalFigure(rutile_structure(*args, **kwargs))


def build_wurtzite(*args, **kwargs):
    _warn_renamed("build_wurtzite", "CrystalFigure examples")
    from crystalfig.examples.presets import wurtzite_structure
    return CrystalFigure(wurtzite_structure(*args, **kwargs))


__all__ = [
    "LatticeBasis",
    "Camera3D",
    "CrystalVisualizer",
    "EquivariantArchitectureVisualizer",
    "compile_tikz_to_pdf",
    "build_perovskite",
    "build_rutile",
    "build_wurtzite",
]
