"""crystalfig: Publication-grade crystal structure visualization toolkit."""

from crystalfig._version import __version__
from crystalfig.exceptions import (
    CrystalFigError,
    ExportError,
    LatexCompilationError,
    OptionalDependencyError,
    RenderError,
    StructureParseError,
)
from crystalfig.figure.builder import CrystalFigure, plot_structure
from crystalfig.model.lattice import Lattice
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure

__all__ = [
    "__version__",
    "CrystalFigure",
    "plot_structure",
    "CrystalStructure",
    "Site",
    "Lattice",
    "CrystalFigError",
    "OptionalDependencyError",
    "StructureParseError",
    "RenderError",
    "ExportError",
    "LatexCompilationError",
]
