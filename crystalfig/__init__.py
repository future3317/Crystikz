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
from crystalfig.figure import (
    CameraConfig,
    CrystalFigure,
    DisplayConfig,
    DisplayPlan,
    FigureConfig,
    FigureGrid,
    MotifConfig,
    RenderConfig,
    plot_structure,
)
from crystalfig.model.lattice import Lattice
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure
from crystalfig.planning.display import BondDisplayPolicy

__all__ = [
    "__version__",
    "CrystalFigure",
    "FigureGrid",
    "plot_structure",
    "DisplayConfig",
    "MotifConfig",
    "DisplayPlan",
    "BondDisplayPolicy",
    "CameraConfig",
    "RenderConfig",
    "FigureConfig",
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
