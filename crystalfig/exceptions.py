"""Exception hierarchy for crystalfig."""


class CrystalFigError(Exception):
    """Base exception for all crystalfig errors."""


class StructureParseError(CrystalFigError):
    """Raised when a structure cannot be parsed from input."""


class SymmetryError(CrystalFigError):
    """Raised when symmetry analysis fails."""


class NeighborError(CrystalFigError):
    """Raised when neighbor detection fails."""


class GeometryError(CrystalFigError):
    """Raised when geometric construction fails."""


class RenderError(CrystalFigError):
    """Raised when rendering fails."""


class ExportError(CrystalFigError):
    """Raised when export fails."""


class LatexCompilationError(ExportError):
    """Raised when LaTeX compilation fails."""


class OptionalDependencyError(CrystalFigError):
    """Raised when an optional dependency is missing."""

    def __init__(self, package: str, extra: str = ""):
        msg = f"Optional dependency '{package}' is required for this operation."
        if extra:
            msg += f" Install it with: pip install crystalfig[{extra}]"
        super().__init__(msg)


class InvalidStyleError(CrystalFigError):
    """Raised when a style configuration is invalid."""
