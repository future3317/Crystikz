"""Small, renderer-independent configuration for a crystal figure."""

from __future__ import annotations

from dataclasses import dataclass, field

from crystalfig.planning.display import DisplayConfig
from crystalfig.styles.specs import FigureStyle


@dataclass(frozen=True)
class CameraConfig:
    mode: str = "auto"
    projection: str = "orthographic"
    padding: float = 0.08
    elevation: float | None = None
    azimuth: float | None = None
    roll: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "fixed"}:
            raise ValueError("camera mode must be 'auto' or 'fixed'")
        if self.projection not in {"orthographic", "perspective"}:
            raise ValueError("projection must be 'orthographic' or 'perspective'")
        if not 0 <= self.padding < 1:
            raise ValueError("camera padding must be in [0, 1)")


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "pyvista"
    quality: str = "publication"
    dpi: int = 600
    width_mm: float = 89.0
    height_mm: float | None = None
    transparent: bool | None = None

    def __post_init__(self) -> None:
        if self.quality not in {"preview", "standard", "publication"}:
            raise ValueError("quality must be 'preview', 'standard', or 'publication'")
        if self.dpi < 1 or self.width_mm <= 0 or (self.height_mm is not None and self.height_mm <= 0):
            raise ValueError("dpi and figure dimensions must be positive")


@dataclass(frozen=True)
class FigureConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    style: str | FigureStyle = "journal_clean"
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
