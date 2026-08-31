"""Base renderer protocol and options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from crystalfig.scene.scene import Scene
from crystalfig.styles.theme import FigureTheme


@dataclass
class RenderOptions:
    """Options passed to a renderer."""

    width: float = 89.0  # mm
    height: float | None = None  # mm
    dpi: int = 300
    transparent: bool = False
    background: str | None = None
    title: str | None = None
    show_legend: bool = True


class Renderer(Protocol):
    """Protocol for all renderers."""

    def render(self, scene: Scene, theme: FigureTheme, options: RenderOptions) -> str:
        """Render scene to a string representation (e.g. TikZ or SVG path data)."""
        ...

    def export(self, scene: Scene, path: str, theme: FigureTheme, options: RenderOptions) -> None:
        """Export scene directly to a file."""
        ...
