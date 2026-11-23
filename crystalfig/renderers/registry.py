"""Renderer lookup and format capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from crystalfig.renderers.matplotlib_3d_renderer import Matplotlib3DRenderer
from crystalfig.renderers.matplotlib_renderer import MatplotlibRenderer
from crystalfig.renderers.pyvista_renderer import PyVistaRenderer
from crystalfig.renderers.svg_renderer import SvgRenderer
from crystalfig.renderers.tikz_renderer import TikzRenderer


@dataclass(frozen=True)
class RendererCapabilities:
    formats: frozenset[str]
    vector_formats: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RendererEntry:
    create: Callable
    capabilities: RendererCapabilities


_RENDERERS = {
    "matplotlib": RendererEntry(
        MatplotlibRenderer,
        RendererCapabilities(frozenset({"pdf", "png", "tif", "tiff", "eps", "pgf"}), frozenset({"pdf", "eps", "pgf"})),
    ),
    "matplotlib3d": RendererEntry(
        Matplotlib3DRenderer,
        RendererCapabilities(frozenset({"pdf", "svg", "png", "tif", "tiff", "eps", "pgf"}), frozenset({"pdf", "svg", "eps", "pgf"})),
    ),
    "svg": RendererEntry(
        SvgRenderer,
        RendererCapabilities(frozenset({"svg"}), frozenset({"svg"})),
    ),
    "tikz": RendererEntry(
        TikzRenderer,
        RendererCapabilities(frozenset({"tex", "tikz"}), frozenset({"tex", "tikz"})),
    ),
    "pyvista": RendererEntry(
        PyVistaRenderer,
        RendererCapabilities(frozenset({"png"})),
    ),
}


def get_renderer(name: str, camera=None) -> tuple[object, RendererCapabilities]:
    """Create a renderer and return its declared output capabilities."""
    try:
        entry = _RENDERERS[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_RENDERERS))
        raise ValueError(f"Unknown renderer backend {name!r}; use {supported}.") from exc
    return entry.create(camera=camera), entry.capabilities
