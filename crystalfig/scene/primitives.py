"""Backend-independent scene primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Primitive:
    """Base class for all scene primitives."""

    color: str | tuple[float, float, float, float] = "black"
    opacity: float = 1.0
    visible: bool = True
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    layer: str = "default"


@dataclass
class Sphere(Primitive):
    """A sphere representing an atom or marker."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.2
    render_style: str = "shaded"  # shaded, flat, wireframe

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class Cylinder(Primitive):
    """A cylinder representing a bond or tube."""

    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    radius: float = 0.05
    dashed: bool = False

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.end = np.asarray(self.end, dtype=float)


@dataclass
class Bond(Cylinder):
    """A chemical bond with periodic-image information."""

    site_i: int = -1
    site_j: int = -1
    jimage: tuple[int, int, int] = (0, 0, 0)
    distance: float = 0.0


@dataclass
class Line(Primitive):
    """A line segment."""

    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    linewidth: float = 1.0

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.end = np.asarray(self.end, dtype=float)


@dataclass
class Polyline(Primitive):
    """A connected sequence of line segments."""

    points: list[np.ndarray] = field(default_factory=list)
    linewidth: float = 1.0
    closed: bool = False

    def __post_init__(self):
        self.points = [np.asarray(p, dtype=float) for p in self.points]


@dataclass
class Polygon(Primitive):
    """A filled polygon (e.g. Miller plane intersection)."""

    points: list[np.ndarray] = field(default_factory=list)
    fill_color: str | tuple[float, float, float, float] | None = None
    edge_color: str | tuple[float, float, float, float] | None = None
    linewidth: float = 0.5

    def __post_init__(self):
        self.points = [np.asarray(p, dtype=float) for p in self.points]


@dataclass
class Polyhedron(Primitive):
    """A coordination polyhedron defined by vertices and triangular faces."""

    center_site: int = -1
    vertices: list[np.ndarray] = field(default_factory=list)
    faces: list[list[int]] = field(default_factory=list)
    fill_color: str | tuple[float, float, float, float] | None = None
    edge_color: str | tuple[float, float, float, float] | None = None
    edge_width: float = 0.5

    def __post_init__(self):
        self.vertices = [np.asarray(v, dtype=float) for v in self.vertices]


@dataclass
class Arrow(Primitive):
    """A vector arrow (polarization, force, magnetic moment, etc.)."""

    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    shaft_radius: float = 0.03
    head_radius: float = 0.08
    head_length: float = 0.15
    normalize: bool = False

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)


@dataclass
class Plane(Primitive):
    """A crystallographic plane (e.g. Miller plane)."""

    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    width: float = 1.0
    height: float = 1.0

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)


@dataclass
class Text(Primitive):
    """Text label at a 3D position."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    text: str = ""
    fontsize: float = 10.0
    color: str = "black"
    halign: str = "center"
    valign: str = "center"
    raw_latex: bool = False

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class CellEdge(Line):
    """A unit-cell edge with front/back styling metadata."""

    is_back: bool = False


@dataclass
class Axis(Arrow):
    """A crystallographic axis arrow (a, b, c or arbitrary)."""

    name: str = ""


@dataclass
class LegendItem(Primitive):
    """Legend entry (symbol + text)."""

    symbol: str = "sphere"
    text: str = ""


@dataclass
class Group:
    """Group of primitives."""

    name: str = ""
    primitives: list[Any] = field(default_factory=list)
    visible: bool = True

    def add(self, primitive: Any) -> None:
        self.primitives.append(primitive)
