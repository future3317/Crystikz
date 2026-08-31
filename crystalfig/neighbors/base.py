"""Base protocol for neighbor detection strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from crystalfig.model.structure import CrystalStructure


@dataclass
class NeighborBond:
    """A bond between two sites with periodic-image information."""

    i: int
    j: int
    jimage: tuple[int, int, int]
    distance: float
    weight: float = 1.0


class NeighborStrategy(Protocol):
    """Protocol for neighbor detection strategies."""

    def get_bonds(self, structure: CrystalStructure) -> list[NeighborBond]:
        """Return a list of bonds for the structure."""
        ...
