"""Crystal site with position, species, and properties."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crystalfig.model.properties import SiteProperties

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def element_symbol(species: str) -> str:
    """Extract element symbol from a species string (e.g. 'Fe2+' -> 'Fe')."""
    match = _ELEMENT_RE.match(species.strip())
    return match.group(1) if match else species


@dataclass
class Site:
    """A single crystallographic site.

    Attributes:
        frac_coords: Fractional coordinates within the lattice cell.
        species: Element symbol (optionally with oxidation state, e.g. 'O2-')
            or an ordered dict of species:occupancy.
        properties: Arbitrary site properties (magmom, force, etc.).
        source_index: Index of the original site in the source structure.
        image_offset: Integer periodic image offset (i, j, k).
        wyckoff: Wyckoff letter if known.
        label: Optional custom label.
    """

    frac_coords: np.ndarray
    species: str | dict[str, float]
    properties: SiteProperties = field(default_factory=SiteProperties)
    source_index: int = -1
    image_offset: tuple[int, int, int] = (0, 0, 0)
    wyckoff: str | None = None
    label: str | None = None

    def __post_init__(self):
        self.frac_coords = np.asarray(self.frac_coords, dtype=float)
        # Occupancy is scientific data: do NOT silently normalize it.
        if isinstance(self.species, dict):
            total = sum(self.species.values())
            if total <= 0:
                raise ValueError("Site occupancy sum must be positive.")

    # ------------------------------------------------------------------
    # Species helpers
    # ------------------------------------------------------------------
    @property
    def dominant_species(self) -> str:
        """Return the species with highest occupancy (includes oxidation state)."""
        if isinstance(self.species, str):
            return self.species
        return max(self.species, key=lambda k: self.species[k])

    @property
    def dominant_element(self) -> str:
        """Return the element symbol of the dominant species."""
        return element_symbol(self.dominant_species)

    @property
    def occupancy(self) -> dict[str, float]:
        if isinstance(self.species, str):
            return {self.species: 1.0}
        return dict(self.species)

    @property
    def is_ordered(self) -> bool:
        return isinstance(self.species, str)

    @property
    def is_disordered(self) -> bool:
        return not self.is_ordered

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    def cart_coords(self, lattice) -> np.ndarray:
        return lattice.frac_to_cart(self.frac_coords)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def identity_key(self) -> tuple[int, tuple[int, int, int]]:
        return (self.source_index, self.image_offset)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frac_coords": self.frac_coords.tolist(),
            "species": self.species,
            "properties": self.properties.as_dict(),
            "source_index": self.source_index,
            "image_offset": list(self.image_offset),
            "wyckoff": self.wyckoff,
            "label": self.label,
        }
