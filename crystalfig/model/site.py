"""Crystal site with position, species, and properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crystalfig.model.properties import SiteProperties


@dataclass
class Site:
    """A single crystallographic site.

    Attributes:
        frac_coords: Fractional coordinates within the lattice cell.
        species: Element symbol or ordered dict of species:occupancy.
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
        if isinstance(self.species, dict):
            # Normalize occupancy sum
            total = sum(self.species.values())
            if abs(total - 1.0) > 1e-3 and total > 0:
                self.species = {k: v / total for k, v in self.species.items()}

    # ------------------------------------------------------------------
    # Species helpers
    # ------------------------------------------------------------------
    @property
    def dominant_species(self) -> str:
        """Return the species with highest occupancy."""
        if isinstance(self.species, str):
            return self.species
        return max(self.species, key=lambda k: self.species[k])

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
