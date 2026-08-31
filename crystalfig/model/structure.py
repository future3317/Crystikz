"""Canonical internal crystal structure representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crystalfig.model.lattice import Lattice
from crystalfig.model.properties import SiteProperties
from crystalfig.model.site import Site


@dataclass
class CrystalStructure:
    """Backend-independent crystal structure.

    This is the canonical internal model used by all renderers.  It stores a
    lattice, a list of sites, and arbitrary structure-level properties.
    """

    lattice: Lattice
    sites: list[Site] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sites)

    @property
    def formula(self) -> str:
        """Reduced formula using pymatgen Composition conventions."""
        from pymatgen.core import Composition

        comp: dict[str, float] = {}
        for site in self.sites:
            for species, occ in site.occupancy.items():
                comp[species] = comp.get(species, 0.0) + occ
        return Composition(comp).reduced_formula

    @property
    def num_sites(self) -> int:
        return len(self.sites)

    @property
    def volume(self) -> float:
        return self.lattice.volume

    @property
    def cart_coords(self) -> np.ndarray:
        """Return all site Cartesian coordinates as (N, 3) array."""
        return np.array([site.cart_coords(self.lattice) for site in self.sites])

    @property
    def frac_coords(self) -> np.ndarray:
        return np.array([site.frac_coords for site in self.sites])

    # ------------------------------------------------------------------
    # Site queries
    # ------------------------------------------------------------------
    def unique_species(self) -> list[str]:
        """Return unique dominant species strings (including oxidation states)."""
        seen = set()
        for site in self.sites:
            seen.add(site.dominant_species)
        return sorted(seen)

    def indices_of_species(self, species: str) -> list[int]:
        return [i for i, site in enumerate(self.sites) if site.dominant_species == species]

    def indices_of_element(self, element: str) -> list[int]:
        return [i for i, site in enumerate(self.sites) if site.dominant_element == element]

    def add_site(self, site: Site) -> int:
        self.sites.append(site)
        return len(self.sites) - 1

    def get_site(self, index: int) -> Site:
        return self.sites[index]

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    def make_supercell(self, scaling: int | tuple[int, int, int] | np.ndarray) -> CrystalStructure:
        """Create a supercell from an integer scaling or 3×3 transformation.

        Supports a scalar ``n`` (n×n×n), a length-3 tuple (na×nb×nc), or an
        arbitrary integer 3×3 matrix.  The transformation is delegated to
        pymatgen so non-diagonal supercells are handled correctly.
        """
        if isinstance(scaling, int):
            sc_matrix = np.diag([scaling, scaling, scaling])
        elif isinstance(scaling, tuple):
            sc_matrix = np.diag(scaling)
        else:
            sc_matrix = np.asarray(scaling, dtype=int).reshape(3, 3)

        from crystalfig.io.pymatgen_adapter import from_pymatgen, to_pymatgen

        pmg = to_pymatgen(self)
        pmg.make_supercell(sc_matrix)
        result = from_pymatgen(pmg)
        result.properties = self.properties.copy()
        return result

    def translate_frac(self, vector: np.ndarray) -> CrystalStructure:
        """Return a copy translated by a fractional vector."""
        vector = np.asarray(vector, dtype=float)
        new_sites = []
        for site in self.sites:
            new_sites.append(Site(
                frac_coords=site.frac_coords + vector,
                species=site.species,
                properties=SiteProperties.from_dict(site.properties.as_dict()),
                source_index=site.source_index,
                image_offset=site.image_offset,
                wyckoff=site.wyckoff,
                label=site.label,
            ))
        return CrystalStructure(lattice=self.lattice, sites=new_sites, properties=self.properties.copy())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "lattice": self.lattice.matrix.tolist(),
            "sites": [s.as_dict() for s in self.sites],
            "properties": self.properties,
        }
