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
        counts: dict[str, int] = {}
        for site in self.sites:
            sp = site.dominant_species
            counts[sp] = counts.get(sp, 0) + 1
        parts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return "".join(f"{k}{v if v > 1 else ''}" for k, v in parts)

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
        seen = set()
        for site in self.sites:
            seen.add(site.dominant_species)
        return sorted(seen)

    def indices_of_species(self, species: str) -> list[int]:
        return [i for i, site in enumerate(self.sites) if site.dominant_species == species]

    def add_site(self, site: Site) -> int:
        self.sites.append(site)
        return len(self.sites) - 1

    def get_site(self, index: int) -> Site:
        return self.sites[index]

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    def make_supercell(self, scaling: int | tuple[int, int, int] | np.ndarray) -> CrystalStructure:
        """Create a supercell by integer scaling along each lattice vector."""
        if isinstance(scaling, int):
            sc = np.diag([scaling, scaling, scaling])
        elif isinstance(scaling, tuple):
            sc = np.diag(scaling)
        else:
            sc = np.asarray(scaling, dtype=int).reshape(3, 3)

        new_lattice = Lattice(self.lattice.supercell_matrix(sc))
        new_sites: list[Site] = []

        # Generate all integer shifts within the supercell
        for i, j, k in np.ndindex(*sc.diagonal()):
            offset = (int(i), int(j), int(k))
            for orig_site in self.sites:
                new_frac = (orig_site.frac_coords + np.array(offset)) / sc.diagonal()
                new_site = Site(
                    frac_coords=new_frac,
                    species=orig_site.species,
                    properties=SiteProperties.from_dict(orig_site.properties.as_dict()),
                    source_index=orig_site.source_index,
                    image_offset=offset,
                    wyckoff=orig_site.wyckoff,
                    label=orig_site.label,
                )
                new_sites.append(new_site)

        return CrystalStructure(lattice=new_lattice, sites=new_sites, properties=self.properties.copy())

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
