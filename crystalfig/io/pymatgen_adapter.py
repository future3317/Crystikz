"""Adapter between pymatgen Structure and CrystalStructure."""

from __future__ import annotations

import numpy as np

from crystalfig.model.lattice import Lattice
from crystalfig.model.properties import SiteProperties
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure


def from_pymatgen(structure) -> CrystalStructure:
    """Convert a pymatgen Structure to CrystalStructure."""
    lattice = Lattice(np.array(structure.lattice.matrix))
    sites = []
    for i, site in enumerate(structure):
        # Preserve oxidation state (e.g. "Fe2+", "O2-") and disordered compositions.
        if len(site.species) == 1:
            species = str(site.specie)
        else:
            species = {str(sp): float(occ) for sp, occ in site.species.items()}

        props = SiteProperties()
        for key, value in site.properties.items():
            if key == "magmom":
                props.magnetic_moment = value
            else:
                props.set(key, value)

        sites.append(Site(
            frac_coords=np.array(site.frac_coords),
            species=species,
            properties=props,
            source_index=i,
            label=getattr(site, "label", None),
        ))

    return CrystalStructure(lattice=lattice, sites=sites)


def _parse_pymatgen_species(sp_str: str):
    """Parse a species string into pymatgen Species or Element."""
    from pymatgen.core import Element, Species

    try:
        return Species.from_str(sp_str)
    except ValueError:
        return Element(sp_str)


def to_pymatgen(structure: CrystalStructure):
    """Convert a CrystalStructure to pymatgen Structure."""
    from pymatgen.core import Composition, Structure
    from pymatgen.core import Lattice as PmgLattice

    species = []
    coords = []
    props: dict[str, list] = {}
    for i, site in enumerate(structure.sites):
        occ = site.occupancy
        if len(occ) == 1:
            sp_str = next(iter(occ))
            species.append(_parse_pymatgen_species(sp_str))
        else:
            species.append(Composition({
                _parse_pymatgen_species(sp): occ for sp, occ in occ.items()
            }))
        coords.append(site.frac_coords.tolist())
        for key, value in site.properties.as_dict().items():
            if key not in props:
                props[key] = [None] * len(structure.sites)
            props[key][i] = value

    lattice = PmgLattice(structure.lattice.matrix)
    return Structure(lattice, species, coords, site_properties=props)
