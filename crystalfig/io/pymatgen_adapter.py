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
            species = str(site.species.elements[0])
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


def to_pymatgen(structure: CrystalStructure):
    """Convert a CrystalStructure to pymatgen Structure."""
    from pymatgen.core import Composition, Element, Species, Structure
    from pymatgen.core import Lattice as PmgLattice

    species = []
    coords = []
    props: dict[str, list] = {}
    for i, site in enumerate(structure.sites):
        occ = site.occupancy
        if len(occ) == 1:
            sp_str = next(iter(occ))
            try:
                # Use plain Element if no oxidation state is specified; otherwise Species.
                if any(c in sp_str for c in "+-1234567890"):
                    sp = Species(sp_str)
                else:
                    sp = Element(sp_str)
            except Exception:
                # Fallback: element symbol if oxidation state parsing fails.
                from crystalfig.model.site import element_symbol
                sp = Element(element_symbol(sp_str))
            species.append(sp)
        else:
            species.append(Composition({
                (Species(sp) if any(c in sp for c in "+-1234567890") else Element(sp)): occ
                for sp, occ in occ.items()
            }))
        coords.append(site.frac_coords.tolist())
        for key, value in site.properties.as_dict().items():
            if key not in props:
                props[key] = [None] * len(structure.sites)
            props[key][i] = value

    lattice = PmgLattice(structure.lattice.matrix)
    return Structure(lattice, species, coords, site_properties=props)
