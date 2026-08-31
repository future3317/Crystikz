"""Adapter between ASE Atoms and CrystalStructure. optional dependency."""

from __future__ import annotations

import numpy as np

from crystalfig.exceptions import OptionalDependencyError
from crystalfig.model.lattice import Lattice
from crystalfig.model.properties import SiteProperties
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure


def _require_ase():
    try:
        import ase  # noqa: F401
    except ImportError as exc:
        raise OptionalDependencyError("ase", "ase") from exc
    return True


def from_ase(atoms) -> CrystalStructure:
    """Convert ASE Atoms to CrystalStructure."""
    _require_ase()
    lattice = Lattice(np.array(atoms.cell))
    sites = []
    for i, atom in enumerate(atoms):
        props = SiteProperties()
        if atom.tag is not None:
            props.set("tag", int(atom.tag))
        if hasattr(atom, "momentum") and atom.momentum is not None:
            props.set("momentum", np.asarray(atom.momentum))
        if atom.magmom is not None and not np.isclose(float(atom.magmom), 0.0):
            props.magnetic_moment = float(atom.magmom)

        sites.append(Site(
            frac_coords=np.array(atom.scaled_position),
            species=atom.symbol,
            properties=props,
            source_index=i,
        ))
    return CrystalStructure(lattice=lattice, sites=sites)


def to_ase(structure: CrystalStructure):
    """Convert CrystalStructure to ASE Atoms."""
    _require_ase()
    from ase import Atoms

    symbols = [site.dominant_species for site in structure.sites]
    positions = [site.cart_coords(structure.lattice).tolist() for site in structure.sites]
    cell = structure.lattice.matrix.tolist()
    magmoms = [site.properties.magnetic_moment for site in structure.sites]
    if all(m is None for m in magmoms):
        magmoms = None
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True, magmoms=magmoms)
