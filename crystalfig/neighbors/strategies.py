"""Concrete neighbor detection strategies."""

from __future__ import annotations

import numpy as np

from crystalfig.exceptions import OptionalDependencyError
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborBond
from crystalfig.styles.radii import get_radius


class CutoffStrategy:
    """Bond detection by global or element-pair distance cutoff."""

    def __init__(self, cutoff: float = 2.5, pair_cutoffs: dict[str, float] | None = None):
        self.cutoff = cutoff
        self.pair_cutoffs = pair_cutoffs or {}

    def _cutoff_for(self, sp1: str, sp2: str) -> float:
        key1 = f"{sp1}-{sp2}"
        key2 = f"{sp2}-{sp1}"
        if key1 in self.pair_cutoffs:
            return self.pair_cutoffs[key1]
        if key2 in self.pair_cutoffs:
            return self.pair_cutoffs[key2]
        return self.cutoff

    def get_bonds(self, structure: CrystalStructure) -> list[NeighborBond]:
        bonds = []
        lattice = structure.lattice
        n = len(structure)
        for i in range(n):
            for j in range(i + 1, n):
                cutoff = self._cutoff_for(
                    structure.sites[i].dominant_species,
                    structure.sites[j].dominant_species,
                )
                # Try nearest image
                frac_i = structure.sites[i].frac_coords
                frac_j = structure.sites[j].frac_coords
                delta = frac_j - frac_i
                delta -= np.round(delta)
                image_frac = frac_i + delta
                dist = np.linalg.norm(lattice.frac_to_cart(delta))
                if dist <= cutoff and dist > 0.1:
                    jimage = tuple(int(round(frac_j[k] - image_frac[k])) for k in range(3))
                    bonds.append(NeighborBond(i=i, j=j, jimage=jimage, distance=dist))
        return bonds


class CovalentRadiiStrategy:
    """Bond detection using covalent radii with a tolerance factor."""

    def __init__(self, tolerance: float = 0.4):
        self.tolerance = tolerance

    def get_bonds(self, structure: CrystalStructure) -> list[NeighborBond]:
        bonds = []
        lattice = structure.lattice
        n = len(structure)
        for i in range(n):
            for j in range(i + 1, n):
                sp1 = structure.sites[i].dominant_species
                sp2 = structure.sites[j].dominant_species
                cutoff = get_radius(sp1, "covalent", 0.2) + get_radius(sp2, "covalent", 0.2) + self.tolerance
                frac_i = structure.sites[i].frac_coords
                frac_j = structure.sites[j].frac_coords
                delta = frac_j - frac_i
                delta -= np.round(delta)
                dist = np.linalg.norm(lattice.frac_to_cart(delta))
                if dist <= cutoff and dist > 0.1:
                    jimage = tuple(int(round(frac_j[k] - (frac_i[k] + delta[k]))) for k in range(3))
                    bonds.append(NeighborBond(i=i, j=j, jimage=jimage, distance=dist))
        return bonds


class CrystalNNStrategy:
    """Use pymatgen's CrystalNN for bond detection."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_bonds(self, structure: CrystalStructure) -> list[NeighborBond]:
        from pymatgen.analysis.local_env import CrystalNN

        from crystalfig.io.pymatgen_adapter import to_pymatgen

        pmg = to_pymatgen(structure)
        cnn = CrystalNN(**self.kwargs)
        bonds = []
        for i, site in enumerate(pmg):
            nn_info = cnn.get_nn_info(pmg, i)
            for info in nn_info:
                j = info["site_index"]
                if j < i:
                    continue
                image = info.get("image", (0, 0, 0))
                dist = site.distance(info["site"])
                bonds.append(NeighborBond(i=i, j=j, jimage=tuple(image), distance=dist))
        return bonds


class ASEStrategy:
    """Use ASE natural cutoffs for neighbor detection."""

    def __init__(self):
        try:
            import ase  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError("ase", "ase") from exc

    def get_bonds(self, structure: CrystalStructure) -> list[NeighborBond]:
        from ase.neighborlist import NeighborList, natural_cutoffs

        from crystalfig.io.ase_adapter import to_ase

        atoms = to_ase(structure)
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
        nl.update(atoms)
        bonds = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j, offset in zip(indices, offsets, strict=True):
                if j < i:
                    continue
                dist = atoms.get_distance(i, j, mic=True)
                bonds.append(NeighborBond(i=i, j=j, jimage=tuple(int(o) for o in offset), distance=dist))
        return bonds
