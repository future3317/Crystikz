"""Concrete neighbor detection strategies."""

from __future__ import annotations

from crystalfig.exceptions import OptionalDependencyError
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborBond
from crystalfig.styles.radii import get_radius


def _pmg_neighbor_list(structure: CrystalStructure, max_cutoff: float):
    """Return pymatgen neighbors with (site, distance, index, image) tuples."""
    from crystalfig.io.pymatgen_adapter import to_pymatgen

    pmg = to_pymatgen(structure)
    return pmg.get_all_neighbors(max_cutoff, include_index=True, include_image=True)


def _canonical_bond_key(i: int, j: int, image: tuple[int, int, int]) -> tuple:
    """Canonical key treating (i,j,image) and (j,i,-image) as the same bond.

    For self-bonds (i == j) each periodic image is kept distinct, so no
    reversal is performed.
    """
    forward = (i, j, image)
    if i == j:
        return forward
    reverse = (j, i, tuple(-x for x in image))
    return min(forward, reverse)


class CutoffStrategy:
    """Bond detection by global or element-pair distance cutoff.

    Uses pymatgen's periodic neighbor search so bonds across periodic
    boundaries are returned with the correct image offset.
    """

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
        seen: set[tuple] = set()
        max_cutoff = max(self.cutoff, *(self.pair_cutoffs.values() or [0.0]))
        all_neighbors = _pmg_neighbor_list(structure, max_cutoff)
        for i, nbrs in enumerate(all_neighbors):
            for _site, dist, j, image in nbrs:
                if dist < 0.1:
                    continue
                jimage = tuple(int(round(x)) for x in image)
                key = _canonical_bond_key(i, j, jimage)
                if key in seen:
                    continue
                seen.add(key)
                cutoff = self._cutoff_for(
                    structure.sites[i].dominant_element,
                    structure.sites[j].dominant_element,
                )
                if dist > cutoff:
                    continue
                bonds.append(NeighborBond(
                    i=i,
                    j=j,
                    jimage=jimage,
                    distance=float(dist),
                ))
        return bonds


class CovalentRadiiStrategy:
    """Bond detection using covalent radii with a tolerance factor."""

    def __init__(self, tolerance: float = 0.4, max_cutoff: float = 6.0):
        self.tolerance = tolerance
        self.max_cutoff = max_cutoff

    def get_bonds(self, structure: CrystalStructure) -> list[NeighborBond]:
        bonds = []
        seen: set[tuple] = set()
        all_neighbors = _pmg_neighbor_list(structure, self.max_cutoff)
        for i, nbrs in enumerate(all_neighbors):
            for _site, dist, j, image in nbrs:
                if dist < 0.1:
                    continue
                jimage = tuple(int(round(x)) for x in image)
                key = _canonical_bond_key(i, j, jimage)
                if key in seen:
                    continue
                seen.add(key)
                sp1 = structure.sites[i].dominant_element
                sp2 = structure.sites[j].dominant_element
                cutoff = get_radius(sp1, "covalent", 0.2) + get_radius(sp2, "covalent", 0.2) + self.tolerance
                if dist > cutoff:
                    continue
                bonds.append(NeighborBond(
                    i=i,
                    j=j,
                    jimage=jimage,
                    distance=float(dist),
                ))
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
