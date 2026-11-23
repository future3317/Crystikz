"""Periodic coordination environments for crystal structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Element

from crystalfig.geometry.polyhedra import coordination_shell_has_volume
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborBond, NeighborStrategy


@dataclass(frozen=True)
class CoordinationShell:
    """One center and its periodic ligand images in Cartesian coordinates."""

    center_index: int
    center_position: np.ndarray
    ligand_indices: tuple[int, ...]
    ligand_positions: np.ndarray
    ligand_image_offsets: tuple[tuple[int, int, int], ...]
    coordination_number: int
    environment: str


class CoordinationAnalyzer:
    """Analyze neighbor strategies without creating visualization objects."""

    @staticmethod
    def adjacency(structure: CrystalStructure, bonds: list[NeighborBond]):
        """Map site indices to periodic neighbor instances from a bond graph."""
        adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = {
            index: [] for index in range(structure.num_sites)
        }
        for bond in bonds:
            image = tuple(int(value) for value in bond.jimage)
            adjacency[bond.i].append((bond.j, image))
            if bond.i != bond.j:
                adjacency[bond.j].append((bond.i, tuple(-value for value in image)))
        return adjacency

    def automatic_center_candidates(
        self, structure: CrystalStructure, bonds: list[NeighborBond]
    ) -> list[int]:
        """Find chemically plausible centers with a 3D periodic ligand shell."""
        adjacency = self.adjacency(structure, bonds)
        shells = {
            shell.center_index: shell
            for shell in self.analyze(structure, list(adjacency), bonds)
        }
        candidates = []
        for center_index, neighbors in adjacency.items():
            if not 4 <= len(neighbors) <= 8:
                continue
            shell = shells[center_index]
            center_x = Element(structure.sites[center_index].dominant_element).X
            ligand_x = [Element(structure.sites[index].dominant_element).X for index in shell.ligand_indices]
            if (
                center_x is not None
                and all(value is not None for value in ligand_x)
                and float(np.mean(ligand_x)) - center_x >= 0.4
                and coordination_shell_has_volume(shell.ligand_positions - shell.center_position)
            ):
                candidates.append(center_index)
        return candidates

    def select_adjacent_centers(
        self,
        structure: CrystalStructure,
        centers: list[int],
        limit: int,
        bonds: list[NeighborBond],
        center_region: tuple[float, float] | None = None,
    ) -> list[int]:
        """Cover center environments, then grow a connected periodic motif."""
        if not centers or limit <= 0:
            return []
        adjacency = self.adjacency(structure, bonds)
        shells = {shell.center_index: shell for shell in self.analyze(structure, centers, bonds)}
        positions = {index: shells[index].center_position for index in centers}
        center_instances = {
            index: tuple(np.round(shells[index].center_position, 8)) for index in centers
        }
        lower, upper = center_region or (0.0, 1.0)
        target = structure.lattice.frac_to_cart(np.full(3, (lower + upper) / 2))
        environments = {}
        ligand_instances = {}
        for index in centers:
            shell = shells[index]
            neighbors = adjacency.get(index, [])
            ligand_species = tuple(sorted({structure.sites[j].dominant_element for j, _ in neighbors}))
            environments[index] = (structure.sites[index].dominant_element, shell.coordination_number, ligand_species)
            ligand_instances[index] = {
                tuple(np.round(position, 8)) for position in shell.ligand_positions
            }
        limit = min(limit, len(centers))
        selected = [min(centers, key=lambda index: (np.linalg.norm(positions[index] - target), index))]
        while len(selected) < limit:
            remaining = [index for index in centers if index not in selected]
            represented = {environments[index] for index in selected}
            connected = [
                index for index in remaining
                if any(
                    ligand_instances[index] & ligand_instances[chosen]
                    or center_instances[index] in ligand_instances[chosen]
                    or center_instances[chosen] in ligand_instances[index]
                    for chosen in selected
                )
            ]
            if not connected:
                break
            next_center = min(
                connected,
                key=lambda index: (
                    -int(environments[index] not in represented),
                    -sum(len(ligand_instances[index] & ligand_instances[chosen]) for chosen in selected),
                    min(np.linalg.norm(positions[index] - positions[chosen]) for chosen in selected),
                    index,
                ),
            )
            selected.append(next_center)
        return selected

    def analyze(
        self,
        structure: CrystalStructure,
        centers: list[int] | tuple[int, ...],
        strategy: NeighborStrategy | list[NeighborBond],
    ) -> list[CoordinationShell]:
        """Return periodic shells for the selected site indices.

        A strategy is evaluated once for the whole structure. A precomputed
        bond list is also accepted so callers that already own a graph reuse it.
        """
        bonds = strategy if isinstance(strategy, list) else strategy.get_bonds(structure)
        adjacency = self.adjacency(structure, bonds)

        shells = []
        for center_index in centers:
            center_frac, center_image = self._fold(structure.sites[center_index].frac_coords)
            center_position = structure.lattice.frac_to_cart(center_frac)
            ligand_indices = []
            ligand_positions = []
            image_offsets = []
            for ligand_index, bond_image in adjacency.get(center_index, []):
                _, ligand_image = self._fold(structure.sites[ligand_index].frac_coords)
                display_image = tuple(
                    bond_image[axis] + ligand_image[axis] - center_image[axis]
                    for axis in range(3)
                )
                ligand_frac, _ = self._fold(structure.sites[ligand_index].frac_coords)
                ligand_indices.append(ligand_index)
                image_offsets.append(display_image)
                ligand_positions.append(
                    structure.lattice.frac_to_cart(ligand_frac + np.asarray(display_image))
                )
            symbols = sorted(
                structure.sites[index].dominant_element for index in ligand_indices
            )
            counts = {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}
            ligand_formula = "".join(
                symbol + (str(count) if count > 1 else "")
                for symbol, count in counts.items()
            )
            shells.append(CoordinationShell(
                center_index=center_index,
                center_position=center_position,
                ligand_indices=tuple(ligand_indices),
                ligand_positions=np.asarray(ligand_positions, dtype=float).reshape((-1, 3)),
                ligand_image_offsets=tuple(image_offsets),
                coordination_number=len(ligand_indices),
                environment=f"{structure.sites[center_index].dominant_element}{ligand_formula}",
            ))
        return shells

    @staticmethod
    def _fold(frac: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
        image = np.floor(frac).astype(int)
        return frac - image, tuple(int(value) for value in image)
