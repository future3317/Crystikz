"""Renderer-independent choices for the finite crystal region to display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from crystalfig.analysis.coordination import CoordinationAnalyzer
from crystalfig.analysis.structure_intent import StructureIntentAnalyzer
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborStrategy

BondDisplayPolicy = Literal[
    "all_bonds", "center_ligand_only", "polyhedron_edges_only", "hidden"
]


@dataclass(frozen=True)
class DisplayConfig:
    intent: str = "auto"
    region: str | tuple[int, int, int] = "auto"
    centers: tuple[str, ...] = ()
    neighbor_strategy: str | NeighborStrategy = "crystalnn"
    coordination_shell: str = "complete"
    show_atoms: bool = True
    show_bonds: bool | None = None
    show_unit_cell: bool | None = None
    max_atoms: int = 60
    max_polyhedra: int = 6
    framing: str = "auto"
    framing_site: int | None = None
    bond_display_policy: BondDisplayPolicy | None = None


@dataclass(frozen=True)
class DisplayPlan:
    intent: str
    confidence: float
    repeat: tuple[int, int, int]
    coordination_shell: str
    image_atoms: str
    cell_frame: str
    center_window: tuple[float, float] | None
    selected_centers: tuple[int, ...] = ()
    show_atoms: bool = True
    show_bonds: bool = True
    show_cell: bool = True
    layer_normal: tuple[float, float, float] | None = None
    layer_translations: tuple[tuple[float, float, float], ...] = ()
    framing: str = "auto"


class DisplayPlanner:
    """Resolve structure intent and a budgeted periodic display window."""

    @staticmethod
    def resolve_intent(structure: CrystalStructure, intent: str, bonds) -> tuple[str, float]:
        return StructureIntentAnalyzer(structure).classify(intent, bonds)

    @staticmethod
    def plan(
        structure: CrystalStructure,
        *,
        intent: str,
        confidence: float,
        region: str | tuple[int, int, int],
        coordination_shell: str,
        display_style: str,
        show_atoms: bool,
        show_bonds: bool | None,
        show_unit_cell: bool | None,
        display_budget: dict[str, int | float],
        estimated_degree: float,
        bonds,
        centers,
        supercell=None,
    ) -> tuple[DisplayPlan, dict[str, int | float]]:
        if coordination_shell not in {"complete", "show_partial", "none"}:
            raise ValueError("coordination_shell must be 'complete', 'show_partial', or 'none'")
        budget = dict(display_budget)
        if any(float(value) <= 0 for value in budget.values()):
            raise ValueError("display_budget values must be positive")
        atom_cap = min(int(budget["max_atoms"]), int(budget["max_bonds"] / max(estimated_degree, 1.0)))
        repeats = DisplayPlanner.publication_window(
            structure, intent, region, max_atoms=max(atom_cap, structure.num_sites),
            supercell=supercell, bonds=bonds,
        )
        polyhedral = display_style in {"nature", "polyhedra"}
        render_bonds = (
            not polyhedral or intent in {"coordination", "hybrid"}
        ) if show_bonds is None else show_bonds
        show_cell = (
            show_unit_cell if show_unit_cell is not None
            else not (polyhedral and intent in {"coordination", "hybrid"})
        )
        analyzer = StructureIntentAnalyzer(structure)
        translations = analyzer.periodic_translation_vectors(bonds) if intent == "layered" else np.empty((0, 3))
        normal = None
        if len(translations) == 2:
            vector = np.cross(translations[0], translations[1])
            if np.linalg.norm(vector) > 1e-12:
                normal = tuple(float(value) for value in vector / np.linalg.norm(vector))
        plan = DisplayPlan(
            intent=intent,
            confidence=confidence,
            repeat=repeats,
            coordination_shell=coordination_shell,
            image_atoms="connected_motif" if polyhedral else "finite_supercell",
            cell_frame="subtle" if show_cell else "hide",
            center_window=None,
            show_atoms=show_atoms,
            show_bonds=bool(render_bonds),
            show_cell=bool(show_cell),
            layer_normal=normal,
            layer_translations=tuple(tuple(float(value) for value in row) for row in translations),
        )
        return plan, budget

    @staticmethod
    def select_centers(
        source_structure: CrystalStructure,
        display_structure: CrystalStructure,
        bonds,
        centers,
        limit: int,
        center_region: tuple[float, float] | None,
    ) -> tuple[int, ...]:
        analyzer = CoordinationAnalyzer()
        if centers == "auto":
            candidates = analyzer.automatic_center_candidates(display_structure, bonds)
        else:
            requests = centers if isinstance(centers, (list, tuple)) else [centers]
            candidates = []
            for center in requests:
                if isinstance(center, (int, np.integer)):
                    base_index = int(center)
                    if not 0 <= base_index < source_structure.num_sites:
                        raise ValueError(f"Center site index {base_index} is outside the structure")
                    base_site = source_structure.sites[base_index]
                    for index, site in enumerate(display_structure.sites):
                        if site.dominant_species != base_site.dominant_species:
                            continue
                        cart = display_structure.lattice.frac_to_cart(site.frac_coords)
                        base_frac = source_structure.lattice.cart_to_frac(cart) % 1.0
                        delta = (base_frac - base_site.frac_coords + 0.5) % 1.0 - 0.5
                        if np.linalg.norm(delta) < 1e-5:
                            candidates.append(index)
                else:
                    candidates.extend(
                        index for index, site in enumerate(display_structure.sites)
                        if site.dominant_element == center or site.dominant_species == center
                    )
        candidates = list(dict.fromkeys(candidates))
        if centers != "auto":
            return tuple(candidates)
        return tuple(analyzer.select_adjacent_centers(
            display_structure, candidates, limit, bonds, center_region
        ))

    @staticmethod
    def publication_window(
        structure: CrystalStructure,
        intent: str,
        region: str | tuple[int, int, int],
        max_atoms: int,
        supercell=None,
        bonds=None,
    ) -> tuple[int, int, int]:
        if isinstance(region, tuple):
            if len(region) != 3 or any(not isinstance(size, int) or size < 1 for size in region):
                raise ValueError("region repeat tuple must contain three positive integers")
            return region
        if region not in {"auto", "cell", "motif", "supercell"}:
            raise ValueError("region must be 'auto', 'cell', 'motif', 'supercell', or a repeat tuple")
        if region == "cell":
            return (1, 1, 1)
        if supercell is not None:
            matrix = np.asarray(supercell)
            if matrix.shape == (3, 3):
                return tuple(int(value) for value in np.diag(matrix))
            if np.isscalar(supercell):
                return (int(supercell),) * 3
            return tuple(int(value) for value in supercell)
        repeats = np.ones(3, dtype=int)
        lengths = np.asarray(structure.lattice.lengths, dtype=float)
        while structure.num_sites * int(np.prod(repeats)) < max_atoms:
            candidates = [
                axis for axis in range(3)
                if structure.num_sites * int(np.prod(repeats))
                * (repeats[axis] + 1) / repeats[axis] <= max_atoms
            ]
            if not candidates:
                break
            axis = min(candidates, key=lambda item: (lengths[item] * repeats[item], item))
            repeats[axis] += 1
        if intent == "layered" and supercell is None:
            translations = StructureIntentAnalyzer(structure).periodic_translation_vectors(
                [] if bonds is None else bonds
            )
            if len(translations) == 2:
                normal = np.cross(translations[0], translations[1])
                normal /= max(np.linalg.norm(normal), 1e-12)
                layer_axis = int(np.argmax(np.abs(structure.lattice.matrix @ normal)))
                repeats[layer_axis] = min(repeats[layer_axis], 1)
        if intent == "minimal" or structure.num_sites >= max_atoms:
            repeats[:] = 1
        return tuple(int(value) for value in repeats)

    @staticmethod
    def resolve_framing(framing, framing_site, intent, region, centers):
        """Apply an explicit figure framing choice to the existing display inputs."""
        if framing not in {"auto", "unit_cell", "local_coordination", "motif"}:
            raise ValueError("framing must be 'auto', 'unit_cell', 'local_coordination', or 'motif'")
        if framing_site is not None and (not isinstance(framing_site, int) or framing_site < 0):
            raise ValueError("framing_site must be a non-negative site index")
        if framing == "unit_cell":
            return intent, "cell", centers
        if framing == "motif":
            return intent, "motif", centers
        if framing == "local_coordination":
            if framing_site is not None:
                centers = [framing_site]
            return "coordination", "cell", centers
        return intent, region, centers

    @staticmethod
    def representative_motif_instances(structure, bonds, limit=12):
        """Grow a compact connected motif in the periodic site-image graph."""
        adjacency = {index: [] for index in range(structure.num_sites)}
        site_images = np.floor(structure.frac_coords).astype(int)
        for bond in bonds:
            image = tuple(
                int(bond.jimage[axis] + site_images[bond.j, axis] - site_images[bond.i, axis])
                for axis in range(3)
            )
            adjacency[bond.i].append((bond.j, image))
            adjacency[bond.j].append((bond.i, tuple(-value for value in image)))
        adjacency = {
            index: list(dict.fromkeys(entries)) for index, entries in adjacency.items()
        }
        if not structure.num_sites:
            return ()
        quotient = {
            index: {other for other, _ in entries if other != index}
            for index, entries in adjacency.items()
        }
        components = []
        remaining = set(range(structure.num_sites))
        while remaining:
            component = set()
            stack = [min(remaining)]
            while stack:
                index = stack.pop()
                if index in component:
                    continue
                component.add(index)
                stack.extend(quotient[index] - component)
            remaining -= component
            components.append(component)
        largest = max(components, key=lambda nodes: (len(nodes), -min(nodes)))
        start = min(largest, key=lambda i: (-len(adjacency[i]), i))
        selected = [(start, (0, 0, 0))]
        visited = set(selected)
        represented = {structure.sites[start].dominant_species}
        cursor = 0
        while cursor < len(selected) and len(selected) < limit:
            index, offset = selected[cursor]
            cursor += 1
            neighbors = [
                (other, tuple(offset[axis] + image[axis] for axis in range(3)))
                for other, image in adjacency[index]
            ]
            neighbors.sort(key=lambda node: (
                structure.sites[node[0]].dominant_species in represented,
                -len(adjacency[node[0]]), node[0], node[1],
            ))
            for node in neighbors:
                if node in visited:
                    continue
                visited.add(node)
                selected.append(node)
                represented.add(structure.sites[node[0]].dominant_species)
                if len(selected) == limit:
                    break
        return tuple(selected)

    @staticmethod
    def local_center(structure, bonds):
        analyzer = CoordinationAnalyzer()
        candidates = analyzer.automatic_center_candidates(structure, bonds)
        if not candidates:
            adjacency = analyzer.adjacency(structure, bonds)
            candidates = [index for index, neighbors in adjacency.items() if neighbors]
        if not candidates:
            return ()
        return tuple(analyzer.select_adjacent_centers(structure, candidates, 1, bonds))
