"""Renderer-independent choices for the finite crystal region to display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from crystalfig.analysis.coordination import CoordinationAnalyzer
from crystalfig.analysis.structure_intent import StructureIntentAnalyzer
from crystalfig.geometry.periodic import nearest_image
from crystalfig.model.site import element_symbol
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborStrategy

BondDisplayPolicy = Literal[
    "all_bonds", "center_ligand_only", "polyhedron_edges_only", "hidden"
]


@dataclass(frozen=True)
class MotifConfig:
    max_atoms: int = 12
    strategy: str = "species_balanced"
    seed_site: int | None = None
    required_species: tuple[str, ...] = ()
    strict_species_coverage: bool = False


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
    motif: MotifConfig = MotifConfig()


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
    motif_species: tuple[str, ...] = ()
    motif_missing_species: tuple[str, ...] = ()
    motif_selected_sites: tuple[int, ...] = ()
    motif_strategy: str | None = None


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
        adjacency = DisplayPlanner._motif_adjacency(structure, bonds)
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
    def _motif_adjacency(structure, bonds):
        adjacency = {index: [] for index in range(structure.num_sites)}
        site_images = np.floor(structure.frac_coords).astype(int)
        for bond in bonds:
            image = tuple(int(bond.jimage[a] + site_images[bond.j, a] - site_images[bond.i, a]) for a in range(3))
            adjacency[bond.i].append((bond.j, image))
            adjacency[bond.j].append((bond.i, tuple(-value for value in image)))
        return {index: list(dict.fromkeys(entries)) for index, entries in adjacency.items()}

    @staticmethod
    def plan_motif(structure, bonds, config: MotifConfig | None = None):
        """Choose periodic site images while preserving the real bond graph."""
        config = config or MotifConfig()
        if config.max_atoms < 1:
            raise ValueError("motif max_atoms must be positive")
        if config.strategy not in {"species_balanced", "largest_component", "centered"}:
            raise ValueError("motif strategy must be species_balanced, largest_component, or centered")
        if config.seed_site is not None and not 0 <= config.seed_site < structure.num_sites:
            raise ValueError(f"motif seed site index {config.seed_site} is outside the structure")
        required_species = config.required_species or tuple(sorted({
            element_symbol(species)
            for site in structure.sites for species, occupancy in site.occupancy.items()
            if occupancy > 0
        }))
        if not structure.num_sites:
            if config.strict_species_coverage and config.required_species:
                raise ValueError(
                    "Motif could not include required species: "
                    + ", ".join(config.required_species)
                    + "; increase motif.max_atoms or use framing='unit_cell'"
                )
            return (), (), required_species
        if config.strategy == "largest_component":
            instances = DisplayPlanner.representative_motif_instances(
                structure, bonds, limit=config.max_atoms
            )
        else:
            adjacency = DisplayPlanner._motif_adjacency(structure, bonds)
            elements = [site.dominant_element for site in structure.sites]
            folded = np.asarray([site.frac_coords % 1.0 for site in structure.sites])
            available_species = set(elements)
            if config.seed_site is not None:
                start = config.seed_site
            else:
                start_candidates = range(structure.num_sites)
                if config.strategy == "species_balanced":
                    start_candidates = [
                        i for i, element in enumerate(elements)
                        if element in required_species
                    ] or start_candidates
                start = min(start_candidates, key=lambda i: (
                    np.linalg.norm(folded[i] - 0.5), i
                ))
            selected = [(start, (0, 0, 0))]
            visited = set(selected)
            seed_shell = set(adjacency[start])
            if all(elements[index] == elements[start] for index, _ in seed_shell):
                seed_shell.clear()
            while len(selected) < config.max_atoms:
                counts = {}
                for index, _ in selected:
                    species = elements[index]
                    counts[species] = counts.get(species, 0) + 1
                candidates = []
                for index, offset in selected:
                    for other, image in adjacency[index]:
                        node = (other, tuple(offset[a] + image[a] for a in range(3)))
                        if node not in visited:
                            candidates.append(node)
                missing_renderable = [
                    species for species in required_species
                    if species in available_species and counts.get(species, 0) == 0
                ]
                if config.strategy == "species_balanced" and missing_renderable:
                    missing_frontier = [
                        node for node in candidates
                        if elements[node[0]] in missing_renderable
                    ]
                    candidates = missing_frontier
                    if not candidates:
                        for species in missing_renderable:
                            nearest = None
                            for index in range(structure.num_sites):
                                if elements[index] != species:
                                    continue
                                for selected_index, selected_offset in selected:
                                    image_cart, image = nearest_image(
                                        folded[selected_index] + np.asarray(selected_offset),
                                        folded[index], structure.lattice,
                                    )
                                    offset = tuple(image)
                                    node = (index, offset)
                                    if node in visited:
                                        continue
                                    reference_cart = structure.lattice.frac_to_cart(
                                        folded[selected_index] + np.asarray(selected_offset)
                                    )
                                    distance = np.linalg.norm(image_cart - reference_cart)
                                    candidate = (distance, index, offset)
                                    if nearest is None or candidate < nearest:
                                        nearest = candidate
                            if nearest is not None:
                                candidates.append((nearest[1], nearest[2]))
                elif config.strategy == "species_balanced":
                    remaining_shell = seed_shell - visited
                    if 0 < len(remaining_shell) <= config.max_atoms - len(selected):
                        candidates = remaining_shell
                if not candidates:
                    break
                def rank(node, counts=counts):
                    index, offset = node
                    species = elements[index]
                    if config.strategy == "centered":
                        delta = folded[index] + np.asarray(offset) - folded[start]
                        distance = np.linalg.norm(structure.lattice.frac_to_cart(delta))
                        return (distance, -len(adjacency[index]), index, offset)
                    return (counts.get(species, 0),
                            0 if offset == (0, 0, 0) else 1,
                            -len(adjacency[index]), index, offset)
                node = min(set(candidates), key=rank)
                visited.add(node)
                selected.append(node)
            instances = tuple(selected)

        species = tuple(sorted({structure.sites[i].dominant_element for i, _ in instances}))
        missing = tuple(item for item in required_species if item not in species)
        if missing and config.strict_species_coverage:
            raise ValueError(
                "Motif could not include required species: "
                + ", ".join(missing)
                + "; increase motif.max_atoms or use framing='unit_cell'"
            )
        return instances, species, missing

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
