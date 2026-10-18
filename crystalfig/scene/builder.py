"""Build a Scene from a CrystalStructure and visualization options."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np
from pymatgen.core import Element
from scipy.spatial import ConvexHull, QhullError

from crystalfig.geometry.planes import MillerPlane
from crystalfig.geometry.polyhedra import (
    build_polyhedron,
    coordination_shell_has_volume,
    merge_coplanar_faces,
)
from crystalfig.model.site import element_symbol
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborBond
from crystalfig.scene.camera import Camera
from crystalfig.scene.primitives import (
    Arrow,
    Axis,
    Bond,
    CellEdge,
    LegendItem,
    Polygon,
    Polyhedron,
    Sphere,
    Text,
)
from crystalfig.scene.scene import Scene
from crystalfig.styles.palette import ColorPalette
from crystalfig.styles.radii import get_radius
from crystalfig.styles.theme import FigureTheme

_DISORDERED_WARNED: set[tuple[int, tuple[int, int, int]]] = set()


def _clip_segment_box(start: np.ndarray, end: np.ndarray, lower: float, upper: float):
    """Clip a fractional-coordinate segment to an axis-aligned display box."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < lower or start[axis] > upper:
                return None
            continue
        t0 = (lower - start[axis]) / direction[axis]
        t1 = (upper - start[axis]) / direction[axis]
        t_min = max(t_min, min(t0, t1))
        t_max = min(t_max, max(t0, t1))
        if t_min > t_max:
            return None
    return start + t_min * direction, start + t_max * direction


@dataclass
class AtomStyleOverride:
    """Per-site or per-species atom style override."""

    species: str | None = None
    indices: set[int] | None = None
    color: str | None = None
    scale: float | None = None
    opacity: float | None = None
    radius: float | None = None
    render_style: str | None = None
    visible: bool = True

    def matches(self, site_index: int, site) -> bool:
        if self.indices is not None and site_index in self.indices:
            return True
        return self.species is not None and (
            self.species == site.dominant_species or self.species == site.dominant_element
        )


@dataclass
class BondStyleOverride:
    """Per-pair or per-index bond style override."""

    pair: tuple | None = None
    indices: set[int] | None = None
    width: float | None = None
    color: str | None = None
    opacity: float | None = None
    visible: bool = True

    def matches(self, bond_index: int, bond: NeighborBond, site_i, site_j) -> bool:
        if self.indices is not None and bond_index in self.indices:
            return True
        if self.pair is None:
            return False
        p = self.pair
        if len(p) == 2 and all(isinstance(x, int) for x in p) and (
            (bond.i, bond.j) == p or (bond.j, bond.i) == p
        ):
            return True
        if len(p) == 2 and all(isinstance(x, str) for x in p):
            sp_i = site_i.dominant_species if site_i else ""
            sp_j = site_j.dominant_species if site_j else ""
            el_i = site_i.dominant_element if site_i else ""
            el_j = site_j.dominant_element if site_j else ""
            if ({sp_i, sp_j} == set(p)) or ({el_i, el_j} == set(p)):
                return True
        return False


@dataclass
class PolyhedraSpec:
    """Specification for one set of coordination polyhedra."""

    centers: list[int | str] | str
    strategy: Callable | None = None
    fill_color: str | None = None
    opacity: float | None = None
    edge_width: float | None = None
    edge_color: str | None = None
    show_bonds: bool | None = None
    color_by_center: bool = False


@dataclass
class Annotation:
    """Screen-space or data-space annotation."""

    kind: str  # "site_label", "formula_label", "panel_label"
    text: str
    site_index: int | None = None
    offset: tuple[float, float] = (0.0, 0.0)
    position: str | tuple[float, float] = "top_left"
    fontsize: float | None = None
    color: str | None = None
    fontweight: str = "normal"


@dataclass
class SceneOptions:
    """Options controlling scene construction."""

    show_unit_cell: bool = True
    show_axes: bool = False
    show_atoms: bool = True
    show_bonds: bool = False
    show_polyhedra: bool = False
    show_legend: bool = False
    atom_style: str = "shaded"
    supercell: int | tuple[int, int, int] | np.ndarray | None = None
    display_boundary: str = "cell_complete"  # cell_complete, connected, polyhedra_complete
    clip_polyhedra_to_cell: bool = False
    bonds: list[NeighborBond] | None = None
    bond_strategy: Callable | None = None
    polyhedra_specs: list[PolyhedraSpec] = None  # type: ignore[assignment]
    vectors: list[tuple[int, np.ndarray, str]] | None = None
    miller_planes: list[MillerPlane] | None = None
    selected_sites: list[int] | None = None
    center_region: tuple[float, float] | None = None
    center_limit: int | None = None
    defect_sites: list[int] | None = None
    atom_overrides: list[AtomStyleOverride] = None  # type: ignore[assignment]
    bond_overrides: list[BondStyleOverride] = None  # type: ignore[assignment]
    annotations: list[Annotation] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.polyhedra_specs is None:
            self.polyhedra_specs = []
        if self.atom_overrides is None:
            self.atom_overrides = []
        if self.bond_overrides is None:
            self.bond_overrides = []
        if self.annotations is None:
            self.annotations = []


class SceneBuilder:
    """Construct a Scene from a CrystalStructure."""

    def __init__(
        self,
        structure: CrystalStructure,
        theme: FigureTheme,
        palette: ColorPalette,
        options: SceneOptions,
        camera: Camera | None = None,
    ):
        self.structure = structure
        self.theme = theme
        self.palette = palette
        self.options = options
        self.camera = camera

    def build(self) -> Scene:
        """Build and return the complete Scene."""
        if self.options.supercell is not None:
            structure = self.structure.make_supercell(self.options.supercell)
        else:
            structure = self.structure

        # Bonds must be computed on the final (possibly supercell) structure.
        # They are needed both for explicit bond rendering and for polyhedra.
        bonds = self.options.bonds
        if (
            bonds is None
            and self.options.bond_strategy is not None
            and (self.options.show_bonds or self.options.show_polyhedra)
        ):
            bonds = self.options.bond_strategy.get_bonds(structure)

        scene = Scene(metadata={"formula": structure.formula, "num_sites": structure.num_sites})

        if self.options.show_unit_cell:
            scene.extend(self._cell_edges(structure), group="cell")

        # Build polyhedra first so we know which image atoms are needed.
        polyhedra: list[Polyhedron] = []
        if self.options.show_polyhedra:
            polyhedra = self._polyhedra(structure, bonds)
            scene.extend(polyhedra, group="polyhedra")

        if self.options.show_bonds and bonds:
            scene.extend(self._bonds(structure, bonds), group="bonds")

        if self.options.show_atoms:
            scene.extend(self._atoms(structure, polyhedra), group="atoms")
            scene.extend(
                self._expand_image_atoms(structure, bonds or [], polyhedra),
                group="image_atoms",
            )

        if self.options.show_axes:
            scene.extend(self._axes(structure), group="axes")

        if self.options.vectors:
            scene.extend(self._vectors(structure), group="vectors")

        if self.options.miller_planes:
            scene.extend(self._miller_planes(structure), group="planes")

        if self.options.selected_sites:
            scene.extend(self._highlights(structure, self.options.selected_sites), group="highlights")

        if self.options.defect_sites:
            scene.extend(self._defects(structure, self.options.defect_sites), group="defects")

        if self.options.show_legend:
            scene.extend(self._legend(structure), group="legend")

        if self.options.annotations:
            scene.extend(self._annotations(structure), group="annotations")

        return scene

    def _resolve_centers(self, structure: CrystalStructure, centers: list[int | str] | str) -> list[int]:
        """Resolve a center spec to site indices, allowing element strings."""
        if isinstance(centers, str):
            indices = structure.indices_of_species(centers)
            if not indices:
                indices = structure.indices_of_element(centers)
            if not indices:
                warnings.warn(f"No sites match species/element '{centers}'.", stacklevel=2)
                return []
            if len(indices) > 1 and centers in {site.dominant_element for site in structure.sites}:
                exact = structure.indices_of_species(centers)
                if not exact:
                    warnings.warn(
                        f"Element '{centers}' is ambiguous; matched {len(indices)} sites by element.",
                        stacklevel=2,
                    )
            return indices
        indices = []
        for center in centers:
            if isinstance(center, int):
                indices.append(center)
            else:
                matches = structure.indices_of_species(center) or structure.indices_of_element(center)
                if not matches:
                    warnings.warn(f"No sites match species/element '{center}'.", stacklevel=2)
                indices.extend(matches)
        return list(dict.fromkeys(indices))

    def _make_sphere(
        self,
        structure: CrystalStructure,
        site_index: int,
        image_offset: tuple[int, int, int] = (0, 0, 0),
        radius_factor: float | None = None,
    ) -> Sphere:
        """Create a sphere for a site, optionally at a periodic image position."""
        site = structure.sites[site_index]
        frac, _ = self._fold_to_unit_cell(site.frac_coords)
        pos = structure.lattice.frac_to_cart(frac + np.asarray(image_offset, dtype=float))
        element = site.dominant_element
        color = self.palette.hex(element)
        radius = self._atom_radius(element, radius_factor)
        label = site.label or site.dominant_species
        render_style = self.theme.atom_style
        opacity = 1.0

        if site.is_disordered:
            key = (site.source_index, site.image_offset)
            if key not in _DISORDERED_WARNED:
                _DISORDERED_WARNED.add(key)
                warnings.warn(
                    f"Site {site_index} ({site.dominant_species}) is disordered; "
                    "only the dominant species is being rendered.",
                    stacklevel=2,
                )

        for override in self.options.atom_overrides:
            if override.matches(site_index, site):
                if override.color is not None:
                    color = override.color
                if override.radius is not None:
                    radius = override.radius
                if override.scale is not None:
                    radius = self._atom_radius(element, override.scale)
                if override.opacity is not None:
                    opacity = override.opacity
                if override.render_style is not None:
                    render_style = override.render_style
                if not override.visible:
                    return None  # type: ignore[return-value]

        return Sphere(
            position=pos,
            radius=radius,
            color=color,
            opacity=opacity,
            label=label,
            metadata={
                "site_index": site_index,
                "image_offset": image_offset,
                "species": site.dominant_species,
                "element": element,
            },
            render_style=render_style,
        )

    def _atom_radius(self, element: str, factor: float | None = None) -> float:
        """Return display radius for an element."""
        if factor is None:
            if self.options.show_polyhedra and self.options.polyhedra_specs:
                factor = getattr(self.theme, "atom_radius_scale_polyhedron", 0.22)
                max_radius = 0.18
            else:
                factor = getattr(self.theme, "atom_radius_scale", 0.30)
                max_radius = 0.38
        else:
            max_radius = 0.40
        base = get_radius(element, "covalent")
        # Clamp very large radii so A-sites do not swallow the cage.
        return min(base * factor, max_radius)

    def _atoms(self, structure: CrystalStructure, polyhedra: list[Polyhedron]) -> list[Sphere]:
        centers: set[int] = {poly.center_site for poly in polyhedra}
        visible_images = None
        if self.options.display_boundary == "polyhedra_complete" and polyhedra:
            visible_images = {(poly.center_site, (0, 0, 0)) for poly in polyhedra}
            visible_images.update(
                (meta["site_index"], tuple(meta.get("image_offset", (0, 0, 0))))
                for poly in polyhedra for meta in poly.vertex_metadata
            )
            center_species = {
                structure.sites[poly.center_site].dominant_element for poly in polyhedra
            }
            center_positions = [
                structure.lattice.frac_to_cart(structure.sites[poly.center_site].frac_coords)
                for poly in polyhedra
            ]
            context_radius = 1.75 * max(
                np.linalg.norm(np.asarray(poly.vertices) - center, axis=1).max()
                for poly, center in zip(polyhedra, center_positions, strict=True)
            )
            for center in center_positions:
                nearest_by_species = {}
                for index, site in enumerate(structure.sites):
                    if site.dominant_element in center_species:
                        continue
                    distance = np.linalg.norm(
                        structure.lattice.frac_to_cart(site.frac_coords) - center
                    )
                    if distance > context_radius:
                        continue
                    current = nearest_by_species.get(site.dominant_element)
                    if current is None or distance < current[0]:
                        nearest_by_species[site.dominant_element] = (distance, index)
                visible_images.update(
                    (index, (0, 0, 0)) for _, index in nearest_by_species.values()
                )
        for spec in self.options.polyhedra_specs:
            if spec.centers != "auto":
                centers.update(self._resolve_centers(structure, spec.centers))

        spheres = []
        for i in range(len(structure.sites)):
            if visible_images is not None and (i, (0, 0, 0)) not in visible_images:
                continue
            if i in centers:
                factor = getattr(self.theme, "atom_radius_scale_polyhedron_center", 0.10)
            else:
                factor = None
            sphere = self._make_sphere(structure, i, radius_factor=factor)
            if sphere is not None:
                spheres.append(sphere)
        return spheres

    def _expand_image_atoms(
        self,
        structure: CrystalStructure,
        bonds: list[NeighborBond],
        polyhedra: list[Polyhedron],
    ) -> list[Sphere]:
        """Generate spheres for periodic-image atoms according to display_boundary."""
        mode = self.options.display_boundary
        images: dict[tuple[int, tuple[int, int, int]], None] = {}

        if mode == "connected":
            for bond in bonds:
                image = self._display_jimage(structure, bond.i, bond.j, bond.jimage)
                frac = self._fold_to_unit_cell(structure.sites[bond.j].frac_coords)[0] + np.asarray(image)
                if any(image) and np.all(frac >= -1.0) and np.all(frac <= 2.0):
                    images[(bond.j, image)] = None
        elif mode == "polyhedra_complete":
            for poly in polyhedra:
                for meta in getattr(poly, "vertex_metadata", []):
                    offset = tuple(meta.get("image_offset", (0, 0, 0)))
                    frac = self._fold_to_unit_cell(structure.sites[meta["site_index"]].frac_coords)[0] + np.asarray(offset)
                    if any(offset) and np.all(frac >= -1.0) and np.all(frac <= 2.0):
                        images[(meta["site_index"], offset)] = None
        else:  # cell_complete (default)
            # Strictly display the closure of the canonical cell.  Cross-cell
            # bonds are routed to boundary replicas inside [0,1]^3 by _bonds(),
            # so we do not pull extra image atoms from neighbouring cells here.
            images = self._cell_complete_images(structure)

        spheres = [self._make_sphere(structure, idx, offset) for idx, offset in images]
        return [sphere for sphere in spheres if sphere is not None]

    def _fold_to_unit_cell(
        self,
        frac: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Fold fractional coordinates into the half-open unit cell [0,1)^3.

        Returns the folded coordinate and the integer image offset that was
        subtracted to bring it there.  Boundary coordinates at 1.0 are folded
        back to 0.0 so they map to the canonical atom; the scene builder's
        boundary replication then creates the needed replica atoms.
        """
        image = np.floor(frac).astype(int)
        folded = frac - image
        return folded, tuple(image.tolist())

    def _display_jimage(
        self,
        structure: CrystalStructure,
        site_i: int,
        site_j: int,
        jimage: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Translate a periodic bond image into the folded display cell."""
        _, image_i = self._fold_to_unit_cell(structure.sites[site_i].frac_coords)
        _, image_j = self._fold_to_unit_cell(structure.sites[site_j].frac_coords)
        return tuple(jimage[axis] + image_j[axis] - image_i[axis] for axis in range(3))

    def _cell_complete_images(
        self,
        structure: CrystalStructure,
    ) -> dict[tuple[int, tuple[int, int, int]], None]:
        """Return image offsets that reproduce atoms on the cell boundaries.

        A site at a face/edge/corner of the canonical cell is visually replicated
        so the displayed unit cell looks complete (e.g. one Ba at (0,0,0) appears
        at all 8 corners).  Interior atoms are not duplicated and the canonical
        (0,0,0) atom is *not* included here; it is rendered by ``_atoms()``.
        """
        images: dict[tuple[int, tuple[int, int, int]], None] = {}
        tol = 1e-6
        for i, site in enumerate(structure.sites):
            frac, _ = self._fold_to_unit_cell(site.frac_coords)
            ranges = []
            for f in frac:
                f = float(f) % 1.0
                opts = []
                if f < tol:
                    # Atom lies on the 0-face; also show its 1-face replica.
                    opts.extend([0, 1])
                elif f > 1.0 - tol:
                    # Atom lies on the 1-face; also show its 0-face replica.
                    opts.extend([0, -1])
                else:
                    opts.append(0)
                ranges.append(opts)
            for di in ranges[0]:
                for dj in ranges[1]:
                    for dk in ranges[2]:
                        if di == 0 and dj == 0 and dk == 0:
                            continue
                        images[(i, (di, dj, dk))] = None
        return images

    def _bonds(self, structure: CrystalStructure, bonds: list[NeighborBond]) -> list[Bond]:
        cylinders = []
        for bond_index, bond in enumerate(bonds):
            site_i = structure.sites[bond.i]
            site_j = structure.sites[bond.j]
            image_offset = self._display_jimage(structure, bond.i, bond.j, bond.jimage)
            frac_i, _ = self._fold_to_unit_cell(site_i.frac_coords)
            frac_j, _ = self._fold_to_unit_cell(site_j.frac_coords)
            start = structure.lattice.frac_to_cart(frac_i)

            # Clip periodic bonds to the visible cell/padding instead of dropping
            # cell-crossing bonds or drawing their endpoints arbitrarily far away.
            if self.options.display_boundary == "cell_complete":
                partner_frac = frac_j + np.array(image_offset, dtype=float)
                clipped = _clip_segment_box(start=frac_i, end=partner_frac, lower=0.0, upper=1.0)
                if clipped is None:
                    continue
                clipped_start, clipped_end = clipped
                start = structure.lattice.frac_to_cart(clipped_start)
                end = structure.lattice.frac_to_cart(clipped_end)
            elif self.options.display_boundary in ("connected", "polyhedra_complete"):
                partner_frac = frac_j + np.array(image_offset, dtype=float)
                clipped = _clip_segment_box(start=frac_i, end=partner_frac, lower=-1.0, upper=2.0)
                if clipped is None:
                    continue
                clipped_start, clipped_end = clipped
                start = structure.lattice.frac_to_cart(clipped_start)
                end = structure.lattice.frac_to_cart(clipped_end)
            else:
                end = structure.lattice.frac_to_cart(frac_j + np.array(image_offset, dtype=float))

            color = self._bond_color(bond)
            width = self._bond_width(bond)
            opacity = self._bond_opacity(bond)
            visible = True
            for override in self.options.bond_overrides:
                if override.matches(bond_index, bond, site_i, site_j):
                    if override.color is not None:
                        color = override.color
                    if override.width is not None:
                        width = override.width
                    if override.opacity is not None:
                        opacity = override.opacity
                    if not override.visible:
                        visible = False
            if not visible:
                continue

            cylinders.append(Bond(
                start=start,
                end=end,
                radius=width,
                color=color,
                opacity=opacity,
                site_i=bond.i,
                site_j=bond.j,
                jimage=image_offset,
                distance=bond.distance,
            ))
        return cylinders

    def _bond_color(self, bond: NeighborBond) -> str:
        return self.theme.bond_color

    def _bond_width(self, bond: NeighborBond) -> float:
        return self.theme.bond_width

    def _bond_opacity(self, bond: NeighborBond) -> float:
        return 0.85

    def _polyhedra(self, structure: CrystalStructure, bonds: list[NeighborBond] | None) -> list[Polyhedron]:
        polyhedra = []
        specs = self.options.polyhedra_specs
        if not specs:
            return polyhedra

        bonds = bonds or []
        def make_adjacency(spec_bonds):
            # Keep periodic images distinct; each spec gets its own neighbor graph.
            adjacency = {i: [] for i in range(len(structure))}
            for bond in spec_bonds:
                adjacency[bond.i].append((bond.j, bond.jimage))
                if bond.i != bond.j:
                    adjacency[bond.j].append((bond.i, tuple(-x for x in bond.jimage)))
            return adjacency

        default_adjacency = make_adjacency(bonds)

        for spec in specs:
            spec_bonds = bonds
            adjacency = default_adjacency
            if spec.strategy is not None:
                spec_bonds = spec.strategy.get_bonds(structure)
                adjacency = make_adjacency(spec_bonds)

            if spec.centers == "auto":
                centers = self._automatic_centers(structure, adjacency)
            else:
                centers = self._resolve_centers(structure, spec.centers)
            if self.options.center_region is not None:
                lower, upper = self.options.center_region
                centers = [
                    index for index in centers
                    if np.all(self._fold_to_unit_cell(structure.sites[index].frac_coords)[0] >= lower - 1e-8)
                    and np.all(self._fold_to_unit_cell(structure.sites[index].frac_coords)[0] <= upper + 1e-8)
                ]
            if spec.centers == "auto" and self.options.center_limit is not None:
                center_species = {
                    structure.sites[index].dominant_element for index in centers
                }
                limit = max(self.options.center_limit, len(center_species))
                centers = self._select_adjacent_centers(structure, centers, limit)
            fill_color = spec.fill_color or self.palette.hex("accent")
            opacity = spec.opacity if spec.opacity is not None else self.theme.polyhedron_opacity
            edge_width = spec.edge_width if spec.edge_width is not None else self.theme.polyhedron_edge_width
            edge_color = spec.edge_color or self.palette.hex("dark")

            for center_idx in centers:
                center_site = structure.sites[center_idx]
                center_color = self._coordination_color(center_site.dominant_element) if spec.color_by_center else fill_color
                center_frac, _ = self._fold_to_unit_cell(center_site.frac_coords)
                center_pos = structure.lattice.frac_to_cart(center_frac)
                nbrs = adjacency.get(center_idx, [])
                if not nbrs:
                    continue
                vertex_positions = []
                vertex_indices = []
                vertex_metadata = []
                for j, jimage in nbrs:
                    display_jimage = self._display_jimage(structure, center_idx, j, jimage)
                    neighbor_frac, _ = self._fold_to_unit_cell(structure.sites[j].frac_coords)
                    offset = np.array(display_jimage, dtype=float)
                    vertex_frac = neighbor_frac + offset
                    vertex_positions.append(structure.lattice.frac_to_cart(vertex_frac))
                    vertex_indices.append(j)
                    vertex_metadata.append({"site_index": j, "image_offset": display_jimage})
                if len(vertex_positions) < 4:
                    continue
                try:
                    cp = build_polyhedron(center_pos, vertex_positions, vertex_indices, center_idx)
                except Exception as exc:
                    warnings.warn(
                        f"Skipping polyhedron for site {center_idx} "
                        f"({center_site.dominant_element}) with {len(vertex_positions)} neighbors: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                poly = Polyhedron(
                    center_site=center_idx,
                    vertices=cp.vertex_positions,
                    faces=cp.faces,
                    fill_color=center_color,
                    edge_color=edge_color,
                    opacity=opacity,
                    edge_width=edge_width,
                    edge_mode=self.theme.polyhedron_edge_mode,
                    back_edges=self.theme.polyhedron_back_edges,
                    vertex_metadata=vertex_metadata,
                )
                if self.options.clip_polyhedra_to_cell:
                    poly = self._clip_polyhedron_to_box(structure, poly, 0.0, 1.0)
                elif self.options.display_boundary == "connected":
                    poly = self._clip_polyhedron_to_box(structure, poly, -1.0, 2.0)
                if poly is not None:
                    polyhedra.append(poly)
        return polyhedra

    def _automatic_centers(self, structure: CrystalStructure, adjacency: dict[int, list]) -> list[int]:
        """Select geometrically coordinated sites that are less electronegative than their neighbors."""
        centers = []
        for index, neighbors in adjacency.items():
            if not 4 <= len(neighbors) <= 8:
                continue
            center_x = Element(structure.sites[index].dominant_element).X
            neighbor_x = [Element(structure.sites[j].dominant_element).X for j, _ in neighbors]
            vectors = [
                structure.lattice.frac_to_cart(
                    structure.sites[j].frac_coords
                    + np.asarray(image)
                    - structure.sites[index].frac_coords
                )
                for j, image in neighbors
            ]
            if (
                center_x is not None
                and all(value is not None for value in neighbor_x)
                and float(np.mean(neighbor_x)) - center_x >= 0.4
                and coordination_shell_has_volume(vectors)
            ):
                centers.append(index)
        return centers

    @staticmethod
    def _coordination_color(element: str) -> str:
        colors = ("#43b7a4", "#8c70c7", "#e2a24c", "#4a91c2", "#d46d7a")
        return colors[int(Element(element).Z) % len(colors)]

    def _select_adjacent_centers(
        self, structure: CrystalStructure, centers: list[int], limit: int
    ) -> list[int]:
        """Keep a compact connected motif from the candidate center sites."""
        if len(centers) <= limit:
            return centers
        cartesian = {
            index: structure.lattice.frac_to_cart(
                self._fold_to_unit_cell(structure.sites[index].frac_coords)[0]
            )
            for index in centers
        }
        lower, upper = self.options.center_region or (0.0, 1.0)
        target = structure.lattice.frac_to_cart(np.full(3, (lower + upper) / 2))
        by_species: dict[str, list[int]] = {}
        for index in centers:
            by_species.setdefault(structure.sites[index].dominant_element, []).append(index)
        selected = [
            min(indices, key=lambda index: (np.linalg.norm(cartesian[index] - target), index))
            for _, indices in sorted(by_species.items())
        ][:limit]
        while len(selected) < limit:
            remaining = [index for index in centers if index not in selected]
            next_center = min(
                remaining,
                key=lambda index: (
                    min(np.linalg.norm(cartesian[index] - cartesian[chosen]) for chosen in selected),
                    index,
                ),
            )
            selected.append(next_center)
        return selected

    def _clip_polyhedron_to_box(
        self, structure: CrystalStructure, poly: Polyhedron, lower: float, upper: float
    ) -> Polyhedron | None:
        """Clip a convex coordination hull to the fractional display box."""
        frac_vertices = structure.lattice.cart_to_frac(np.asarray(poly.vertices))
        points = [
            point for point in frac_vertices
            if np.all(point >= lower - 1e-8) and np.all(point <= upper + 1e-8)
        ]
        edges = {
            tuple(sorted((face[i], face[(i + 1) % len(face)])))
            for face in poly.faces for i in range(len(face))
        }
        for i, j in edges:
            a, b = frac_vertices[i], frac_vertices[j]
            for axis in range(3):
                if abs(b[axis] - a[axis]) < 1e-12:
                    continue
                for boundary in (lower, upper):
                    t = (boundary - a[axis]) / (b[axis] - a[axis])
                    if 0.0 <= t <= 1.0:
                        point = a + t * (b - a)
                        if np.all(point >= lower - 1e-8) and np.all(point <= upper + 1e-8):
                            points.append(point)
        try:
            source_hull = ConvexHull(frac_vertices)
        except QhullError:
            return None
        for corner in product((lower, upper), repeat=3):
            corner = np.asarray(corner)
            if np.all(source_hull.equations[:, :3] @ corner + source_hull.equations[:, 3] <= 1e-8):
                points.append(corner)
        if len(points) < 4:
            return None
        unique = np.unique(np.round(points, decimals=10), axis=0)
        try:
            hull = ConvexHull(unique)
        except QhullError:
            return None
        cart_vertices = structure.lattice.frac_to_cart(unique)
        faces = merge_coplanar_faces(cart_vertices, hull.simplices)
        original_metadata = {
            tuple(np.round(frac, 10)): metadata
            for frac, metadata in zip(frac_vertices, poly.vertex_metadata, strict=False)
        }
        vertex_metadata = [
            original_metadata.get(tuple(frac), {"kind": "cut_vertex"})
            for frac in unique
        ]
        face_metadata = []
        for face in faces:
            fractional_face = unique[np.asarray(face, dtype=int)]
            on_box_plane = any(
                np.all(np.abs(fractional_face[:, axis] - boundary) <= 1e-8)
                for axis in range(3) for boundary in (lower, upper)
            )
            has_cut_vertices = sum(vertex_metadata[index].get("kind") == "cut_vertex" for index in face) >= 2
            face_metadata.append({"kind": "cut_face" if on_box_plane and has_cut_vertices else "coordination_face"})
        return Polyhedron(
            center_site=poly.center_site,
            vertices=list(cart_vertices),
            faces=faces,
            fill_color=poly.fill_color,
            edge_color=poly.edge_color,
            opacity=poly.opacity,
            edge_width=poly.edge_width,
            edge_mode=poly.edge_mode,
            back_edges=poly.back_edges,
            vertex_metadata=vertex_metadata,
            face_metadata=face_metadata,
            metadata={**poly.metadata, "clipped": True},
        )

    def _cell_edges(self, structure: CrystalStructure) -> list[CellEdge]:
        style = getattr(self.theme, "cell_style", "light")
        if style == "none":
            return []
        corners = structure.lattice.unit_cell_corners()
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        # Use the actual rendering camera (or a default) to classify front/back edges
        camera = self.camera if self.camera is not None else Camera(elevation=25.0, azimuth=45.0)
        depths = camera.depth(corners)
        center_depth = float(np.mean(depths))
        lines = []
        for i, j in edges:
            is_back = (depths[i] + depths[j]) / 2.0 < center_depth
            if style == "full":
                opacity = 0.62 if is_back else 0.88
                width_scale = 1.0
            else:
                opacity = 0.22 if is_back else 0.48
                width_scale = 0.72
            lines.append(CellEdge(
                start=corners[i],
                end=corners[j],
                color=self.palette.hex("gray"),
                linewidth=self.theme.cell_edge_width * width_scale * (0.75 if is_back else 1.0),
                opacity=opacity,
                is_back=is_back,
                layer="background",
            ))
        return lines

    def _axes(self, structure: CrystalStructure) -> list[Axis]:
        origin = np.zeros(3)
        matrix = structure.lattice.matrix
        arrows = []
        labels = ["a", "b", "c"]
        colors = ["secondary", "accent", "primary"]
        for i, (label, color_name) in enumerate(zip(labels, colors, strict=True)):
            vec = matrix[i] * 0.3
            arrows.append(Axis(
                start=origin,
                direction=vec,
                color=self.palette.hex(color_name),
                shaft_radius=0.03,
                head_radius=0.08,
                head_length=0.12,
                name=label,
                label=f"\\mathbf{{{label}}}",
            ))
        return arrows

    def _vectors(self, structure: CrystalStructure) -> list[Arrow]:
        arrows = []
        for site_idx, direction, color_key in (self.options.vectors or []):
            site = structure.sites[site_idx]
            start, _ = self._fold_to_unit_cell(site.frac_coords)
            start = structure.lattice.frac_to_cart(start)
            direction = np.asarray(direction, dtype=float)
            color = self.palette.hex(color_key) if color_key in self.palette.accents else color_key
            arrows.append(Arrow(
                start=start,
                direction=direction * self.theme.vector_scale,
                color=color,
                shaft_radius=0.03,
                head_radius=self.theme.vector_head_size,
                head_length=self.theme.vector_head_size * 1.5,
            ))
        return arrows

    def _miller_planes(self, structure: CrystalStructure) -> list[Polygon]:
        polygons = []
        for plane in (self.options.miller_planes or []):
            final_plane = MillerPlane(
                hkl=plane.hkl,
                lattice=structure.lattice,
                offset=plane.offset,
                fill_color=plane.fill_color,
                edge_color=plane.edge_color,
                opacity=plane.opacity,
            )
            pts = final_plane.intersection_polygon()
            if pts is None:
                continue
            fill = final_plane.fill_color or self.palette.hex("purple")
            edge = final_plane.edge_color or self.palette.hex("dark")
            polygons.append(Polygon(
                points=pts.tolist(),
                fill_color=fill,
                edge_color=edge,
                opacity=final_plane.opacity,
                linewidth=0.5,
            ))
        return polygons

    def _highlights(self, structure: CrystalStructure, indices: list[int]) -> list[Sphere]:
        spheres = []
        for i in indices:
            site = structure.sites[i]
            spheres.append(Sphere(
                position=structure.lattice.frac_to_cart(self._fold_to_unit_cell(site.frac_coords)[0]),
                radius=self._atom_radius(site.dominant_element) * 1.2,
                color=self.palette.hex("amber"),
                opacity=0.4,
                render_style="wireframe",
            ))
        return spheres

    def _defects(self, structure: CrystalStructure, indices: list[int]) -> list[Sphere]:
        spheres = []
        for i in indices:
            site = structure.sites[i]
            spheres.append(Sphere(
                position=structure.lattice.frac_to_cart(self._fold_to_unit_cell(site.frac_coords)[0]),
                radius=self._atom_radius(site.dominant_element) * 1.1,
                color=self.palette.hex("secondary"),
                opacity=0.5,
                render_style="wireframe",
            ))
        return spheres

    def _legend(self, structure: CrystalStructure) -> list[LegendItem]:
        items = []
        for species in structure.unique_species():
            items.append(LegendItem(
                symbol="sphere",
                text=species,
                color=self.palette.hex(element_symbol(species)),
            ))
        return items

    def _annotations(self, structure: CrystalStructure) -> list[Text]:
        texts = []
        for ann in self.options.annotations:
            fontsize = ann.fontsize or self.theme.label_size
            color = ann.color or self.theme.palette.hex("dark")
            if ann.kind == "site_label":
                if ann.site_index is None or ann.site_index >= len(structure.sites):
                    continue
                site = structure.sites[ann.site_index]
                pos = structure.lattice.frac_to_cart(self._fold_to_unit_cell(site.frac_coords)[0])
                texts.append(Text(
                    position=pos,
                    text=ann.text,
                    fontsize=fontsize,
                    color=color,
                    halign="left",
                    valign="bottom",
                    layer="annotation",
                    fontweight=ann.fontweight,
                    coordinate_space="world",
                    metadata={"offset": ann.offset, "kind": "site_label"},
                ))
            elif ann.kind == "formula_label":
                texts.append(Text(
                    position=self._screen_position(ann.position),
                    text=ann.text,
                    fontsize=fontsize,
                    color=color,
                    halign="left",
                    valign="top",
                    layer="annotation",
                    coordinate_space="screen",
                    metadata={"position": ann.position, "kind": "formula_label"},
                ))
            elif ann.kind == "panel_label":
                texts.append(Text(
                    position=self._screen_position(ann.position),
                    text=ann.text,
                    fontsize=fontsize,
                    color=color,
                    fontweight=ann.fontweight or "bold",
                    halign="left",
                    valign="top",
                    layer="annotation",
                    coordinate_space="screen",
                    metadata={"position": ann.position, "kind": "panel_label"},
                ))
        return texts

    def _screen_position(self, position: str | tuple[float, float]) -> np.ndarray:
        """Map a named screen position to normalized device coordinates."""
        mapping = {
            "top_left": np.array([0.02, 0.98]),
            "top_right": np.array([0.98, 0.98]),
            "bottom_left": np.array([0.02, 0.02]),
            "bottom_right": np.array([0.98, 0.02]),
        }
        if isinstance(position, str):
            return mapping.get(position, np.array([0.02, 0.98]))
        return np.asarray(position, dtype=float)
