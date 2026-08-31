"""Build a Scene from a CrystalStructure and visualization options."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from crystalfig.geometry.planes import MillerPlane
from crystalfig.geometry.polyhedra import build_polyhedron
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

    centers: list[int] | str
    strategy: Callable | None = None
    fill_color: str | None = None
    opacity: float | None = None
    edge_width: float | None = None
    edge_color: str | None = None
    show_bonds: bool | None = None


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
    bonds: list[NeighborBond] | None = None
    bond_strategy: Callable | None = None
    polyhedra_specs: list[PolyhedraSpec] = None  # type: ignore[assignment]
    vectors: list[tuple[int, np.ndarray, str]] | None = None
    miller_planes: list[MillerPlane] | None = None
    selected_sites: list[int] | None = None
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
            scene.extend(self._atoms(structure), group="atoms")
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

    def _resolve_centers(self, structure: CrystalStructure, centers: list[int] | str) -> list[int]:
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
        return list(centers)

    def _make_sphere(
        self,
        structure: CrystalStructure,
        site_index: int,
        image_offset: tuple[int, int, int] = (0, 0, 0),
        radius_factor: float | None = None,
    ) -> Sphere:
        """Create a sphere for a site, optionally at a periodic image position."""
        site = structure.sites[site_index]
        pos = site.cart_coords(structure.lattice) + structure.lattice.frac_to_cart(np.array(image_offset, dtype=float))
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

    def _atoms(self, structure: CrystalStructure) -> list[Sphere]:
        centers: set[int] = set()
        for spec in self.options.polyhedra_specs:
            centers.update(self._resolve_centers(structure, spec.centers))

        spheres = []
        for i in range(len(structure.sites)):
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
                if any(bond.jimage):
                    images[(bond.j, bond.jimage)] = None
        elif mode == "polyhedra_complete":
            for bond in bonds:
                if any(bond.jimage):
                    images[(bond.j, bond.jimage)] = None
            for poly in polyhedra:
                for meta in getattr(poly, "vertex_metadata", []):
                    offset = tuple(meta.get("image_offset", (0, 0, 0)))
                    if any(offset):
                        images[(meta["site_index"], offset)] = None
        else:  # cell_complete (default)
            # Strictly display the closure of the canonical cell.  Cross-cell
            # bonds are routed to boundary replicas inside [0,1]^3 by _bonds(),
            # so we do not pull extra image atoms from neighbouring cells here.
            images = self._cell_complete_images(structure)

        return [self._make_sphere(structure, idx, offset) for (idx, offset) in images]

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
            frac = site.frac_coords
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
        tol = 1e-6
        for bond_index, bond in enumerate(bonds):
            site_i = structure.sites[bond.i]
            site_j = structure.sites[bond.j]
            start = site_i.cart_coords(structure.lattice)

            # In cell_complete mode, only draw bonds whose partner lies inside or
            # on the boundary of the displayed [0,1]^3 cell.  This prevents bonds
            # from dangling at periodic images that have no displayed atom.
            if self.options.display_boundary == "cell_complete":
                partner_frac = site_j.frac_coords + np.array(bond.jimage, dtype=float)
                if not all(-tol <= f <= 1.0 + tol for f in partner_frac):
                    continue
                end = site_j.cart_coords(structure.lattice) + structure.lattice.frac_to_cart(np.array(bond.jimage))
                image_offset = bond.jimage
            else:
                end = site_j.cart_coords(structure.lattice) + structure.lattice.frac_to_cart(np.array(bond.jimage))
                image_offset = bond.jimage

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
        # Store (neighbor_index, image_offset) so periodic images are not folded back.
        adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = {i: [] for i in range(len(structure))}
        for bond in bonds:
            adjacency[bond.i].append((bond.j, bond.jimage))
            adjacency[bond.j].append((bond.i, tuple(-x for x in bond.jimage)))

        for spec in specs:
            spec_bonds = bonds
            if spec.strategy is not None:
                spec_bonds = spec.strategy.get_bonds(structure)
                # Rebuild adjacency for this spec.
                adjacency = {i: [] for i in range(len(structure))}
                for bond in spec_bonds:
                    adjacency[bond.i].append((bond.j, bond.jimage))
                    adjacency[bond.j].append((bond.i, tuple(-x for x in bond.jimage)))

            centers = self._resolve_centers(structure, spec.centers)
            fill_color = spec.fill_color or self.palette.hex("accent")
            opacity = spec.opacity if spec.opacity is not None else self.theme.polyhedron_opacity
            edge_width = spec.edge_width if spec.edge_width is not None else self.theme.polyhedron_edge_width
            edge_color = spec.edge_color or self.palette.hex("dark")

            for center_idx in centers:
                center_site = structure.sites[center_idx]
                center_pos = center_site.cart_coords(structure.lattice)
                nbrs = adjacency.get(center_idx, [])
                if not nbrs:
                    continue
                vertex_positions = []
                vertex_indices = []
                vertex_metadata = []
                for j, jimage in nbrs:
                    offset = np.array(jimage, dtype=float)
                    vertex_frac = structure.sites[j].frac_coords + offset
                    vertex_positions.append(structure.lattice.frac_to_cart(vertex_frac))
                    vertex_indices.append(j)
                    vertex_metadata.append({"site_index": j, "image_offset": jimage})
                if len(vertex_positions) < 4:
                    continue
                try:
                    cp = build_polyhedron(center_pos, vertex_positions, vertex_indices, center_idx)
                except Exception:
                    continue
                polyhedra.append(Polyhedron(
                    center_site=center_idx,
                    vertices=cp.vertex_positions,
                    faces=cp.faces,
                    fill_color=fill_color,
                    edge_color=edge_color,
                    opacity=opacity,
                    edge_width=edge_width,
                    vertex_metadata=vertex_metadata,
                ))
        return polyhedra

    def _cell_edges(self, structure: CrystalStructure) -> list[CellEdge]:
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
            lines.append(CellEdge(
                start=corners[i],
                end=corners[j],
                color=self.palette.hex("gray"),
                linewidth=self.theme.cell_edge_width * (0.55 if is_back else 1.0),
                opacity=0.35 if is_back else 0.70,
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
            start = site.cart_coords(structure.lattice)
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
                position=site.cart_coords(structure.lattice),
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
                position=site.cart_coords(structure.lattice),
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
                pos = site.cart_coords(structure.lattice)
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
