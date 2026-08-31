"""Build a Scene from a CrystalStructure and visualization options."""

from __future__ import annotations

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
)
from crystalfig.scene.scene import Scene
from crystalfig.styles.palette import ColorPalette
from crystalfig.styles.radii import get_radius
from crystalfig.styles.theme import FigureTheme


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
    supercell: tuple[int, int, int] | None = None
    display_boundary: str = "cell_complete"  # cell_complete, connected, polyhedra_complete
    bonds: list[NeighborBond] | None = None
    bond_strategy: Callable | None = None
    polyhedra_centers: list[int] | str | None = None
    polyhedra_strategy: Callable | None = None
    vectors: list[tuple[int, np.ndarray, str]] | None = None
    miller_planes: list[MillerPlane] | None = None
    selected_sites: list[int] | None = None
    defect_sites: list[int] | None = None


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

        return scene

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
        return Sphere(
            position=pos,
            radius=radius,
            color=color,
            opacity=1.0,
            label=label,
            metadata={
                "site_index": site_index,
                "image_offset": image_offset,
                "species": site.dominant_species,
                "element": element,
            },
            render_style=self.theme.atom_style,
        )

    def _atom_radius(self, element: str, factor: float | None = None) -> float:
        """Return display radius for an element."""
        if factor is None:
            if self.options.show_polyhedra:
                factor = getattr(self.theme, "atom_radius_scale_polyhedron", 0.22)
                max_radius = 0.18
            else:
                factor = getattr(self.theme, "atom_radius_scale", 0.30)
                max_radius = 0.38
        else:
            max_radius = 0.40
        base = get_radius(element, "covalent", default=0.2)
        # Clamp very large radii so A-sites do not swallow the cage.
        return min(base * factor, max_radius)

    def _atoms(self, structure: CrystalStructure) -> list[Sphere]:
        centers: set[int] = set()
        if self.options.show_polyhedra and self.options.polyhedra_centers is not None:
            pc = self.options.polyhedra_centers
            centers = set(structure.indices_of_species(pc)) if isinstance(pc, str) else set(pc)
        spheres = []
        for i in range(len(structure.sites)):
            if i in centers:
                factor = getattr(self.theme, "atom_radius_scale_polyhedron_center", 0.10)
            else:
                factor = None
            spheres.append(self._make_sphere(structure, i, radius_factor=factor))
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
        for bond in bonds:
            site_i = structure.sites[bond.i]
            site_j = structure.sites[bond.j]
            start = site_i.cart_coords(structure.lattice)

            # In cell_complete mode, route every cross-cell bond to the partner
            # replica that lies inside the displayed [0,1]^3 cell.  This avoids
            # dangling bonds pointing at atoms we are not drawing.
            if self.options.display_boundary == "cell_complete":
                partner_frac = site_j.frac_coords + np.array(bond.jimage, dtype=float)
                _, image_offset = self._fold_to_unit_cell(partner_frac)
                end = site_j.cart_coords(structure.lattice) + structure.lattice.frac_to_cart(np.array(image_offset, dtype=float))
            else:
                end = site_j.cart_coords(structure.lattice) + structure.lattice.frac_to_cart(np.array(bond.jimage))
                image_offset = bond.jimage

            color = self.theme.bond_color
            cylinders.append(Bond(
                start=start,
                end=end,
                radius=self.theme.bond_width,
                color=color,
                opacity=0.85,
                site_i=bond.i,
                site_j=bond.j,
                jimage=image_offset,
                distance=bond.distance,
            ))
        return cylinders

    def _polyhedra(self, structure: CrystalStructure, bonds: list[NeighborBond] | None) -> list[Polyhedron]:
        polyhedra = []
        centers = self.options.polyhedra_centers
        if centers is None:
            return polyhedra
        if isinstance(centers, str):
            centers = structure.indices_of_species(centers)

        bonds = bonds or []
        # Store (neighbor_index, image_offset) so periodic images are not folded back.
        adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = {i: [] for i in range(len(structure))}
        for bond in bonds:
            adjacency[bond.i].append((bond.j, bond.jimage))
            adjacency[bond.j].append((bond.i, tuple(-x for x in bond.jimage)))

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
            fill = self.options.polyhedra_strategy.get("fill_color", self.palette.hex("accent")) if isinstance(self.options.polyhedra_strategy, dict) else self.palette.hex("accent")
            polyhedra.append(Polyhedron(
                center_site=center_idx,
                vertices=cp.vertex_positions,
                faces=cp.faces,
                fill_color=fill,
                edge_color=self.palette.hex("dark"),
                opacity=self.theme.polyhedron_opacity,
                edge_width=self.theme.polyhedron_edge_width,
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
            pts = plane.intersection_polygon()
            if pts is None:
                continue
            fill = plane.fill_color or self.palette.hex("purple")
            edge = plane.edge_color or self.palette.hex("dark")
            polygons.append(Polygon(
                points=pts.tolist(),
                fill_color=fill,
                edge_color=edge,
                opacity=plane.opacity,
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
