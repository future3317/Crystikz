"""High-level CrystalFigure fluent API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from crystalfig.exceptions import OptionalDependencyError, StructureParseError
from crystalfig.export.exporter import Exporter, ExportResult, RenderOptions
from crystalfig.geometry.planes import MillerPlane
from crystalfig.geometry.polyhedra import coordination_shell_has_volume
from crystalfig.io.loader import load_structure
from crystalfig.io.pymatgen_adapter import from_pymatgen
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborBond
from crystalfig.neighbors.strategies import CovalentRadiiStrategy, CrystalNNStrategy, CutoffStrategy
from crystalfig.renderers.matplotlib_renderer import MatplotlibRenderer
from crystalfig.scene.builder import (
    Annotation,
    AtomStyleOverride,
    BondStyleOverride,
    PolyhedraSpec,
    SceneBuilder,
    SceneOptions,
)
from crystalfig.scene.camera import Camera
from crystalfig.scene.primitives import Bond, Polyhedron
from crystalfig.scene.scene import Scene
from crystalfig.styles.palette import get_palette
from crystalfig.styles.theme import FigureTheme


@dataclass(frozen=True)
class DisplayPlan:
    """Resolved publication display choices, kept separate from structure data."""

    intent: str
    confidence: float
    repeat: tuple[int, int, int]
    coordination_shell: str
    image_atoms: str
    cell_frame: str
    center_window: tuple[float, float] | None


def _convex_polygon_intersection_area(subject, clipper):
    """Return the intersection area of two projected convex hull polygons."""
    def cross2d(first, second):
        return first[0] * second[1] - first[1] * second[0]

    polygon = [np.asarray(point, dtype=float) for point in subject]
    clipper = [np.asarray(point, dtype=float) for point in clipper]
    for start, end in zip(clipper, clipper[1:] + clipper[:1], strict=True):
        clipped = []
        for current, following in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            edge = end - start
            side_current = cross2d(edge, current - start)
            side_following = cross2d(edge, following - start)
            if side_current >= -1e-9:
                clipped.append(current)
            if (side_current >= 0) != (side_following >= 0):
                direction = following - current
                denominator = cross2d(edge, direction)
                if abs(float(denominator)) > 1e-12:
                    fraction = cross2d(edge, start - current) / denominator
                    clipped.append(current + fraction * direction)
        polygon = clipped
        if not polygon:
            return 0.0
    points = np.asarray(polygon)
    twice_area = np.dot(points[:, 0], np.roll(points[:, 1], -1))
    twice_area -= np.dot(points[:, 1], np.roll(points[:, 0], -1))
    return abs(float(twice_area)) / 2


class CrystalFigure:
    """Fluent high-level API for building publication crystal figures."""

    def __init__(
        self,
        structure: CrystalStructure,
        theme: str | FigureTheme = "publication",
        camera: Camera | None = None,
    ):
        self.structure = structure
        self.theme = theme if isinstance(theme, FigureTheme) else FigureTheme.from_preset(theme)
        self.palette = get_palette(self.theme.palette.name).copy()
        self.camera = camera or Camera(elevation=25.0, azimuth=45.0)
        self._scene_options = SceneOptions()
        self._auto_view = False
        self._nature_auto = False
        self.diagnostics: list[str] = []
        if camera is None:
            self.view_preset(self.theme.camera_preset)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> CrystalFigure:
        structure = load_structure(path)
        return cls(structure, **kwargs)

    @classmethod
    def from_structure(cls, structure, **kwargs) -> CrystalFigure:
        """Accept internal CrystalStructure, pymatgen Structure, or ASE Atoms."""
        if isinstance(structure, CrystalStructure):
            return cls(structure, **kwargs)
        try:
            from pymatgen import core as pmg_core

            pmg_structure_type = pmg_core.Structure
        except ImportError:
            pmg_structure_type = None
        if pmg_structure_type is not None and isinstance(structure, pmg_structure_type):
            return cls(from_pymatgen(structure), **kwargs)
        try:
            import ase

            ase_atoms_type = ase.Atoms
        except ImportError:
            ase_atoms_type = None
        if ase_atoms_type is not None and isinstance(structure, ase_atoms_type):
            from crystalfig.io.ase_adapter import from_ase
            return cls(from_ase(structure), **kwargs)
        raise StructureParseError("Unsupported structure type. Provide CrystalStructure, pymatgen Structure, or ASE Atoms.")

    # ------------------------------------------------------------------
    # Structure transformations
    # ------------------------------------------------------------------
    def conventional_cell(self) -> CrystalFigure:
        """Convert to conventional cell using spglib."""
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            pmg = self._to_pymatgen()
            conv = SpacegroupAnalyzer(pmg).get_conventional_standard_structure()
            self.structure = from_pymatgen(conv)
        except Exception as exc:
            raise StructureParseError(f"Could not get conventional cell: {exc}") from exc
        return self

    def primitive_cell(self) -> CrystalFigure:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            pmg = self._to_pymatgen()
            prim = SpacegroupAnalyzer(pmg).get_primitive_standard_structure()
            self.structure = from_pymatgen(prim)
        except Exception as exc:
            raise StructureParseError(f"Could not get primitive cell: {exc}") from exc
        return self

    def supercell(self, scaling: int | tuple[int, int, int] | np.ndarray) -> CrystalFigure:
        """Apply a scalar, diagonal tuple, or arbitrary 3x3 supercell matrix."""
        self._scene_options.supercell = scaling
        return self

    # ------------------------------------------------------------------
    # View / camera
    # ------------------------------------------------------------------
    def view(self, direction: str | list[int] | np.ndarray) -> CrystalFigure:
        """Set camera to look along a crystallographic direction [uvw]."""
        self._auto_view = False
        if isinstance(direction, str):
            mapping = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}
            if direction not in mapping:
                raise ValueError(f"Unknown lattice axis '{direction}'. Use 'a', 'b', or 'c'.")
            direction = mapping[direction]
        direction = np.asarray(direction, dtype=float).reshape(3)
        direction = self.structure.lattice.frac_to_cart(direction)
        self.camera = Camera.along_direction(direction, target=self._centroid())
        return self

    def view_along_lattice(self, axis: int) -> CrystalFigure:
        self._auto_view = False
        direction = self.structure.lattice.matrix[axis]
        self.camera = Camera.along_direction(direction, target=self._centroid())
        return self

    def view_preset(self, name: str) -> CrystalFigure:
        """Set a deterministic view preset or defer selection to the final scene."""
        target = self._centroid()
        directions = {
            "isometric": [1, 1, 1],
            "front": [0, 0, 1],
            "side": [0, 1, 0],
            "along_a": [1, 0, 0],
            "along_b": [0, 1, 0],
            "along_c": [0, 0, 1],
            "layer_top": [0, 0, 1],
            "layer_oblique": [1, 1, 2],
            "polyhedron_focus": [1, 1, 1],
        }
        if name in {"auto", "optimize"}:
            self._auto_view = True
            return self
        self._auto_view = False
        if name in {"layer_top", "top"}:
            self.camera = Camera.normal_to_plane((0, 0, 1), self.structure.lattice, target=target)
            return self
        if name not in directions:
            raise ValueError(f"Unknown view preset {name!r}.")
        direction = self.structure.lattice.frac_to_cart(np.asarray(directions[name], dtype=float))
        self.camera = Camera.along_direction(direction, target=target)
        return self

    def report_view_candidates(self, scene: Scene | None = None) -> list[dict]:
        """Return deterministic view candidates scored against the rendered scene."""
        if scene is None:
            scene = SceneBuilder(
                self.structure, self.theme, self.palette, self._scene_options, camera=self.camera
            ).build()
        return [report for report, _ in self._scored_view_candidates(scene)]

    def _scored_view_candidates(self, scene: Scene):
        world_bounds = scene.bounding_box()
        points = np.asarray(world_bounds) if world_bounds is not None else None
        target = points.mean(axis=0) if points is not None else self._centroid()

        directions = []
        labels = []
        for label, vector in (
            [(f"direct_{axis}", row) for axis, row in zip("abc", self.structure.lattice.matrix, strict=True)]
            + [(f"plane_{axis}", row) for axis, row in zip("abc", self.structure.lattice.reciprocal_matrix, strict=True)]
        ):
            directions.append(np.asarray(vector, dtype=float))
            labels.append(label)

        for coefficients in product((-2, -1, 0, 1, 2), repeat=3):
            if not any(coefficients):
                continue
            first_nonzero = next(value for value in coefficients if value)
            if first_nonzero < 0:
                continue
            direction = np.asarray(coefficients, dtype=float) @ self.structure.lattice.matrix
            if np.linalg.norm(direction) > 1e-8:
                directions.append(direction)
                direction_label = "".join(
                    "+" if value > 0 else "-" if value < 0 else "0" for value in coefficients
                )
                labels.append(f"lattice_{direction_label}")

        geometry_points = []
        for primitive in scene.all_primitives():
            if getattr(primitive, "layer", "geometry") == "annotation":
                continue
            if hasattr(primitive, "position"):
                geometry_points.append(np.asarray(primitive.position, dtype=float))
            elif hasattr(primitive, "vertices"):
                geometry_points.extend(np.asarray(p, dtype=float) for p in primitive.vertices)
            elif hasattr(primitive, "points"):
                geometry_points.extend(np.asarray(p, dtype=float) for p in primitive.points)
        if len(geometry_points) >= 3:
            _, _, axes = np.linalg.svd(np.asarray(geometry_points) - target, full_matrices=False)
            for index, direction in enumerate(axes[:3]):
                directions.append(direction)
                labels.append(f"principal_{index + 1}")

        base = directions[:6]
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                vector = base[i] / np.linalg.norm(base[i]) + base[j] / np.linalg.norm(base[j])
                if np.linalg.norm(vector) > 1e-8:
                    directions.append(vector)
                    labels.append(f"tilt_{labels[i]}_{labels[j]}")

        camera_candidates = []
        seen = []
        for label, direction in zip(labels, directions, strict=True):
            direction = direction / np.linalg.norm(direction)
            if any(np.dot(direction, previous) > 0.9995 for previous in seen):
                continue
            seen.append(direction)
            camera_candidates.extend(
                (label, direction, roll) for roll in (0.0, 90.0)
            )

        scored = []
        settings = self.camera
        layer_normal = (
            self._periodic_layer_normal()
            if getattr(self, "_display_plan", None) and self._display_plan.intent == "layered"
            else None
        )
        for label, direction, roll in camera_candidates:
            camera = Camera.along_direction(
                direction, target=target, projection=settings.projection, roll=roll
            )
            camera.distance = settings.distance
            camera.fov = settings.fov
            camera.padding = settings.padding
            camera.auto_fit = settings.auto_fit
            frame = camera.scene_frame(scene)
            if frame is None:
                score = 0.0
                bond_visibility = 0.0
                atom_overlap = 0.0
                polyhedron_area = 0.0
                polyhedron_depth = 0.0
                polyhedron_overlap = 0.0
            else:
                diagonal = float(np.hypot(frame.width, frame.height))
                spheres = [p for p in scene.all_primitives() if p.__class__.__name__ == "Sphere"]
                projected = camera.project(np.asarray([p.position for p in spheres])) if spheres else np.empty((0, 2))
                radii = [float(p.radius) for p in spheres]
                overlapping_atoms: set[int] = set()
                if len(spheres) > 1:
                    candidate_pairs = cKDTree(projected).query_pairs(
                        1.56 * max(radii), output_type="ndarray"
                    )
                    for i, j in candidate_pairs:
                        if np.linalg.norm(spheres[i].position - spheres[j].position) < 1e-8:
                            continue
                        if np.linalg.norm(projected[i] - projected[j]) < 0.78 * (radii[i] + radii[j]):
                            overlapping_atoms.update((int(i), int(j)))
                atom_overlap = len(overlapping_atoms) / max(len(spheres), 1)
                bonds = [p for p in scene.all_primitives() if isinstance(p, Bond)]
                bond_visibility = float(np.mean([
                    min(np.linalg.norm(camera.project(b.end) - camera.project(b.start)) / max(b.distance, 1e-9), 1.0)
                    for b in bonds
                ])) if bonds else 0.5
                polyhedra = [p for p in scene.all_primitives() if isinstance(p, Polyhedron)]
                polyhedron_area = 0.0
                polyhedron_depth = 0.0
                polyhedron_overlap = 0.0
                if polyhedra:
                    areas = []
                    depth_ratios = []
                    projected_polygons = []
                    for poly in polyhedra:
                        view_vertices = np.asarray(poly.vertices) @ camera.rotation_matrix().T
                        uv = camera.project(np.asarray(poly.vertices))
                        if len(uv) >= 3:
                            try:
                                hull = ConvexHull(uv)
                                areas.append(float(hull.volume))
                                projected_polygons.append(uv[hull.vertices])
                                projected_span = float(np.linalg.norm(np.ptp(uv, axis=0)))
                                depth_ratios.append(
                                    float(np.ptp(view_vertices[:, 2])) / max(projected_span, 1e-9)
                                )
                            except QhullError:
                                continue
                    polyhedron_area = min(float(np.mean(areas)) / max(diagonal * diagonal, 1e-9), 1.0) if areas else 0.0
                    polyhedron_depth = min(float(np.mean(depth_ratios)), 1.0) if depth_ratios else 0.0
                    overlaps = []
                    for index, first in enumerate(projected_polygons):
                        first_area = abs(float(ConvexHull(first).volume))
                        for second in projected_polygons[index + 1:]:
                            second_area = abs(float(ConvexHull(second).volume))
                            intersection = _convex_polygon_intersection_area(first, second)
                            overlaps.append(intersection / max(min(first_area, second_area), 1e-9))
                    polyhedron_overlap = float(np.mean(overlaps)) if overlaps else 0.0
                aspect_penalty = abs(np.log(frame.aspect)) + (0.025 if frame.aspect < 1 else 0.0)
                score = (
                    1.0 - atom_overlap + 0.35 * bond_visibility
                    + 0.35 * polyhedron_area + 0.22 * polyhedron_depth
                    - 0.45 * polyhedron_overlap - 0.08 * aspect_penalty
                )
                if layer_normal is not None:
                    cosine = abs(float(np.dot(direction, layer_normal)))
                    score += 2.5 * 4.0 * cosine * (1.0 - cosine)
            report = {
                "label": label if roll == 0 else f"{label}_roll90",
                "direction_cartesian": direction.tolist(),
                "roll_degrees": roll,
                "score": float(score),
                "atom_overlap_fraction": float(atom_overlap),
                "bond_visibility": float(bond_visibility),
                "polyhedron_projected_area": float(polyhedron_area),
                "polyhedron_depth_ratio": float(polyhedron_depth),
                "polyhedron_overlap_fraction": float(polyhedron_overlap),
                "projected_aspect": frame.aspect if frame is not None else None,
            }
            scored.append((report, camera))
        return sorted(scored, key=lambda item: (-item[0]["score"], item[0]["label"]))

    def view_normal_to_plane(self, hkl: tuple[int, int, int]) -> CrystalFigure:
        self._auto_view = False
        self.camera = Camera.normal_to_plane(np.array(hkl), self.structure.lattice, target=self._centroid())
        return self

    def projection(self, proj: str) -> CrystalFigure:
        self._auto_view = False
        self.camera.projection = proj  # type: ignore
        return self

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def show_unit_cell(self, show: bool = True) -> CrystalFigure:
        self._scene_options.show_unit_cell = show
        return self

    def show_axes(self, show: bool = True) -> CrystalFigure:
        self._scene_options.show_axes = show
        return self

    def boundary_mode(self, mode: str) -> CrystalFigure:
        """Set which periodic-image atoms are displayed.

        Modes:
            cell_complete: show all atoms within/on the displayed cell, including
                boundary duplicates (e.g. one corner atom appears at all 8 corners).
            connected: only show image atoms needed by visible bonds/polyhedra.
            polyhedra_complete: connected + guarantee every polyhedron vertex sphere.
        """
        self._scene_options.clip_polyhedra_to_cell = mode in {"cell", "supercell"}
        mode = {"cell": "cell_complete", "supercell": "cell_complete"}.get(mode, mode)
        if mode not in ("cell_complete", "connected", "polyhedra_complete"):
            raise ValueError(f"Unknown boundary mode: {mode}")
        self._scene_options.display_boundary = mode
        return self

    def add_bonds(
        self,
        strategy: str | object = "crystalnn",
        cutoff: float = 2.5,
        pair_cutoffs: dict[str, float] | None = None,
    ) -> CrystalFigure:
        self._scene_options.show_bonds = True
        if isinstance(strategy, str):
            strat = self._make_bond_strategy(strategy, cutoff=cutoff, pair_cutoffs=pair_cutoffs)
        else:
            strat = strategy
        # Defer bond computation until build_scene so supercell is applied first.
        self._scene_options.bond_strategy = strat
        self._scene_options.bonds = None
        return self

    def add_manual_bonds(self, bonds: list[NeighborBond]) -> CrystalFigure:
        """Add bonds from an explicit list of NeighborBond objects."""
        self._scene_options.bonds = list(bonds)
        self._scene_options.show_bonds = True
        return self

    def _make_bond_strategy(
        self,
        strategy: str,
        cutoff: float = 2.5,
        pair_cutoffs: dict[str, float] | None = None,
    ):
        if strategy == "crystalnn":
            return CrystalNNStrategy()
        if strategy == "cutoff":
            return CutoffStrategy(cutoff=cutoff, pair_cutoffs=pair_cutoffs or {})
        if strategy == "covalent":
            return CovalentRadiiStrategy()
        if strategy == "ase":
            return _try_ase_strategy()
        raise ValueError(f"Unknown bond strategy: {strategy}")

    def add_polyhedra(
        self,
        centers: str | list[int | str],
        strategy: str | object | None = "crystalnn",
        fill_color: str | None = None,
        opacity: float | None = None,
        edge_width: float | None = None,
        edge_color: str | None = None,
        show_bonds: bool | None = None,
        color_by_center: bool = False,
    ) -> CrystalFigure:
        self._scene_options.show_polyhedra = True

        strat = None
        if strategy is not None:
            strat = self._make_bond_strategy(strategy) if isinstance(strategy, str) else strategy

        # Global bond strategy: if none set, use the first polyhedra strategy.
        if self._scene_options.bond_strategy is None and strat is not None:
            self._scene_options.bond_strategy = strat

        if show_bonds is not None:
            self._scene_options.show_bonds = show_bonds

        self._scene_options.polyhedra_specs.append(PolyhedraSpec(
            centers=centers,
            strategy=strat,
            fill_color=fill_color,
            opacity=opacity,
            edge_width=edge_width,
            edge_color=edge_color,
            show_bonds=show_bonds,
            color_by_center=color_by_center,
        ))
        return self

    def add_vector(
        self,
        site_index: int,
        vector: list[float] | np.ndarray,
        color: str = "amber",
        scale: float = 1.0,
    ) -> CrystalFigure:
        if self._scene_options.vectors is None:
            self._scene_options.vectors = []
        self._scene_options.vectors.append((site_index, np.asarray(vector, dtype=float) * scale, color))
        return self

    def add_miller_plane(self, hkl: tuple[int, int, int], **kwargs) -> CrystalFigure:
        if self._scene_options.miller_planes is None:
            self._scene_options.miller_planes = []
        plane = MillerPlane(hkl=np.array(hkl), lattice=self.structure.lattice, **kwargs)
        self._scene_options.miller_planes.append(plane)
        return self

    def add_direction(self, uvw: tuple[int, int, int], **kwargs) -> CrystalFigure:
        """Draw an arrow along a crystallographic direction [uvw]."""
        direction = self.structure.lattice.frac_to_cart(np.array(uvw))
        # Attach to centroid site or first site
        return self.add_vector(0, direction, **kwargs)

    def add_label(
        self,
        site_index: int,
        text: str,
        offset: tuple[float, float] = (6, 2),
        fontsize: float | None = None,
        color: str | None = None,
        fontweight: str = "normal",
    ) -> CrystalFigure:
        """Add a screen-space label next to a site."""
        self._scene_options.annotations.append(Annotation(
            kind="site_label",
            text=text,
            site_index=site_index,
            offset=offset,
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
        ))
        return self

    def add_formula_label(
        self,
        text: str,
        position: str | tuple[float, float] = "top_left",
        fontsize: float | None = None,
        color: str | None = None,
    ) -> CrystalFigure:
        """Add a screen-space formula label."""
        self._scene_options.annotations.append(Annotation(
            kind="formula_label",
            text=text,
            position=position,
            fontsize=fontsize,
            color=color,
        ))
        return self

    def add_panel_label(
        self,
        text: str,
        position: str | tuple[float, float] = "top_left",
        fontsize: float | None = None,
        color: str | None = None,
    ) -> CrystalFigure:
        """Add a bold screen-space panel label."""
        self._scene_options.annotations.append(Annotation(
            kind="panel_label",
            text=text,
            position=position,
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        ))
        return self

    def select(
        self,
        indices: int | list[int] | None = None,
        species: str | None = None,
        **props,
    ) -> CrystalFigure:
        """Select sites by indices, species, or property predicate/value."""
        selected: set[int] = set()
        if indices is not None:
            if isinstance(indices, int):
                selected.add(indices)
            else:
                selected.update(indices)
        if species:
            selected.update(self.structure.indices_of_species(species))
        for key, predicate in props.items():
            for i, site in enumerate(self.structure.sites):
                val = site.properties.get(key)
                if val is None:
                    continue
                if callable(predicate):
                    if predicate(val):
                        selected.add(i)
                elif val == predicate:
                    selected.add(i)
        existing = set(self._scene_options.selected_sites or [])
        self._scene_options.selected_sites = sorted(existing | selected)
        return self

    def mark_defects(self, indices: list[int]) -> CrystalFigure:
        self._scene_options.defect_sites = indices
        return self

    def style_atoms(
        self,
        species: str | None = None,
        indices: list[int] | None = None,
        color: str | None = None,
        scale: float | None = None,
        opacity: float | None = None,
        radius: float | None = None,
        render_style: str | None = None,
    ) -> CrystalFigure:
        """Apply a style override to a subset of atoms."""
        self._scene_options.atom_overrides.append(AtomStyleOverride(
            species=species,
            indices=set(indices) if indices is not None else None,
            color=color,
            scale=scale,
            opacity=opacity,
            radius=radius,
            render_style=render_style,
            visible=True,
        ))
        return self

    def hide_atoms(
        self,
        species: str | None = None,
        indices: list[int] | None = None,
    ) -> CrystalFigure:
        """Hide a subset of atoms."""
        self._scene_options.atom_overrides.append(AtomStyleOverride(
            species=species,
            indices=set(indices) if indices is not None else None,
            visible=False,
        ))
        return self

    def style_bonds(
        self,
        pair: tuple | None = None,
        indices: list[int] | None = None,
        width: float | None = None,
        color: str | None = None,
        opacity: float | None = None,
    ) -> CrystalFigure:
        """Apply a style override to a subset of bonds."""
        self._scene_options.bond_overrides.append(BondStyleOverride(
            pair=pair,
            indices=set(indices) if indices is not None else None,
            width=width,
            color=color,
            opacity=opacity,
            visible=True,
        ))
        return self

    def hide_bonds(
        self,
        pair: tuple | None = None,
        indices: list[int] | None = None,
    ) -> CrystalFigure:
        """Hide a subset of bonds."""
        self._scene_options.bond_overrides.append(BondStyleOverride(
            pair=pair,
            indices=set(indices) if indices is not None else None,
            visible=False,
        ))
        return self

    def style(self, name: str) -> CrystalFigure:
        self.theme = FigureTheme.from_preset(name)
        self.palette = get_palette(self.theme.palette.name).copy()
        if self.theme.camera_preset:
            self.view_preset(self.theme.camera_preset)
        return self

    def quality(self, name: str) -> CrystalFigure:
        """Select preview, standard, or publication rendering effort."""
        if name not in {"preview", "standard", "publication"}:
            raise ValueError("quality must be preview, standard, or publication")
        self.theme.render_quality = name
        return self

    def publication(
        self,
        mode: str = "auto",
        *,
        style: str | None = None,
        boundary: str | None = None,
        polyhedra: str | dict | None = None,
        centers: str | list[int | str] | None = None,
        strategy: str = "crystalnn",
        camera: str | Camera | None = None,
        show_atoms: bool = True,
        show_bonds: bool | None = None,
        show_unit_cell: bool | None = None,
        show_legend: bool = False,
        background: str | None = None,
        palette: str | dict | None = None,
        polyhedron_alpha: float | None = None,
        polyhedron_edge_alpha: float | None = None,
        atom_scale: float | None = None,
        intent: str = "auto",
        region: str | tuple[int, int, int] = "auto",
        coordination_shell: str = "complete",
    ) -> CrystalFigure:
        """Plan a material-agnostic publication view from periodic structure features.

        ``intent='auto'`` chooses a display grammar and finite display window from
        the selected neighbor model. Automatic neighbors and centers remain
        visualization heuristics, not unique chemical assignments.
        """
        selected_intent, intent_confidence = self._resolve_figure_intent(intent, strategy)
        selected_style = style or ("nature" if polyhedra is not None else mode)
        if selected_style in {"polyhedral", "nature"}:
            selected_style = "nature"
        intent_styles = {
            "network": "ballstick", "layered": "ballstick", "ionic": "ballstick",
            "coordination": "nature", "minimal": "minimal",
        }
        if style is None and (intent != "auto" or mode == "auto"):
            selected_style = intent_styles[selected_intent]
        if selected_style not in {"auto", "ballstick", "polyhedra", "minimal", "nature"}:
            raise ValueError("mode must be auto, ballstick, polyhedra, or minimal")
        if coordination_shell not in {"complete", "show_partial", "none"}:
            raise ValueError("coordination_shell must be complete, show_partial, or none")
        if isinstance(region, str) and region not in {"auto", "cell"}:
            raise ValueError("region must be 'auto', 'cell', or a three-integer repeat tuple")
        theme_by_style = {
            "nature": "nature",
            "polyhedra": "polyhedra_clean",
            "minimal": "overview_minimal",
        }
        theme = theme_by_style.get(selected_style, "ballstick_clean")
        default_boundary = "polyhedra_complete" if selected_style == "nature" else "connected"
        if selected_style == "polyhedra":
            default_boundary = "polyhedra_complete"
        render_bonds = selected_style not in {"nature", "polyhedra"} if show_bonds is None else show_bonds
        repeats = self._publication_window(selected_intent, region)
        show_cell_frame = (
            show_unit_cell if show_unit_cell is not None
            else not (selected_style in {"nature", "polyhedra"} and repeats != (1, 1, 1))
        )
        self.style(theme).quality("publication").show_unit_cell(show_cell_frame)
        if repeats != (1, 1, 1):
            self._scene_options.supercell = repeats
            self._scene_options.center_region = (0.25, 0.75) if selected_style in {"nature", "polyhedra"} else None
            self._scene_options.center_limit = 4 if selected_style in {"nature", "polyhedra"} else None
        else:
            self._scene_options.center_region = None
            self._scene_options.center_limit = None
        selected_boundary = boundary or (
            "cell" if coordination_shell == "show_partial" else default_boundary
        )
        if coordination_shell == "none" and selected_style in {"nature", "polyhedra"}:
            selected_style = "ballstick"
            render_bonds = True if show_bonds is None else show_bonds
            theme = "ballstick_clean"
            self.style(theme)
            selected_boundary = boundary or "connected"
        self.boundary_mode(selected_boundary).view_preset(self.theme.camera_preset or "auto")
        if camera is not None:
            if isinstance(camera, Camera):
                self.camera = camera
                self._auto_view = False
            else:
                self.view_preset(camera)
        self.diagnostics = []
        self._nature_auto = False
        self._scene_options.show_atoms = show_atoms
        self._scene_options.show_legend = show_legend
        self._scene_options.show_bonds = bool(render_bonds)
        self._display_plan = DisplayPlan(
            intent=selected_intent,
            confidence=intent_confidence,
            repeat=repeats,
            coordination_shell=coordination_shell,
            image_atoms="connected_motif" if selected_style == "nature" else "finite_supercell",
            cell_frame="subtle" if show_cell_frame else "hide",
            center_window=(0.25, 0.75) if selected_style in {"nature", "polyhedra"} and repeats != (1, 1, 1) else None,
        )
        if background is not None:
            self.theme.background = "#20242c" if background == "dark" else background
            self.theme.transparent = background == "transparent"
        if isinstance(palette, str):
            self.palette = get_palette(palette).copy()
            self.theme.palette = self.palette
        elif isinstance(palette, dict):
            for element, value in palette.items():
                if isinstance(value, str):
                    from matplotlib.colors import to_rgb
                    value = tuple(round(channel * 255) for channel in to_rgb(value))
                elif max(value) <= 1:
                    value = tuple(round(channel * 255) for channel in value)
                self.palette.element_colors[element] = tuple(value)
            self.theme.palette = self.palette
        if atom_scale is not None:
            self.theme.atom_radius_scale = atom_scale
            self.theme.atom_radius_scale_polyhedron = atom_scale
        if polyhedron_alpha is not None:
            self.theme.polyhedron_opacity = polyhedron_alpha
        if polyhedron_edge_alpha is not None:
            self.theme.polyhedron_edge_opacity = polyhedron_edge_alpha
        self.add_bonds(strategy=strategy)
        self._scene_options.show_bonds = bool(render_bonds)
        self.diagnostics.append(
            f"Bonds generated with {strategy}; verify coordination for the intended chemical interpretation."
        )
        if selected_style in {"nature", "polyhedra"} and coordination_shell != "none":
            spec = polyhedra
            if isinstance(spec, dict):
                centers = spec.get("centers", centers)
                polyhedron_alpha = spec.get("alpha", polyhedron_alpha)
            if centers is None:
                centers = "auto" if selected_style == "nature" else None
            if centers is None:
                raise ValueError("centers is required when style='polyhedral'.")
            self.add_polyhedra(
                centers=centers,
                strategy=strategy,
                opacity=polyhedron_alpha,
                color_by_center=selected_style == "nature",
                show_bonds=render_bonds,
            )
            if centers == "auto":
                self._nature_auto = True
                self.diagnostics.append(
                    "Automatic centers require coordination 4–8 and a mean neighbor electronegativity at least 0.4 higher; verify the chemical interpretation."
                )
        return self

    def _resolve_figure_intent(self, intent: str, strategy: str) -> tuple[str, float]:
        """Choose a display grammar from periodic connectivity and local chemistry."""
        allowed = {"auto", "network", "coordination", "layered", "ionic", "minimal"}
        if intent not in allowed:
            raise ValueError(f"intent must be one of {sorted(allowed)}")
        if intent != "auto":
            return intent, 1.0

        bonds = self._make_bond_strategy(strategy).get_bonds(self.structure)
        coordination = np.zeros(self.structure.num_sites, dtype=int)
        contrasts = []
        for bond in bonds:
            coordination[bond.i] += 1
            coordination[bond.j] += 1
            x_first = self._electronegativity(self.structure.sites[bond.i].dominant_element)
            x_second = self._electronegativity(self.structure.sites[bond.j].dominant_element)
            if x_first is not None and x_second is not None:
                contrasts.append(abs(x_first - x_second))
        mean_contrast = float(np.mean(contrasts)) if contrasts else 0.0
        periodic_dimension = self._periodic_connectivity_dimension(bonds)
        if periodic_dimension == 2:
            return "layered", 0.84
        if mean_contrast >= 0.75 and np.median(coordination) >= 5 and np.std(coordination) < 1.0:
            return "ionic", 0.82
        centers = self._automatic_center_candidates(bonds)
        center_species = {
            self.structure.sites[index].dominant_element for index in centers
        }
        if centers and len(center_species) <= 2 and np.median(coordination) >= 3:
            return "coordination", 0.78
        if np.mean(coordination) >= 2.5:
            return "network", 0.72
        return "minimal", 0.58

    def _periodic_translation_vectors(self, bonds) -> np.ndarray:
        """Return Cartesian lattice translations generated by periodic graph cycles."""
        adjacency = [[] for _ in self.structure.sites]
        for bond in bonds:
            image = np.asarray(bond.jimage, dtype=int)
            adjacency[bond.i].append((bond.j, image))
            adjacency[bond.j].append((bond.i, -image))

        offsets: dict[int, np.ndarray] = {}
        translations = []
        for root in range(len(adjacency)):
            if root in offsets:
                continue
            offsets[root] = np.zeros(3, dtype=int)
            pending = [root]
            while pending:
                site = pending.pop()
                for neighbor, image in adjacency[site]:
                    candidate = offsets[site] + image
                    if neighbor not in offsets:
                        offsets[neighbor] = candidate
                        pending.append(neighbor)
                    else:
                        translation = candidate - offsets[neighbor]
                        if np.any(translation):
                            translations.append(translation)
        if not translations:
            return np.empty((0, 3))
        independent = []
        for translation in translations:
            if np.linalg.matrix_rank(np.asarray(independent + [translation])) > len(independent):
                independent.append(translation)
        return np.asarray(independent, dtype=float) @ self.structure.lattice.matrix

    def _periodic_connectivity_dimension(self, bonds) -> int:
        """Return the rank of lattice translations generated by graph cycles."""
        translations = self._periodic_translation_vectors(bonds)
        return int(np.linalg.matrix_rank(translations)) if len(translations) else 0

    def _periodic_layer_normal(self) -> np.ndarray | None:
        strategy = self._scene_options.bond_strategy
        if strategy is None:
            return None
        translations = self._periodic_translation_vectors(strategy.get_bonds(self.structure))
        if len(translations) != 2:
            return None
        normal = np.cross(translations[0], translations[1])
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-12 else None

    @staticmethod
    def _electronegativity(element: str) -> float | None:
        from pymatgen.core import Element
        return Element(element).X

    def _automatic_center_candidates(self, bonds):
        neighbors = [[] for _ in range(self.structure.num_sites)]
        for bond in bonds:
            neighbors[bond.i].append((bond.j, bond.jimage))
            neighbors[bond.j].append((bond.i, tuple(-value for value in bond.jimage)))
        candidates = []
        for index, adjacent in enumerate(neighbors):
            if not 4 <= len(adjacent) <= 8:
                continue
            center = self._electronegativity(self.structure.sites[index].dominant_element)
            surrounding = [
                self._electronegativity(self.structure.sites[j].dominant_element)
                for j, _ in adjacent
            ]
            vectors = [
                self.structure.lattice.frac_to_cart(
                    self.structure.sites[j].frac_coords
                    + np.asarray(image)
                    - self.structure.sites[index].frac_coords
                )
                for j, image in adjacent
            ]
            if (
                center is not None
                and all(value is not None for value in surrounding)
                and float(np.mean(surrounding)) - center >= 0.4
                and coordination_shell_has_volume(vectors)
            ):
                candidates.append(index)
        return candidates

    def _publication_window(self, intent: str, region: str | tuple[int, int, int]):
        if isinstance(region, tuple):
            if len(region) != 3 or any(not isinstance(size, int) or size < 1 for size in region):
                raise ValueError("region repeat tuple must contain three positive integers")
            return region
        if region == "cell":
            return (1, 1, 1)
        if self._scene_options.supercell is not None:
            scaling = self._scene_options.supercell
            matrix = np.asarray(scaling)
            return tuple(int(value) for value in np.diag(matrix)) if matrix.shape == (3, 3) else (int(scaling),) * 3 if np.isscalar(scaling) else tuple(int(value) for value in scaling)
        lengths = self.structure.lattice.lengths
        if intent == "layered" or (
            intent == "network" and np.max(lengths) > 1.7 * np.median(lengths)
        ):
            longest = int(np.argmax(lengths))
            return tuple(1 if axis == longest else 2 for axis in range(3))
        if intent == "ionic":
            return (2, 2, 1)
        return (2, 2, 2) if intent in {"network", "coordination"} else (1, 1, 1)

    def render(self, style: str = "nature", **kwargs) -> CrystalFigure:
        """Configure a deterministic publication render; call ``save`` to export."""
        return self.publication(style=style, **kwargs)

    # ------------------------------------------------------------------
    # Build & export
    # ------------------------------------------------------------------
    def build_scene(self) -> Scene:
        scene_options = self._scene_options
        builder = SceneBuilder(self.structure, self.theme, self.palette, scene_options, camera=self.camera)
        scene = builder.build()
        poly_group = scene.get_group("polyhedra")
        if getattr(self, "_nature_auto", False) and (poly_group is None or not poly_group.primitives):
            message = "Automatic coordination geometry produced no valid polyhedra; showing the bond network instead."
            if message not in self.diagnostics:
                self.diagnostics.append(message)
            scene_options = replace(self._scene_options, show_bonds=True)
            scene = SceneBuilder(
                self.structure, self.theme, self.palette,
                scene_options, camera=self.camera,
            ).build()
        selected_view = None
        if self._auto_view:
            selected_view, camera = self._scored_view_candidates(scene)[0]
            self.camera = camera
            scene = SceneBuilder(
                self.structure, self.theme, self.palette, scene_options, camera=self.camera
            ).build()
        if self.camera.auto_fit:
            self.camera.fit_to_scene(scene)
        plan = getattr(self, "_display_plan", None)
        if plan is not None:
            scene.metadata.update({
                "selected_intent": plan.intent,
                "intent_confidence": plan.confidence,
                "display_window": list(plan.repeat),
                "coordination_shell_policy": plan.coordination_shell,
                "image_atom_policy": plan.image_atoms,
                "cell_frame_policy": plan.cell_frame,
            })
        scene.metadata["camera_used"] = selected_view["label"] if selected_view else "explicit"
        scene.metadata["view_dir_cartesian"] = (-self.camera.rotation_matrix()[2]).tolist()
        scene.metadata["projection"] = self.camera.projection
        scene.metadata["roll"] = self.camera.roll
        polyhedra = scene.get_group("polyhedra")
        polys = polyhedra.primitives if polyhedra else []
        scene.metadata["num_real_polyhedra"] = sum(not poly.metadata.get("clipped", False) for poly in polys)
        scene.metadata["num_clipped_polyhedra"] = sum(bool(poly.metadata.get("clipped", False)) for poly in polys)
        return scene

    def draw(self, ax, options: RenderOptions | None = None) -> None:
        """Build the scene and draw it into an existing Matplotlib Axes."""
        scene = self.build_scene()
        renderer = MatplotlibRenderer(camera=self.camera)
        options = options or RenderOptions(
            width=self.theme.figure_width,
            height=self.theme.figure_height,
            transparent=self.theme.transparent,
            dpi=self.theme.dpi,
        )
        renderer.draw(ax, scene, self.theme, options)

    def export(
        self,
        path: str,
        fmt: str | None = None,
        width: float | None = None,
        transparent: bool | None = None,
        *,
        backend: str | None = None,
        dpi: int | None = None,
        height: float | None = None,
    ) -> ExportResult:
        """Export via an explicit renderer or choose one from the file format.

        ``backend`` and ``dpi`` are keyword-only additions; the original positional
        ``fmt, width, transparent`` arguments retain their meaning.
        """
        scene = self.build_scene()
        theme = self.theme
        options = RenderOptions(
            width=width if width is not None else theme.figure_width,
            height=height if height is not None else theme.figure_height,
            transparent=transparent if transparent is not None else theme.transparent,
            dpi=dpi if dpi is not None else theme.dpi,
        )
        exporter = Exporter(scene, theme, camera=self.camera)
        return exporter.export(path, fmt=fmt, options=options, backend=backend)

    def save(self, path: str, **kwargs) -> ExportResult:
        """Save the current figure using the configured export backend."""
        return self.export(path, **kwargs)

    def export_scene(self, scene: Scene, path: str, fmt: str | None = None, options=None) -> ExportResult:
        """Export a user-modified Scene directly."""
        exporter = Exporter(scene, self.theme, camera=self.camera)
        return exporter.export(path, fmt=fmt, options=options)

    def export_tikz_pdf(self, path: str) -> ExportResult:
        """Export via TikZ + LaTeX compile to PDF."""
        scene = self.build_scene()
        exporter = Exporter(scene, self.theme, camera=self.camera)
        return exporter.export_pdf_with_latex(path)

    def export_hybrid_pdf(
        self,
        path: str,
        width: float | None = None,
        height: float | None = None,
        dpi: int | None = None,
        transparent: bool | None = None,
        background_backend: str = "pyvista",
    ) -> ExportResult:
        """Export a 3D raster scene with editable vector annotations in PDF."""
        scene = self.build_scene()
        options = RenderOptions(
            width=width if width is not None else self.theme.figure_width,
            height=height if height is not None else self.theme.figure_height,
            dpi=dpi if dpi is not None else self.theme.dpi,
            transparent=transparent if transparent is not None else self.theme.transparent,
        )
        return Exporter(scene, self.theme, camera=self.camera).export_hybrid_pdf(
            path, options, background_backend=background_backend
        )

    def export_3d(
        self,
        path: str,
        width: float | None = None,
        height: float | None = None,
        transparent: bool | None = None,
    ) -> ExportResult:
        """Export using Matplotlib's true 3D projection."""
        from crystalfig.export.exporter import RenderOptions

        scene = self.build_scene()
        exporter = Exporter(scene, self.theme, camera=self.camera)
        options = RenderOptions(
            width=width or self.theme.figure_width,
            height=height,
            transparent=transparent if transparent is not None else self.theme.transparent,
            dpi=self.theme.dpi,
        )
        return exporter.export_3d(path, options)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_pymatgen(self):
        from crystalfig.io.pymatgen_adapter import to_pymatgen
        return to_pymatgen(self.structure)

    def _centroid(self) -> np.ndarray:
        coords = self.structure.cart_coords
        return np.mean(coords, axis=0)

    def quick(self) -> CrystalFigure:
        """Apply a quick display heuristic with a fixed 3 Å bond cutoff."""
        self.diagnostics = [
            "Quick display uses a 3.0 Å cutoff to draw candidate neighbors; verify coordination before chemical interpretation."
        ]
        return self.show_unit_cell().add_bonds(strategy="cutoff", cutoff=3.0)


def plot_structure(structure: str | Path | CrystalStructure, **kwargs) -> CrystalFigure:
    """One-line helper to create a CrystalFigure from a file or structure."""
    if isinstance(structure, (str, Path)):
        return CrystalFigure.from_file(structure, **kwargs)
    return CrystalFigure.from_structure(structure, **kwargs)


def _try_ase_strategy():
    try:
        from crystalfig.neighbors.strategies import ASEStrategy
        return ASEStrategy()
    except OptionalDependencyError as exc:
        raise OptionalDependencyError("ase", "ase") from exc
