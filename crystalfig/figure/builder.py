"""High-level CrystalFigure fluent API."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from crystalfig.exceptions import OptionalDependencyError, StructureParseError
from crystalfig.export.exporter import Exporter, ExportResult, RenderOptions
from crystalfig.geometry.planes import MillerPlane
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
        if name == "auto":
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

        for coefficients in product((-1, 0, 1), repeat=3):
            if sum(value != 0 for value in coefficients) < 2:
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
                if polyhedra:
                    areas = []
                    for poly in polyhedra:
                        uv = camera.project(np.asarray(poly.vertices))
                        if len(uv) >= 3:
                            try:
                                areas.append(float(ConvexHull(uv).volume))
                            except QhullError:
                                continue
                    polyhedron_area = min(float(np.mean(areas)) / max(diagonal * diagonal, 1e-9), 1.0) if areas else 0.0
                aspect_penalty = abs(np.log(frame.aspect)) + (0.025 if frame.aspect < 1 else 0.0)
                score = 1.0 - atom_overlap + 0.35 * bond_visibility + 0.45 * polyhedron_area - 0.08 * aspect_penalty
            report = {
                "label": label if roll == 0 else f"{label}_roll90",
                "direction_cartesian": direction.tolist(),
                "roll_degrees": roll,
                "score": float(score),
                "atom_overlap_fraction": float(atom_overlap),
                "bond_visibility": float(bond_visibility),
                "polyhedron_projected_area": float(polyhedron_area),
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
        show_unit_cell: bool = True,
        show_legend: bool = False,
        background: str | None = None,
        palette: str | dict | None = None,
        polyhedron_alpha: float | None = None,
        polyhedron_edge_alpha: float | None = None,
        atom_scale: float | None = None,
    ) -> CrystalFigure:
        """Apply a generic publication figure setup without material-specific rules.

        ``auto`` uses a ball-and-stick view; it reports the selected neighbor model
        so the inferred bonds are not mistaken for a unique chemical assignment.
        Use ``mode='polyhedra'`` with explicit ``centers`` when that geometry is
        the figure's intended scientific emphasis.
        """
        selected_style = style or ("nature" if polyhedra is not None else mode)
        if selected_style in {"polyhedral", "nature"}:
            selected_style = "nature"
        if selected_style not in {"auto", "ballstick", "polyhedra", "minimal", "nature"}:
            raise ValueError("mode must be auto, ballstick, polyhedra, or minimal")
        theme_by_style = {
            "nature": "nature",
            "polyhedra": "polyhedra_clean",
            "minimal": "overview_minimal",
        }
        theme = theme_by_style.get(selected_style, "ballstick_clean")
        default_boundary = "cell" if selected_style == "nature" else "connected"
        render_bonds = False if selected_style == "nature" and show_bonds is None else show_bonds
        self.style(theme).quality("publication").show_unit_cell(show_unit_cell)
        self.boundary_mode(boundary or default_boundary).view_preset(self.theme.camera_preset or "auto")
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
        if selected_style != "minimal":
            self.add_bonds(strategy=strategy)
            if render_bonds is not None:
                self._scene_options.show_bonds = render_bonds
            self.diagnostics.append(
                f"Bonds generated with {strategy}; verify coordination for the intended chemical interpretation."
            )
        if selected_style in {"nature", "polyhedra"}:
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

    def render(self, style: str = "nature", **kwargs) -> CrystalFigure:
        """Configure a deterministic publication render; call ``save`` to export."""
        return self.publication(style=style, **kwargs)

    # ------------------------------------------------------------------
    # Build & export
    # ------------------------------------------------------------------
    def build_scene(self) -> Scene:
        builder = SceneBuilder(self.structure, self.theme, self.palette, self._scene_options, camera=self.camera)
        scene = builder.build()
        poly_group = scene.get_group("polyhedra")
        if getattr(self, "_nature_auto", False) and (poly_group is None or not poly_group.primitives):
            self._scene_options.show_bonds = True
            message = "Automatic coordination geometry produced no valid polyhedra; showing the bond network instead."
            if message not in self.diagnostics:
                self.diagnostics.append(message)
            scene = SceneBuilder(self.structure, self.theme, self.palette, self._scene_options, camera=self.camera).build()
        if self._auto_view:
            _, camera = self._scored_view_candidates(scene)[0]
            self.camera = camera
            scene = SceneBuilder(
                self.structure, self.theme, self.palette, self._scene_options, camera=self.camera
            ).build()
        if self.camera.auto_fit:
            self.camera.fit_to_scene(scene)
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
