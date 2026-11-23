"""High-level CrystalFigure fluent API."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from crystalfig.analysis.structure_intent import StructureIntentAnalyzer
from crystalfig.exceptions import OptionalDependencyError, StructureParseError
from crystalfig.export.exporter import Exporter, ExportResult, RenderOptions
from crystalfig.geometry.planes import MillerPlane
from crystalfig.io.loader import load_structure
from crystalfig.io.pymatgen_adapter import from_pymatgen
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborBond, NeighborStrategy
from crystalfig.neighbors.strategies import CovalentRadiiStrategy, CrystalNNStrategy, CutoffStrategy
from crystalfig.planning.camera import CameraPlanner
from crystalfig.planning.display import BondDisplayPolicy, DisplayConfig, DisplayPlanner
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
        self.config = None
        if camera is None:
            self.view_preset(self.theme.camera_preset)

    def configure(self, config) -> CrystalFigure:
        """Apply a FigureConfig while preserving the fluent legacy API."""
        from crystalfig.figure.config import FigureConfig

        if not isinstance(config, FigureConfig):
            raise TypeError("config must be a FigureConfig")
        self.config = config
        display = config.display
        if not isinstance(display, DisplayConfig):
            raise TypeError("FigureConfig.display must be a DisplayConfig")
        if isinstance(config.style, str):
            from crystalfig.styles.presets import get_style
            style = get_style(config.style)
        else:
            style = config.style
        self.theme = style.to_legacy_theme()
        configured_theme = self.theme
        self.palette = self.theme.palette.copy()
        self.theme.palette = self.palette
        camera_config = config.camera
        if camera_config.mode not in {"auto", "fixed"}:
            raise ValueError("camera mode must be 'auto' or 'fixed'")
        budget = {"max_atoms": display.max_atoms, "max_polyhedra": display.max_polyhedra}
        strategy = display.neighbor_strategy
        selected_style = {
            "vesta_ballstick": "ballstick",
            "vesta_coordination": "nature",
            "layered_crystal": "ballstick",
        }.get(style.name, "nature" if display.intent in {"coordination", "hybrid"} else None)
        display_intent = display.intent
        if display.intent == "auto" and style.name == "vesta_coordination":
            display_intent = "coordination"
        elif display.intent == "auto" and style.name == "layered_crystal":
            display_intent = "layered"
        self.publication(
            style=selected_style,
            intent=display_intent,
            region=display.region,
            framing=display.framing,
            framing_site=display.framing_site,
            centers=list(display.centers) if display.centers else None,
            strategy=strategy,
            show_atoms=display.show_atoms,
            show_bonds=display.show_bonds,
            bond_display_policy=display.bond_display_policy,
            show_unit_cell=display.show_unit_cell,
            coordination_shell=display.coordination_shell,
            display_budget={**budget, "max_bonds": max(200, display.max_atoms * 4)},
        )
        self.theme = configured_theme
        self.theme.render_quality = config.render.quality
        self.palette = configured_theme.palette
        self.camera = Camera(
            projection=camera_config.projection,
            padding=camera_config.padding,
            roll=camera_config.roll,
        )
        if camera_config.mode == "fixed":
            self.camera.elevation = camera_config.elevation if camera_config.elevation is not None else 25.0
            self.camera.azimuth = camera_config.azimuth if camera_config.azimuth is not None else 45.0
            self._auto_view = False
        else:
            self._auto_view = True
        return self

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
            self._scene_options.bonds = None
        except Exception as exc:
            raise StructureParseError(f"Could not get conventional cell: {exc}") from exc
        return self

    def primitive_cell(self) -> CrystalFigure:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            pmg = self._to_pymatgen()
            prim = SpacegroupAnalyzer(pmg).get_primitive_standard_structure()
            self.structure = from_pymatgen(prim)
            self._scene_options.bonds = None
        except Exception as exc:
            raise StructureParseError(f"Could not get primitive cell: {exc}") from exc
        return self

    def supercell(self, scaling: int | tuple[int, int, int] | np.ndarray) -> CrystalFigure:
        """Apply a scalar, diagonal tuple, or arbitrary 3x3 supercell matrix."""
        self._scene_options.supercell = scaling
        self._scene_options.bonds = None
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
        plan = getattr(self, "_display_plan", None)
        layer_normal = plan.layer_normal if plan else None
        layer_translations = plan.layer_translations if plan else None
        centroid = self._centroid()
        if plan and plan.framing == "local_coordination" and plan.selected_centers:
            centroid = self.structure.cart_coords[plan.selected_centers[0]]
        return CameraPlanner(
            self.structure, self.camera, intent=plan.intent if plan else "auto",
            layer_normal=layer_normal, layer_translations=layer_translations,
            centroid=centroid,
        ).score(scene)

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
        strategy: str | NeighborStrategy = "crystalnn",
        cutoff: float = 2.5,
        pair_cutoffs: dict[str, float] | None = None,
    ) -> CrystalFigure:
        self._scene_options.show_bonds = True
        self._scene_options.bond_display_policy = "all_bonds"
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
        strategy: str | NeighborStrategy,
        cutoff: float = 2.5,
        pair_cutoffs: dict[str, float] | None = None,
    ):
        if not isinstance(strategy, str):
            return strategy
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
        strategy: str | NeighborStrategy | None = "crystalnn",
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
        from crystalfig.styles.presets import PRESETS, get_style

        if name in PRESETS:
            style = get_style(name)
            self.theme = style.to_legacy_theme()
            self.palette = style.palette.copy()
        else:
            self.theme = FigureTheme.from_preset(name)
            self.palette = self.theme.palette.copy()
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
        strategy: str | NeighborStrategy = "crystalnn",
        camera: str | Camera | None = None,
        show_atoms: bool = True,
        show_bonds: bool | None = None,
        bond_display_policy: BondDisplayPolicy | None = None,
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
        display_budget: dict[str, int | float] | None = None,
        framing: str = "auto",
        framing_site: int | None = None,
    ) -> CrystalFigure:
        """Plan a material-agnostic publication view from periodic structure features.

        ``intent='auto'`` chooses a display grammar and finite display window from
        the selected neighbor model. Automatic neighbors and centers remain
        visualization heuristics, not unique chemical assignments.
        """
        named_figure_style = None
        if isinstance(style, str) and style in {"vesta_ballstick", "vesta_coordination", "layered_crystal"}:
            from crystalfig.styles.presets import get_style

            named_figure_style = get_style(style)
        elif style is None and getattr(self.theme, "figure_style", None) is not None:
            candidate_style = self.theme.figure_style
            if candidate_style.name in {"vesta_ballstick", "vesta_coordination", "layered_crystal"}:
                named_figure_style = candidate_style
        bond_strategy = self._make_bond_strategy(strategy)
        if bond_display_policy not in {None, "all_bonds", "center_ligand_only", "polyhedron_edges_only", "hidden"}:
            raise ValueError("bond_display_policy must be all_bonds, center_ligand_only, polyhedron_edges_only, or hidden")
        selected_bonds = bond_strategy.get_bonds(self.structure)
        intent, region, centers = DisplayPlanner.resolve_framing(
            framing, framing_site, intent, region, centers
        )
        if framing == "unit_cell" and show_unit_cell is None:
            show_unit_cell = True
        elif framing == "local_coordination" and show_unit_cell is None:
            show_unit_cell = False
        if framing_site is not None and framing_site >= self.structure.num_sites:
            raise ValueError(f"framing site index {framing_site} is outside the structure")
        selected_intent, intent_confidence = DisplayPlanner.resolve_intent(
            self.structure, intent, selected_bonds
        )
        if intent == "auto" and named_figure_style is not None:
            if named_figure_style.name == "vesta_coordination":
                selected_intent, intent_confidence = "coordination", 1.0
            elif named_figure_style.name == "layered_crystal":
                selected_intent, intent_confidence = "layered", 1.0
        requested_style = style or ("nature" if polyhedra is not None else mode)
        theme_style = requested_style if requested_style in {"journal_clean", "journal_vivid"} else None
        selected_style = (
            {"vesta_ballstick": "ballstick", "vesta_coordination": "nature", "layered_crystal": "ballstick"}[named_figure_style.name]
            if named_figure_style is not None else "nature" if theme_style else requested_style
        )
        if selected_style in {"polyhedral", "nature"}:
            selected_style = "nature"
        intent_styles = {
            "network": "ballstick", "layered": "ballstick", "ionic": "ballstick",
            "coordination": "nature", "hybrid": "nature", "minimal": "minimal",
        }
        if theme_style:
            selected_style = intent_styles[selected_intent]
        if style is None and (intent != "auto" or mode == "auto"):
            selected_style = intent_styles[selected_intent]
        if selected_style not in {"auto", "ballstick", "polyhedra", "minimal", "nature"}:
            raise ValueError("mode must be auto, ballstick, polyhedra, or minimal")
        if coordination_shell not in {"complete", "show_partial", "none"}:
            raise ValueError("coordination_shell must be complete, show_partial, or none")
        if isinstance(region, str) and region not in {"auto", "cell", "motif", "supercell"}:
            raise ValueError("region must be 'auto', 'cell', 'motif', or 'supercell'")
        theme_by_style = {
            "nature": "nature",
            "polyhedra": "polyhedra_clean",
            "minimal": "overview_minimal",
        }
        theme = theme_style or theme_by_style.get(selected_style, "ballstick_clean")
        default_boundary = (
            "polyhedra_complete"
            if selected_style == "nature" or framing == "local_coordination"
            else "connected"
        )
        if selected_style == "polyhedra":
            default_boundary = "polyhedra_complete"
        budget = {"max_atoms": 60, "max_polyhedra": 8, "max_bonds": 200}
        if display_budget:
            unknown = set(display_budget) - set(budget)
            if unknown:
                raise ValueError(f"Unknown display_budget keys: {sorted(unknown)}")
            budget.update(display_budget)
        if (
            framing == "auto" and region == "auto"
            and self._scene_options.supercell is None
            and selected_intent in {"network", "hybrid"}
            and (
                self.structure.num_sites > int(budget["max_atoms"])
                or len(selected_bonds) > int(budget["max_bonds"])
            )
        ):
            framing = "motif"
            region = "motif"
        estimated_degree = max(
            1.0,
            2.0 * len(selected_bonds)
            / max(self.structure.num_sites, 1),
        )
        spec = polyhedra
        if isinstance(spec, dict):
            centers = spec.get("centers", centers)
            polyhedron_alpha = spec.get("alpha", polyhedron_alpha)
        if coordination_shell == "none" and selected_style in {"nature", "polyhedra"}:
            selected_style = "ballstick"
            theme = "ballstick_clean"
        planner_centers = centers if centers is not None else (
            "auto" if selected_style == "nature" or framing == "local_coordination" else None
        )
        self._display_plan, budget = DisplayPlanner.plan(
            self.structure, intent=selected_intent, confidence=intent_confidence,
            region=region, coordination_shell=coordination_shell,
            display_style=selected_style, show_atoms=show_atoms,
            show_bonds=show_bonds, show_unit_cell=show_unit_cell,
            display_budget=budget, estimated_degree=estimated_degree,
            bonds=selected_bonds, centers=planner_centers,
            supercell=self._scene_options.supercell,
        )
        if framing == "local_coordination":
            self._display_plan = replace(
                self._display_plan, image_atoms="complete_coordination_shell", framing=framing
            )
        elif framing == "motif":
            self._display_plan = replace(
                self._display_plan,
                image_atoms="connected_motif",
                show_cell=bool(show_unit_cell),
                cell_frame="subtle" if show_unit_cell else "hide",
                framing=framing,
            )
        else:
            self._display_plan = replace(self._display_plan, framing=framing)
        repeats = self._display_plan.repeat
        explicit_scaling = self._scene_options.supercell
        if isinstance(region, tuple):
            display_scaling = region
        elif region == "cell":
            display_scaling = (1, 1, 1)
        else:
            display_scaling = explicit_scaling if explicit_scaling is not None else repeats
        scaling_array = np.asarray(display_scaling)
        is_unit_cell = (
            int(display_scaling) == 1 if np.isscalar(display_scaling)
            else np.array_equal(scaling_array, np.eye(3, dtype=int))
            if scaling_array.shape == (3, 3)
            else tuple(int(value) for value in display_scaling) == (1, 1, 1)
        )
        self._scene_options.supercell = None if is_unit_cell else display_scaling
        final_structure = (
            self.structure.make_supercell(display_scaling)
            if not is_unit_cell else self.structure
        )
        final_bonds = (
            bond_strategy.get_bonds(final_structure)
            if final_structure is not self.structure else selected_bonds
        )
        centers_for_selection = planner_centers
        if (
            selected_style in {"nature", "polyhedra"} or framing == "local_coordination"
        ) and (coordination_shell != "none" or framing == "local_coordination"):
            if framing == "local_coordination" and centers_for_selection == "auto":
                selected_centers = DisplayPlanner.local_center(final_structure, final_bonds)
            else:
                selected_centers = DisplayPlanner.select_centers(
                    self.structure, final_structure, final_bonds, centers_for_selection,
                    1 if framing == "local_coordination" else int(budget["max_polyhedra"]),
                    self._display_plan.center_window,
                )
            if framing == "local_coordination":
                selected_centers = selected_centers[:1]
            self._display_plan = replace(
                self._display_plan, selected_centers=selected_centers
            )
        self._scene_options.local_shell_center = (
            self._display_plan.selected_centers[0]
            if framing == "local_coordination" and self._display_plan.selected_centers
            else None
        )
        self._scene_options.show_all_cell_atoms = framing == "unit_cell"
        self._scene_options.motif_instances = (
            DisplayPlanner.representative_motif_instances(
                final_structure, final_bonds,
                limit=min(12, int(budget["max_atoms"])),
            )
            if framing == "motif" else ()
        )
        self._bond_policy_was_explicit = bond_display_policy is not None
        render_bonds = self._display_plan.show_bonds
        show_cell_frame = self._display_plan.show_cell
        self.style(theme).quality("publication").show_unit_cell(show_cell_frame)
        if named_figure_style is not None:
            self.theme = named_figure_style.to_legacy_theme()
            self.theme.render_quality = "publication"
            self.palette = named_figure_style.palette.copy()
            self.theme.palette = self.palette
        if repeats != (1, 1, 1):
            self._scene_options.center_region = self._display_plan.center_window
        else:
            self._scene_options.center_region = None
        self._scene_options.center_limit = (
            int(budget["max_polyhedra"])
            if selected_style in {"nature", "polyhedra"} else None
        )
        selected_boundary = boundary or (
            "cell" if coordination_shell == "show_partial" and framing != "local_coordination"
            else default_boundary
        )
        self._boundary_was_explicit = boundary is not None
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
        selected_bond_policy = bond_display_policy or (
            "center_ligand_only"
            if selected_intent == "coordination"
            and (selected_style in {"nature", "polyhedra"} or framing == "local_coordination")
            else "all_bonds"
        )
        if show_bonds is False:
            selected_bond_policy = "hidden"
        self._display_budget = budget
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
        self._scene_options.bond_display_policy = selected_bond_policy
        self._scene_options.bond_display_centers = tuple(self._display_plan.selected_centers)
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
                centers=list(self._display_plan.selected_centers)
                if self._display_plan.selected_centers else centers,
                strategy=None,
                opacity=polyhedron_alpha,
                color_by_center=selected_style == "nature",
                show_bonds=render_bonds,
            )
            if centers == "auto":
                self._nature_auto = True
                self.diagnostics.append(
                "Automatic coordination candidates are inferred from periodic neighbors and geometry; verify their chemical interpretation."
                )
        self._scene_options.bonds = final_bonds
        self._scene_options.bond_strategy = bond_strategy
        return self

    def _resolve_figure_intent(self, intent: str, strategy: str | NeighborStrategy) -> tuple[str, float]:
        """Choose a display grammar from periodic connectivity and local chemistry."""
        bonds = self._make_bond_strategy(strategy).get_bonds(self.structure)
        return DisplayPlanner.resolve_intent(self.structure, intent, bonds)

    def _periodic_translation_vectors(self, bonds) -> np.ndarray:
        """Return Cartesian lattice translations generated by periodic graph cycles."""
        return StructureIntentAnalyzer(self.structure).periodic_translation_vectors(bonds)

    def _periodic_connectivity_dimension(self, bonds) -> int:
        """Return the rank of lattice translations generated by graph cycles."""
        return int(np.linalg.matrix_rank(self._periodic_translation_vectors(bonds))) if bonds else 0

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

    def _publication_window(self, intent: str, region: str | tuple[int, int, int], max_atoms: int = 220):
        return DisplayPlanner.publication_window(
            self.structure, intent, region, max_atoms, self._scene_options.supercell,
            self._scene_options.bond_strategy.get_bonds(self.structure)
            if self._scene_options.bond_strategy is not None else [],
        )

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
            scene_options = replace(
                self._scene_options,
                show_bonds=True,
                bond_display_policy=(
                    self._scene_options.bond_display_policy
                    if (
                        getattr(self, "_bond_policy_was_explicit", False)
                        or not self._scene_options.show_bonds
                    )
                    else "all_bonds"
                ),
                display_boundary=(
                    self._scene_options.display_boundary
                    if getattr(self, "_boundary_was_explicit", False)
                    or self._scene_options.local_shell_center is not None
                    else "connected"
                ),
            )
            builder = SceneBuilder(
                self.structure, self.theme, self.palette,
                scene_options, camera=self.camera,
            )
            scene = builder.build()
        selected_view = None
        if self._auto_view:
            candidates = self._scored_view_candidates(scene)
            plan = getattr(self, "_display_plan", None)
            layer_normal = plan.layer_normal if plan and plan.intent == "layered" else None
            if layer_normal is not None:
                candidates = [
                    candidate for candidate in candidates
                    if abs(float(np.dot(candidate[0]["direction_cartesian"], layer_normal))) <= 0.25
                ]
            selected_view, camera = candidates[0]
            self.camera = camera
            builder.reorient_cell_edges(scene, self.camera)
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
                "framing": plan.framing,
            })
        scene.metadata["camera_used"] = selected_view["label"] if selected_view else "explicit"
        scene.metadata["camera_score"] = selected_view["score"] if selected_view else None
        scene.metadata["camera_score_components"] = selected_view if selected_view else None
        scene.metadata["view_dir_cartesian"] = (-self.camera.rotation_matrix()[2]).tolist()
        scene.metadata["camera_rotation_matrix"] = self.camera.rotation_matrix().tolist()
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
        render_config = self.config.render if self.config is not None else None
        options = RenderOptions(
            width=width if width is not None else (render_config.width_mm if render_config else theme.figure_width),
            height=height if height is not None else (render_config.height_mm if render_config else theme.figure_height),
            transparent=transparent if transparent is not None else (render_config.transparent if render_config and render_config.transparent is not None else theme.transparent),
            dpi=dpi if dpi is not None else (render_config.dpi if render_config else theme.dpi),
        )
        exporter = Exporter(scene, theme, camera=self.camera)
        selected_backend = backend or (render_config.backend if render_config else None)
        return exporter.export(path, fmt=fmt, options=options, backend=selected_backend)

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
