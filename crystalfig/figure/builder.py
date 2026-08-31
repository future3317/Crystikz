"""High-level CrystalFigure fluent API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
            from pymatgen.core import Structure as PmgStructure
            if isinstance(structure, PmgStructure):
                return cls(from_pymatgen(structure), **kwargs)
        except Exception:
            pass
        try:
            from ase import Atoms
            if isinstance(structure, Atoms):
                from crystalfig.io.ase_adapter import from_ase
                return cls(from_ase(structure), **kwargs)
        except Exception:
            pass
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

    def supercell(self, scaling: int | tuple[int, int, int]) -> CrystalFigure:
        self._scene_options.supercell = scaling if isinstance(scaling, tuple) else (scaling, scaling, scaling)
        return self

    # ------------------------------------------------------------------
    # View / camera
    # ------------------------------------------------------------------
    def view(self, direction: str | list[int] | np.ndarray) -> CrystalFigure:
        """Set camera to look along a crystallographic direction [uvw]."""
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
        direction = self.structure.lattice.matrix[axis]
        self.camera = Camera.along_direction(direction, target=self._centroid())
        return self

    def view_normal_to_plane(self, hkl: tuple[int, int, int]) -> CrystalFigure:
        self.camera = Camera.normal_to_plane(np.array(hkl), self.structure.lattice, target=self._centroid())
        return self

    def projection(self, proj: str) -> CrystalFigure:
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
        centers: str | list[int],
        strategy: str | object | None = "crystalnn",
        fill_color: str | None = None,
        opacity: float | None = None,
        edge_width: float | None = None,
        edge_color: str | None = None,
        show_bonds: bool | None = None,
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
        return self

    # ------------------------------------------------------------------
    # Build & export
    # ------------------------------------------------------------------
    def build_scene(self) -> Scene:
        builder = SceneBuilder(self.structure, self.theme, self.palette, self._scene_options, camera=self.camera)
        scene = builder.build()
        # Apply camera auto-fit using the full projected primitive extent.
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

    def export(self, path: str, fmt: str | None = None, width: float | None = None, transparent: bool | None = None) -> ExportResult:
        from crystalfig.export.exporter import RenderOptions
        scene = self.build_scene()
        theme = self.theme
        options = RenderOptions(
            width=width or theme.figure_width,
            height=theme.figure_height,
            transparent=transparent if transparent is not None else theme.transparent,
            dpi=theme.dpi,
        )
        exporter = Exporter(scene, theme, camera=self.camera)
        return exporter.export(path, fmt=fmt, options=options)

    def export_scene(self, scene: Scene, path: str, fmt: str | None = None, options=None) -> ExportResult:
        """Export a user-modified Scene directly."""
        exporter = Exporter(scene, self.theme, camera=self.camera)
        return exporter.export(path, fmt=fmt, options=options)

    def export_tikz_pdf(self, path: str) -> ExportResult:
        """Export via TikZ + LaTeX compile to PDF."""
        scene = self.build_scene()
        exporter = Exporter(scene, self.theme, camera=self.camera)
        return exporter.export_pdf_with_latex(path)

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
        """Apply a sensible default visualization."""
        return (
            self.show_unit_cell()
            .add_bonds(strategy="cutoff", cutoff=3.0)
        )


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
