"""High-level CrystalFigure fluent API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crystalfig.exceptions import OptionalDependencyError, StructureParseError
from crystalfig.export.exporter import Exporter, ExportResult
from crystalfig.geometry.planes import MillerPlane
from crystalfig.io.loader import load_structure
from crystalfig.io.pymatgen_adapter import from_pymatgen
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.strategies import CovalentRadiiStrategy, CrystalNNStrategy, CutoffStrategy
from crystalfig.scene.builder import SceneBuilder, SceneOptions
from crystalfig.scene.camera import Camera
from crystalfig.scene.scene import Scene
from crystalfig.styles.palette import get_palette
from crystalfig.styles.theme import FigureTheme


class CrystalFigure:
    """Fluent high-level API for building publication crystal figures."""

    def __init__(
        self,
        structure: CrystalStructure,
        theme: str | FigureTheme = "publication_muted",
        camera: Camera | None = None,
    ):
        self.structure = structure
        self.theme = theme if isinstance(theme, FigureTheme) else FigureTheme.from_preset(theme)
        self.palette = get_palette(self.theme.palette.name)
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
        """Set camera to look along a crystallographic direction."""
        if isinstance(direction, str):
            mapping = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}
            direction = mapping.get(direction, [1, 1, 0])
        direction = np.asarray(direction, dtype=float)
        # Convert fractional [uvw] to Cartesian unless already given in Cartesian.
        if direction.shape == (3,):
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

    def add_bonds(
        self,
        strategy: str = "crystalnn",
        cutoff: float = 2.5,
    ) -> CrystalFigure:
        self._scene_options.show_bonds = True
        if strategy == "crystalnn":
            strat = CrystalNNStrategy()
        elif strategy == "cutoff":
            strat = CutoffStrategy(cutoff=cutoff)
        elif strategy == "covalent":
            strat = CovalentRadiiStrategy()
        elif strategy == "ase":
            strat = _try_ase_strategy()
        else:
            raise ValueError(f"Unknown bond strategy: {strategy}")
        # Defer bond computation until build_scene so supercell is applied first.
        self._scene_options.bond_strategy = strat
        self._scene_options.bonds = None
        return self

    def add_polyhedra(
        self,
        centers: str | list[int],
        strategy: str = "crystalnn",
        fill_color: str | None = None,
        opacity: float | None = None,
    ) -> CrystalFigure:
        self._scene_options.show_polyhedra = True
        self._scene_options.polyhedra_centers = centers
        if self._scene_options.bond_strategy is None:
            self.add_bonds(strategy=strategy)
        self._scene_options.polyhedra_strategy = {"fill_color": fill_color} if fill_color else {}
        if opacity is not None:
            self.theme.polyhedron_opacity = opacity
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

    def select(self, indices: list[int] | None = None, species: str | None = None, **props) -> CrystalFigure:
        """Select sites by indices, species, or property predicate."""
        selected = set(indices or [])
        if species:
            selected.update(self.structure.indices_of_species(species))
        for key, predicate in props.items():
            for i, site in enumerate(self.structure.sites):
                val = site.properties.get(key)
                if val is not None and predicate(val):
                    selected.add(i)
        self._scene_options.selected_sites = list(selected)
        return self

    def mark_defects(self, indices: list[int]) -> CrystalFigure:
        self._scene_options.defect_sites = indices
        return self

    def style(self, name: str) -> CrystalFigure:
        self.theme = FigureTheme.from_preset(name)
        self.palette = get_palette(self.theme.palette.name)
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

    def export_tikz_pdf(self, path: str) -> ExportResult:
        """Export via TikZ + LaTeX compile to PDF."""
        scene = self.build_scene()
        exporter = Exporter(scene, self.theme, camera=self.camera)
        return exporter.export_pdf_with_latex(path)

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
