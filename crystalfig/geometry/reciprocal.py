"""Reciprocal lattice and Brillouin zone utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Voronoi

from crystalfig.exceptions import OptionalDependencyError
from crystalfig.model.lattice import Lattice


def reciprocal_lattice_vectors(lattice: Lattice) -> np.ndarray:
    """Return reciprocal lattice vectors (rows) with 2π factor."""
    return lattice.reciprocal_matrix


@dataclass
class BrillouinZone:
    """First Brillouin zone of a lattice."""

    lattice: Lattice
    vertices: np.ndarray
    edges: list[tuple[int, int]]
    faces: list[list[int]]

    @classmethod
    def from_lattice(cls, lattice: Lattice) -> BrillouinZone:
        """Construct the first Brillouin zone using Voronoi of reciprocal lattice."""
        rec = lattice.reciprocal_matrix
        # Generate reciprocal lattice points around origin
        grid = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    grid.append([i, j, k])
        grid = np.array(grid)
        points = grid @ rec
        vor = Voronoi(points)
        origin_index = np.argmin(np.linalg.norm(points, axis=1))
        region = vor.regions[vor.point_region[origin_index]]
        if -1 in region:
            region.remove(-1)

        vertices = vor.vertices[region]
        # Build edge list from ridge vertices
        edges = []
        for ridge in vor.ridge_vertices:
            if all(v in region for v in ridge):
                for i in range(len(ridge)):
                    a, b = ridge[i], ridge[(i + 1) % len(ridge)]
                    if a != -1 and b != -1:
                        edges.append((int(a), int(b)))

        return cls(
            lattice=lattice,
            vertices=vertices,
            edges=edges,
            faces=[],
        )

    def kpath_high_symmetry(self) -> dict[str, np.ndarray] | None:
        """Return high-symmetry k-points using SeeK-path if available."""
        try:
            import seekpath
        except ImportError as exc:
            raise OptionalDependencyError("seekpath", "reciprocal") from exc

        lattice = self.lattice
        a, b, c = lattice.matrix
        structure = ([a, b, c], [[0, 0, 0]], [1])
        path = seekpath.get_path(structure)
        return {label: np.array(coord) for label, coord in path["point_coords"].items()}
