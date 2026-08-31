"""Miller plane construction and intersection with unit cell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crystalfig.exceptions import GeometryError
from crystalfig.model.lattice import Lattice


@dataclass
class MillerPlane:
    """A crystallographic plane (hkl)."""

    hkl: np.ndarray
    lattice: Lattice
    offset: float | None = None
    fill_color: str | None = None
    edge_color: str | None = None
    opacity: float = 0.3

    def __post_init__(self):
        self.hkl = np.asarray(self.hkl, dtype=float)
        if np.linalg.norm(self.hkl) < 1e-12:
            raise GeometryError("Miller indices cannot all be zero.")

    @property
    def normal(self) -> np.ndarray:
        """Return the reciprocal-space normal vector (not normalized)."""
        return self.lattice.reciprocal_vector(self.hkl)

    def cartesian_equation(self) -> tuple:
        """Return plane as (normal, d) such that normal·x = d.

        ``offset`` is the Cartesian distance from the origin along the unit
        normal.  If ``None``, the offset is chosen so the plane passes through
        the centre of the unit cell.
        """
        n = self.normal
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise GeometryError("Plane normal has zero length.")
        n_unit = n / n_norm

        sc = np.eye(3, dtype=int) if self.offset is not None else None
        corners_frac = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        corners = self.lattice.frac_to_cart(corners_frac @ (sc.T if sc is not None else np.eye(3)))
        values = corners @ n_unit
        d = float((values.min() + values.max()) / 2.0) if self.offset is None else self.offset
        return n, d

    def intersection_polygon(self, supercell: np.ndarray | None = None) -> np.ndarray | None:
        """Compute the intersection polygon of the plane with the cell edges."""
        sc = np.eye(3, dtype=int) if supercell is None else np.asarray(supercell, dtype=int).reshape(3, 3)
        corners_frac = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        corners = self.lattice.frac_to_cart(corners_frac @ sc.T)

        n, d = self.cartesian_equation()
        if np.linalg.norm(n) < 1e-12:
            return None
        n_unit = n / np.linalg.norm(n)
        values = corners @ n_unit - d

        # Edge list of the unit cell
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        points = []
        for i, j in edges:
            vi, vj = values[i], values[j]
            if abs(vi - vj) < 1e-12:
                continue
            t = vi / (vi - vj)
            if 0 <= t <= 1:
                pt = corners[i] + t * (corners[j] - corners[i])
                points.append(pt)

        if len(points) < 3:
            return None

        points = np.array(points)
        # Sort points around centroid in plane-local coordinates
        centroid = points.mean(axis=0)
        centered = points - centroid
        # Build orthonormal basis in plane
        u_norm = np.linalg.norm(centered[0])
        if u_norm < 1e-12:
            return points
        u = centered[0] / u_norm
        v = np.cross(n_unit, u)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            return points
        v = v / v_norm
        angles = np.arctan2(centered @ v, centered @ u)
        order = np.argsort(angles)
        return points[order]
