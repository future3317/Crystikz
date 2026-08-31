"""Lattice representation with explicit convention documentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crystalfig.constants import DEFAULT_TOLERANCE


@dataclass(frozen=True)
class Lattice:
    """3D periodic lattice.

    Convention:
        The lattice matrix ``matrix`` stores the Cartesian components of the
        direct lattice vectors **a**, **b**, **c** as *rows*::

            matrix = [[a_x, a_y, a_z],
                      [b_x, b_y, b_z],
                      [c_x, c_y, c_z]]

        This matches pymatgen/ASE convention.  Fractional coordinates ``frac``
        are column vectors and Cartesian coordinates ``cart`` are computed as::

            cart = matrix.T @ frac

        or equivalently ``cart = frac @ matrix``.

        Reciprocal lattice vectors **a***, **b***, **c*** are stored as rows of
        ``reciprocal_matrix`` and satisfy ``a* · a = 2π``, ``a* · b = 0`` etc.
    """

    matrix: np.ndarray

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float).reshape(3, 3)
        object.__setattr__(self, "matrix", matrix)
        if np.linalg.matrix_rank(matrix) < 3:
            raise ValueError("Lattice matrix is singular.")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> Lattice:
        """Build lattice from lengths (Å) and angles (degrees)."""
        alpha_r = np.radians(alpha)
        beta_r = np.radians(beta)
        gamma_r = np.radians(gamma)

        ax = a
        bx = b * np.cos(gamma_r)
        by = b * np.sin(gamma_r)
        cx = c * np.cos(beta_r)
        term = (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
        term = float(np.clip(term, -1.0, 1.0))
        cy = c * term
        cz_sq = c**2 - cx**2 - cy**2
        if cz_sq < -DEFAULT_TOLERANCE:
            raise ValueError("Invalid lattice parameters.")
        cz = float(np.sqrt(max(0.0, cz_sq)))

        mat = np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])
        return cls(mat)

    @classmethod
    def cubic(cls, a: float) -> Lattice:
        return cls.from_parameters(a, a, a, 90.0, 90.0, 90.0)

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    def frac_to_cart(self, frac: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates to Cartesian."""
        frac = np.asarray(frac, dtype=float)
        return frac @ self.matrix

    def cart_to_frac(self, cart: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to fractional."""
        cart = np.asarray(cart, dtype=float)
        return cart @ np.linalg.inv(self.matrix)

    def wrap_frac(self, frac: np.ndarray) -> np.ndarray:
        """Wrap fractional coordinates into [0, 1)."""
        frac = np.asarray(frac, dtype=float)
        wrapped = frac % 1.0
        wrapped[np.abs(wrapped - 1.0) < 1e-12] = 0.0
        return wrapped

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def lengths(self) -> tuple[float, float, float]:
        lengths = np.linalg.norm(self.matrix, axis=1)
        return float(lengths[0]), float(lengths[1]), float(lengths[2])

    @property
    def angles(self) -> tuple[float, float, float]:
        a, b, c = self.matrix
        alpha = np.degrees(np.arccos(np.clip(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c)), -1.0, 1.0)))
        beta = np.degrees(np.arccos(np.clip(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)), -1.0, 1.0)))
        gamma = np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))
        return float(alpha), float(beta), float(gamma)

    @property
    def volume(self) -> float:
        return float(np.abs(np.linalg.det(self.matrix)))

    @property
    def reciprocal_matrix(self) -> np.ndarray:
        """Reciprocal lattice vectors as rows (each multiplied by 2π)."""
        return 2.0 * np.pi * np.linalg.inv(self.matrix.T)

    def reciprocal_vector(self, hkl: np.ndarray) -> np.ndarray:
        """Return the reciprocal lattice vector G = h a* + k b* + l c*."""
        hkl = np.asarray(hkl, dtype=float)
        return hkl @ self.reciprocal_matrix

    # ------------------------------------------------------------------
    # Cell corners
    # ------------------------------------------------------------------
    def unit_cell_corners(self) -> np.ndarray:
        """Return the 8 corner Cartesian coordinates of the unit cell."""
        frac = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        return self.frac_to_cart(frac)

    def supercell_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Return the lattice matrix for a supercell transformation."""
        matrix = np.asarray(matrix, dtype=int).reshape(3, 3)
        return matrix @ self.matrix
