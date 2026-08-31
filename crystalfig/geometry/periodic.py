"""Periodic boundary condition utilities."""

from __future__ import annotations

import numpy as np

from crystalfig.model.lattice import Lattice


class PeriodicImages:
    """Generate periodic image sites for a structure."""

    def __init__(self, lattice: Lattice, coords: np.ndarray):
        self.lattice = lattice
        self.coords = np.asarray(coords, dtype=float)

    def images_within_radius(
        self,
        center: np.ndarray,
        radius: float,
        max_images: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return image coordinates and offsets within radius of center.

        Returns:
            images: (M, 3) Cartesian coordinates
            offsets: (M, 3) integer image offsets
        """
        center = np.asarray(center, dtype=float)
        shifts = []
        for i in range(-max_images, max_images + 1):
            for j in range(-max_images, max_images + 1):
                for k in range(-max_images, max_images + 1):
                    shifts.append([i, j, k])
        shifts = np.array(shifts)
        image_frac = self.coords[:, None, :] + shifts[None, :, :]
        image_cart = self.lattice.frac_to_cart(image_frac.reshape(-1, 3))
        center_cart = center if center.ndim == 1 else self.lattice.frac_to_cart(center)
        dists = np.linalg.norm(image_cart - center_cart, axis=1)
        mask = dists <= radius
        return image_cart[mask], shifts.reshape(-1, 3)[mask]


def nearest_image(
    frac_i: np.ndarray,
    frac_j: np.ndarray,
) -> np.ndarray:
    """Return the fractional coordinate of site j in its nearest image to site i."""
    delta = frac_j - frac_i
    delta -= np.round(delta)
    return frac_i + delta


def wrapped_frac(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into [0, 1)."""
    frac = np.asarray(frac, dtype=float)
    wrapped = frac % 1.0
    wrapped[np.abs(wrapped - 1.0) < 1e-12] = 0.0
    return wrapped
