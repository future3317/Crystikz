"""Periodic boundary condition utilities."""

from __future__ import annotations

import numpy as np

from crystalfig.model.lattice import Lattice


class PeriodicImages:
    """Generate periodic image sites for a structure."""

    def __init__(self, lattice: Lattice, coords: np.ndarray):
        self.lattice = lattice
        self.coords = np.asarray(coords, dtype=float)
        if self.coords.ndim == 1:
            self.coords = self.coords.reshape(1, -1)

    def images_within_radius(
        self,
        center: np.ndarray,
        radius: float,
        max_images: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return image coordinates and offsets within radius of center.

        Args:
            center: Cartesian coordinate of the query center.
            radius: Search radius in Cartesian units.
            max_images: Integer image search range along each axis.

        Returns:
            images: (M, 3) Cartesian coordinates
            offsets: (M, 3) integer image offsets
        """
        center = np.asarray(center, dtype=float)
        if center.ndim == 1:
            center = center.reshape(1, -1)

        n = self.coords.shape[0]
        shifts = []
        for i in range(-max_images, max_images + 1):
            for j in range(-max_images, max_images + 1):
                for k in range(-max_images, max_images + 1):
                    shifts.append([i, j, k])
        shifts = np.array(shifts, dtype=int)

        # (n, n_shifts, 3) arrays
        image_frac = self.coords[:, None, :] + shifts[None, :, :]
        image_cart = self.lattice.frac_to_cart(image_frac.reshape(-1, 3))
        center_cart = center[0]

        # Offset array broadcast to match image_cart
        image_offsets = np.broadcast_to(shifts[None, :, :], (n, len(shifts), 3)).reshape(-1, 3)

        dists = np.linalg.norm(image_cart - center_cart, axis=1)
        mask = dists <= radius
        return image_cart[mask], image_offsets[mask]


def nearest_image(
    frac_i: np.ndarray,
    frac_j: np.ndarray,
    lattice: Lattice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest image of site j relative to site i.

    Args:
        frac_i: Fractional coordinates of the reference site.
        frac_j: Fractional coordinates of the target site.
        lattice: Optional lattice for a proper triclinic minimum image search.
            If provided, returns (cartesian_image, jimage_offset); otherwise
            returns (fractional_image, zero_offset) using the simple cubic
            nearest-image convention.

    Returns:
        image: Cartesian or fractional coordinate of the nearest image of j.
        jimage: Integer periodic image offset applied to frac_j.
    """
    frac_i = np.asarray(frac_i, dtype=float)
    frac_j = np.asarray(frac_j, dtype=float)
    delta = frac_j - frac_i
    jimage = tuple(int(x) for x in -np.round(delta))
    image_frac = frac_j + np.array(jimage, dtype=float)
    if lattice is not None:
        return lattice.frac_to_cart(image_frac), jimage
    return image_frac, jimage


def wrapped_frac(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into [0, 1)."""
    frac = np.asarray(frac, dtype=float)
    wrapped = frac % 1.0
    wrapped[np.abs(wrapped - 1.0) < 1e-12] = 0.0
    return wrapped
