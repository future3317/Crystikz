"""Atomic radii for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class AtomicRadii:
    """A set of atomic radii in angstroms."""

    name: str
    radii: dict[str, float]

    def get(self, element: str, default: float = 0.2) -> float:
        return self.radii.get(element, default)


# Covalent radii in angstroms (Cordero et al., 2008)
COVALENT = AtomicRadii(
    name="covalent",
    radii={
        "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
        "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
        "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
        "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
        "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
        "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
        "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
        "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
        "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
        "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
        "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
        "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pb": 1.46,
    },
)

# Van der Waals radii in angstroms (Bondi, 1964)
VDW = AtomicRadii(
    name="vdw",
    radii={
        "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92,
        "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54,
        "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80,
        "S": 1.80, "Cl": 1.75, "Ar": 1.88, "K": 2.75, "Ca": 2.31,
        "Fe": 2.04, "Cu": 1.96, "Zn": 2.01, "Br": 1.85, "I": 1.98,
    },
)

UNIFORM = AtomicRadii(
    name="uniform",
    radii={},
)

_RADII_SETS = {
    "covalent": COVALENT,
    "vdw": VDW,
    "van_der_waals": VDW,
    "uniform": UNIFORM,
}


def get_radius(
    element: str,
    kind: Literal["covalent", "vdw", "uniform"] = "covalent",
    default: float = 0.2,
    uniform_value: float | None = None,
) -> float:
    """Get atomic radius in angstroms."""
    radii = _RADII_SETS.get(kind, COVALENT)
    if kind == "uniform" and uniform_value is not None:
        return uniform_value
    return radii.get(element, default)
