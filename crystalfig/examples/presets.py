"""Programmatic structure presets for examples and tests."""

from __future__ import annotations

from crystalfig.model.lattice import Lattice
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure


def rocksalt_structure(a: float = 5.64) -> CrystalStructure:
    """NaCl rocksalt structure."""
    lattice = Lattice.cubic(a)
    sites = [
        Site(frac_coords=[0.0, 0.0, 0.0], species="Na"),
        Site(frac_coords=[0.5, 0.5, 0.5], species="Na"),
        Site(frac_coords=[0.5, 0.0, 0.0], species="Cl"),
        Site(frac_coords=[0.0, 0.5, 0.5], species="Cl"),
        Site(frac_coords=[0.5, 0.5, 0.0], species="Cl"),
        Site(frac_coords=[0.0, 0.0, 0.5], species="Cl"),
        Site(frac_coords=[0.5, 0.0, 0.5], species="Cl"),
        Site(frac_coords=[0.0, 0.5, 0.0], species="Cl"),
    ]
    return CrystalStructure(lattice=lattice, sites=sites)


def diamond_structure(a: float = 5.43) -> CrystalStructure:
    """Diamond cubic Si structure."""
    lattice = Lattice.cubic(a)
    sites = [
        Site(frac_coords=[0.0, 0.0, 0.0], species="Si"),
        Site(frac_coords=[0.25, 0.25, 0.25], species="Si"),
        Site(frac_coords=[0.5, 0.5, 0.0], species="Si"),
        Site(frac_coords=[0.5, 0.0, 0.5], species="Si"),
        Site(frac_coords=[0.0, 0.5, 0.5], species="Si"),
        Site(frac_coords=[0.75, 0.75, 0.25], species="Si"),
        Site(frac_coords=[0.75, 0.25, 0.75], species="Si"),
        Site(frac_coords=[0.25, 0.75, 0.75], species="Si"),
    ]
    return CrystalStructure(lattice=lattice, sites=sites)


def perovskite_structure(a: float = 3.95, c: float | None = None) -> CrystalStructure:
    """BaTiO3 tetragonal perovskite."""
    c = c or a
    lattice = Lattice.from_parameters(a, a, c, 90.0, 90.0, 90.0)
    corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    sites = [Site(frac_coords=pt, species="Ba") for pt in corners]
    sites.append(Site(frac_coords=[0.5, 0.5, 0.5], species="Ti"))
    o_faces = [
        [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
        [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],
        [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],
    ]
    sites.extend([Site(frac_coords=pt, species="O") for pt in o_faces])
    return CrystalStructure(lattice=lattice, sites=sites)


def rutile_structure(a: float = 4.59, c: float = 2.96) -> CrystalStructure:
    """TiO2 rutile structure."""
    lattice = Lattice.from_parameters(a, a, c, 90.0, 90.0, 90.0)
    corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    sites = [Site(frac_coords=pt, species="Ti") for pt in corners]
    sites.append(Site(frac_coords=[0.5, 0.5, 0.5], species="Ti"))
    u = 0.305
    o_positions = [
        [u, u, 0.0], [1 - u, 1 - u, 0.0],
        [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5],
    ]
    sites.extend([Site(frac_coords=pt, species="O") for pt in o_positions])
    return CrystalStructure(lattice=lattice, sites=sites)


def wurtzite_structure(a: float = 3.82, c: float = 6.26) -> CrystalStructure:
    """ZnO wurtzite structure."""
    lattice = Lattice.from_parameters(a, a, c, 90.0, 90.0, 120.0)
    sites = [
        Site(frac_coords=[0.0, 0.0, 0.0], species="Zn"),
        Site(frac_coords=[1 / 3, 2 / 3, 0.5], species="Zn"),
        Site(frac_coords=[0.0, 0.0, 0.375], species="O"),
        Site(frac_coords=[1 / 3, 2 / 3, 0.875], species="O"),
    ]
    return CrystalStructure(lattice=lattice, sites=sites)


def mos2_structure(a: float = 3.19, c: float = 12.3) -> CrystalStructure:
    """MoS2 layered structure."""
    lattice = Lattice.from_parameters(a, a, c, 90.0, 90.0, 120.0)
    sites = [
        Site(frac_coords=[1 / 3, 2 / 3, 0.25], species="Mo"),
        Site(frac_coords=[2 / 3, 1 / 3, 0.75], species="Mo"),
        Site(frac_coords=[1 / 3, 2 / 3, 0.37], species="S"),
        Site(frac_coords=[2 / 3, 1 / 3, 0.63], species="S"),
        Site(frac_coords=[1 / 3, 2 / 3, 0.13], species="S"),
        Site(frac_coords=[2 / 3, 1 / 3, 0.87], species="S"),
    ]
    return CrystalStructure(lattice=lattice, sites=sites)
