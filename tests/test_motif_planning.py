from pathlib import Path

import pytest

from crystalfig.analysis.coordination import CoordinationAnalyzer
from crystalfig.io.loader import load_structure
from crystalfig.model.lattice import Lattice
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure
from crystalfig.neighbors.base import NeighborBond
from crystalfig.neighbors.strategies import CrystalNNStrategy
from crystalfig.planning.display import DisplayPlanner, MotifConfig


def disconnected_structure():
    return CrystalStructure(
        Lattice.from_parameters(10, 10, 10, 90, 90, 90),
        [
            Site([0.45, 0.5, 0.5], "Ce"),
            Site([0.5, 0.5, 0.5], "Te"),
            Site([0.2, 0.2, 0.2], "Cd"),
            Site([0.25, 0.2, 0.2], "Cd"),
        ],
    )


def test_species_balanced_motif_covers_disconnected_elements_without_bonds():
    structure = disconnected_structure()
    instances, species, missing = DisplayPlanner.plan_motif(
        structure, [], MotifConfig(max_atoms=3)
    )

    assert {structure.sites[index].dominant_element for index, _ in instances} == {"Ce", "Te", "Cd"}
    assert species == ("Cd", "Ce", "Te")
    assert missing == ()
    assert len(instances) == 3


def test_coverage_reports_missing_species_and_strict_mode_names_them():
    structure = disconnected_structure()
    _, _, missing = DisplayPlanner.plan_motif(
        structure, [], MotifConfig(max_atoms=1, seed_site=0, required_species=("Te", "Cd"))
    )
    assert missing == ("Te", "Cd")
    with pytest.raises(ValueError, match="required species: Te, Cd"):
        DisplayPlanner.plan_motif(
            structure, [], MotifConfig(
                max_atoms=1, seed_site=0, required_species=("Te", "Cd"), strict_species_coverage=True
            )
        )


def test_default_coverage_includes_minor_positive_occupancies():
    structure = CrystalStructure(
        Lattice.from_parameters(5, 5, 5, 90, 90, 90),
        [Site([0.5, 0.5, 0.5], {"Si": 0.9, "O": 0.1})],
    )
    _, species, missing = DisplayPlanner.plan_motif(structure, [], MotifConfig(max_atoms=1))
    assert species == ("Si",)
    assert missing == ("O",)
    with pytest.raises(ValueError, match="framing='unit_cell'"):
        DisplayPlanner.plan_motif(
            structure, [], MotifConfig(max_atoms=1, strict_species_coverage=True)
        )


def test_species_balanced_reserves_last_slot_for_unconnected_required_species():
    structure = CrystalStructure(
        Lattice.from_parameters(5, 5, 5, 90, 90, 90),
        [Site([0.1, 0.1, 0.1], "Cd"), Site([0.7, 0.7, 0.7], "Te")],
    )
    bonds = [NeighborBond(0, 0, (1, 0, 0), 5.0)]
    instances, species, missing = DisplayPlanner.plan_motif(
        structure, bonds, MotifConfig(max_atoms=2, seed_site=0)
    )
    assert {structure.sites[index].dominant_element for index, _ in instances} == {"Cd", "Te"}
    assert species == ("Cd", "Te")
    assert missing == ()
    assert instances[1][0] == 1
    assert instances[1][1] == (-1, -1, -1)


def test_centered_strategy_stays_with_seed_component_and_reports_missing_species():
    structure = disconnected_structure()
    instances, _, missing = DisplayPlanner.plan_motif(
        structure, [], MotifConfig(
            max_atoms=4, strategy="centered", seed_site=0, required_species=("Te",)
        )
    )
    assert instances == ((0, (0, 0, 0)),)
    assert missing == ("Te",)


def test_largest_component_keeps_legacy_behavior():
    structure = disconnected_structure()
    bonds = [NeighborBond(2, 3, (0, 0, 0), 0.5)]
    instances, _, missing = DisplayPlanner.plan_motif(
        structure, bonds, MotifConfig(max_atoms=3, strategy="largest_component")
    )
    assert {index for index, _ in instances} == {2, 3}
    assert missing == ("Ce", "Te")


def test_real_crystal_covers_present_elements_within_motif_budget():
    structure = load_structure(
        Path(__file__).parents[1] / "examples/data/visual_regression/Ce6Cd23Te.cif"
    )
    bonds = CrystalNNStrategy().get_bonds(structure)
    instances, species, missing = DisplayPlanner.plan_motif(
        structure, bonds, MotifConfig()
    )

    assert len(instances) <= 12
    assert set(species) == {site.dominant_element for site in structure.sites}
    assert missing == ()
    (te, te_image), = [
        instance for instance in instances if structure.sites[instance[0]].dominant_element == "Te"
    ]
    shell = CoordinationAnalyzer.adjacency(structure, bonds)[te]
    assert len(shell) == 6
    assert all(
        (index, tuple(origin + offset for origin, offset in zip(te_image, image, strict=True))) in instances
        for index, image in shell
    )
