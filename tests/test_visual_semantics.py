"""Scientific display-planning regressions for publication figures."""

from collections import Counter
from math import prod
from pathlib import Path

import numpy as np

from crystalfig.analysis.coordination import CoordinationAnalyzer
from crystalfig.examples.presets import diamond_structure, mos2_structure, perovskite_structure
from crystalfig.figure.builder import CrystalFigure
from crystalfig.figure.config import CameraConfig, FigureConfig, RenderConfig
from crystalfig.io.loader import load_structure
from crystalfig.neighbors.base import NeighborBond
from crystalfig.neighbors.strategies import CrystalNNStrategy
from crystalfig.planning.display import DisplayConfig, DisplayPlanner

DATA = Path(__file__).parent.parent / "examples" / "data" / "visual_regression"


def _polyhedron_environments(figure, scene):
    display = figure.structure.make_supercell(figure._display_plan.repeat)
    group = scene.get_group("polyhedra")
    return Counter(
        (display.sites[poly.center_site].dominant_element, len(poly.vertex_metadata))
        for poly in (group.primitives if group else [])
    )


def test_auto_planner_covers_representative_local_environments():
    cases = [
        ("li_fepo4_mp_workshop.cif", {("Fe", 6), ("P", 4)}),
        ("mg_al2o4_normal_spinel.cif", {("Mg", 4), ("Al", 6)}),
        ("alpha_quartz_cod5000035.cif", {("Si", 4)}),
    ]
    for filename, required in cases:
        figure = CrystalFigure(load_structure(DATA / filename)).publication(intent="auto")
        scene = figure.build_scene()
        environments = _polyhedron_environments(figure, scene)
        assert required <= environments.keys()
        displayed = scene.get_group("polyhedra").primitives
        required_colors = {
            key: next(
                poly.fill_color
                for poly in displayed
                if (
                    figure.structure.make_supercell(figure._display_plan.repeat)
                    .sites[poly.center_site].dominant_element,
                    len(poly.vertex_metadata),
                ) == key
            )
            for key in required
        }
        assert len(set(required_colors.values())) == len(required)
        assert len(displayed) >= len(required)
        assert len(displayed) <= figure._display_budget["max_polyhedra"]
        for poly in displayed:
            identities = [
                (item["site_index"], tuple(item["image_offset"]))
                for item in poly.vertex_metadata
            ]
            assert len(identities) == len(set(identities))
            assert all(index >= 0 for index, _ in identities)


def test_publication_display_plans_are_deterministic_and_budgeted():
    structure = load_structure(DATA / "li_fepo4_mp_workshop.cif")
    first = CrystalFigure(structure).publication(intent="auto")
    second = CrystalFigure(structure).publication(intent="auto")
    scene_a = first.build_scene()
    scene_b = second.build_scene()
    assert first._display_plan == second._display_plan
    assert scene_a.metadata["camera_used"] == scene_b.metadata["camera_used"]
    assert structure.num_sites * prod(first._display_plan.repeat) <= first._display_budget["max_atoms"]
    assert scene_a.metadata["num_real_polyhedra"] <= first._display_budget["max_polyhedra"]


def test_layered_candidates_include_side_and_oblique_views():
    figure = CrystalFigure(mos2_structure()).publication(intent="layered")
    scene = figure.build_scene()
    candidates = figure.report_view_candidates(scene)
    labels = {item["label"].removesuffix("_roll90") for item in candidates}
    assert {"layered_side", "layered_oblique"} <= labels
    assert all(0 <= item["layer_separation"] <= 1 for item in candidates)


def test_mos2_neighbor_graph_does_not_bridge_adjacent_trilayers():
    structure = mos2_structure()
    figure = CrystalFigure(structure).publication(intent="layered")
    bonds = CrystalNNStrategy().get_bonds(structure)
    normal = figure._periodic_layer_normal()
    assert normal is not None
    assert figure._periodic_connectivity_dimension(bonds) == 2
    out_of_plane = [
        abs(
            float(
                structure.lattice.frac_to_cart(
                    structure.frac_coords[bond.j]
                    + bond.jimage
                    - structure.frac_coords[bond.i]
                ) @ normal
            )
        )
        for bond in bonds
    ]
    assert max(out_of_plane) < 2.0


def test_automatic_hybrid_view_preserves_network_bonds_and_coordination():
    figure = CrystalFigure(perovskite_structure()).configure(FigureConfig(style="journal_vivid"))
    scene = figure.build_scene()
    assert figure._display_plan.intent == "hybrid"
    assert scene.get_group("polyhedra").primitives
    assert scene.get_group("bonds").primitives
    score = scene.metadata["camera_score_components"]
    assert score["center_ligand_collapse_fraction"] < 0.1


def test_coordination_crop_hides_distant_cell_frame_unless_requested():
    structure = load_structure(DATA / "mg_al2o4_normal_spinel.cif")
    automatic = CrystalFigure(structure).publication(intent="auto").build_scene()
    explicit_cell = CrystalFigure(structure).publication(
        intent="auto", show_unit_cell=True
    ).build_scene()
    assert automatic.get_group("cell") is None
    assert explicit_cell.get_group("cell").primitives


def test_silicon_camera_avoids_projected_bond_crossings():
    figure = CrystalFigure(diamond_structure(conventional=True)).publication(intent="auto")
    scene = figure.build_scene()
    assert scene.metadata["camera_score_components"]["projected_bond_crossing_fraction"] < 0.1
    assert 0 <= scene.metadata["camera_score_components"]["atom_visibility_fraction"] <= 1


def test_configured_coordination_plan_keeps_connected_bonds_and_camera_settings():
    figure = CrystalFigure(perovskite_structure()).configure(FigureConfig(
        display=DisplayConfig(intent="coordination", region="cell"),
        camera=CameraConfig(mode="fixed", elevation=31, azimuth=47, roll=12),
        render=RenderConfig(backend="matplotlib", quality="preview"),
    ))
    scene = figure.build_scene()
    assert scene.metadata["selected_intent"] == "coordination"
    assert scene.get_group("bonds").primitives
    assert figure.camera.elevation == 31
    assert figure.camera.azimuth == 47
    assert figure.camera.roll == 12
    assert figure.theme.render_quality == "preview"
    assert scene.metadata["coordination_shell_policy"] == "complete"
    assert scene.metadata["num_real_polyhedra"] > 0


def test_configured_network_motif_expands_beyond_unit_cell():
    config = FigureConfig(
        display=DisplayConfig(intent="network", region="motif"),
        style="vesta_ballstick",
        render=RenderConfig(backend="matplotlib", quality="preview"),
    )
    motif = CrystalFigure(perovskite_structure()).configure(config)
    cell_config = FigureConfig(
        display=DisplayConfig(intent="network", region="cell"),
        style="vesta_ballstick",
        render=RenderConfig(backend="matplotlib", quality="preview"),
    )
    cell = CrystalFigure(perovskite_structure()).configure(cell_config)
    assert motif._display_plan.repeat != (1, 1, 1)
    assert cell._display_plan.repeat == (1, 1, 1)


def test_explicit_framing_choices_on_dense_cell_fixture():
    structure = load_structure(DATA / "Ce6Cd23Te.cif")
    unit_cell = CrystalFigure(structure).publication(
        framing="unit_cell", style="ballstick", intent="network"
    )
    motif = CrystalFigure(structure).publication(
        framing="motif", style="ballstick", intent="network"
    )
    local = CrystalFigure(structure).publication(
        framing="local_coordination", framing_site=0,
        style="nature", strategy="crystalnn",
    )
    local_without_hull = CrystalFigure(structure).publication(
        framing="local_coordination", framing_site=0,
        style="ballstick", coordination_shell="none", strategy="crystalnn",
    ).add_bonds(strategy="crystalnn")
    automatic = CrystalFigure(structure).publication(style="vesta_ballstick", intent="auto")
    assert unit_cell._display_plan.repeat == (1, 1, 1)
    assert motif._display_plan.repeat == DisplayPlanner.publication_window(
        structure, "network", "motif", max_atoms=60
    )
    assert local._display_plan.repeat == (1, 1, 1)
    assert local._display_plan.selected_centers == (0,)
    assert local._display_plan.intent == "coordination"
    assert automatic._display_plan.framing == "motif"
    unit_scene = unit_cell.build_scene()
    motif_scene = motif.build_scene()
    local_scene = local.build_scene()
    local_without_hull_scene = local_without_hull.build_scene()
    automatic_scene = automatic.build_scene()
    assert len(motif_scene.get_group("atoms").primitives) < len(unit_scene.get_group("atoms").primitives)
    motif_atoms = motif_scene.get_group("atoms").primitives
    motif_positions = [sphere.position for sphere in motif_atoms]
    motif_edges = []
    for bond in motif_scene.get_group("bonds").primitives:
        first_distances = [np.linalg.norm(bond.start - position) for position in motif_positions]
        second_distances = [np.linalg.norm(bond.end - position) for position in motif_positions]
        first, second = int(np.argmin(first_distances)), int(np.argmin(second_distances))
        assert first_distances[first] < 1e-7
        assert second_distances[second] < 1e-7
        motif_edges.append((first, second))
    reachable = {0}
    while True:
        expanded = reachable | {
            second if first in reachable else first
            for first, second in motif_edges
            if first in reachable or second in reachable
        }
        if expanded == reachable:
            break
        reachable = expanded
    assert len(reachable) == len(motif_atoms)
    coordination_cell = CrystalFigure(structure).publication(
        framing="unit_cell", style="vesta_coordination", intent="coordination"
    ).build_scene()
    hidden_cell = CrystalFigure(structure).publication(
        framing="unit_cell", style="vesta_coordination", intent="coordination",
        show_unit_cell=False,
    ).build_scene()
    assert coordination_cell.get_group("cell") is not None
    assert hidden_cell.get_group("cell") is None
    assert len(coordination_cell.get_group("atoms").primitives) >= structure.num_sites
    assert len(local_scene.get_group("atoms").primitives) > 1
    assert local_scene.get_group("bonds").primitives
    assert local_without_hull_scene.get_group("atoms").primitives
    assert local_without_hull_scene.get_group("bonds").primitives
    assert local_without_hull_scene.get_group("polyhedra") is None
    assert len(automatic_scene.get_group("atoms").primitives) < len(unit_scene.get_group("atoms").primitives)
    shell = CoordinationAnalyzer().analyze(
        structure, [0], CrystalNNStrategy()
    )[0]
    rendered = {
        (atom.metadata["site_index"], tuple(atom.metadata["image_offset"]))
        for atom in local_scene.get_group("atoms").primitives
    }
    expected = {(0, (0, 0, 0))} | set(zip(
        shell.ligand_indices, shell.ligand_image_offsets, strict=True
    ))
    assert expected <= rendered
    positions = [atom.position for atom in local_scene.get_group("atoms").primitives]
    for bond in local_scene.get_group("bonds").primitives:
        assert min(np.linalg.norm(bond.start - position) for position in positions) < 1e-7
        assert min(np.linalg.norm(bond.end - position) for position in positions) < 1e-7
    assert 0 <= unit_scene.metadata["camera_score_components"]["projection_crowding_fraction"] <= 1


def test_periodic_self_bond_produces_both_local_ligand_images():
    shell = CoordinationAnalyzer().analyze(
        perovskite_structure(), [0],
        [
            NeighborBond(0, 0, (1, 0, 0), 1.0),
            NeighborBond(0, 0, (-1, 0, 0), 1.0),
        ],
    )[0]
    assert shell.coordination_number == 2
    assert set(shell.ligand_image_offsets) == {(-1, 0, 0), (1, 0, 0)}
