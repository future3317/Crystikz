"""Scientific display-planning regressions for publication figures."""

from collections import Counter
from math import prod
from pathlib import Path

from crystalfig.examples.presets import diamond_structure, mos2_structure, perovskite_structure
from crystalfig.figure.builder import CrystalFigure
from crystalfig.io.loader import load_structure
from crystalfig.neighbors.strategies import CrystalNNStrategy

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
        assert len(displayed) <= max(2, len(environments))
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
    figure = CrystalFigure(perovskite_structure()).publication(intent="auto")
    scene = figure.build_scene()
    assert figure._display_plan.intent == "hybrid"
    assert scene.get_group("polyhedra").primitives


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
