"""Tests for CLI commands."""

import importlib
import tempfile
from pathlib import Path

from crystalfig.cli.main import _configure_figure, main
from crystalfig.examples.presets import rocksalt_structure
from crystalfig.io.pymatgen_adapter import to_pymatgen

cli_module = importlib.import_module("crystalfig.cli.main")


class TestCLI:
    def test_nested_configuration_uses_figure_config(self, tmp_path):
        cif_path = tmp_path / "input.cif"
        to_pymatgen(rocksalt_structure()).to(filename=str(cif_path))
        figure = _configure_figure({
            "input": str(cif_path),
            "display": {"intent": "coordination", "region": "cell", "centers": ["Na"]},
            "camera": {"mode": "fixed", "elevation": 15, "azimuth": 30},
            "render": {"backend": "matplotlib", "quality": "preview", "dpi": 144},
        }, tmp_path)
        assert figure.config.display.intent == "coordination"
        assert figure.config.display.centers == ("Na",)
        assert figure.config.render.backend == "matplotlib"
        assert figure.camera.elevation == 15
        style_only = _configure_figure({"input": str(cif_path), "style": "vesta_ballstick"}, tmp_path)
        assert style_only.config.style == "vesta_ballstick"

    def test_nested_region_repeat_list_is_normalized(self, tmp_path):
        cif_path = tmp_path / "input.cif"
        to_pymatgen(rocksalt_structure()).to(filename=str(cif_path))
        figure = _configure_figure({
            "input": str(cif_path), "display": {"region": [2, 1, 1]},
        }, tmp_path)
        assert figure.config.display.region == (2, 1, 1)

    def test_nested_config_preserves_legacy_bond_and_polyhedron_options(self, tmp_path):
        cif_path = tmp_path / "input.cif"
        to_pymatgen(rocksalt_structure()).to(filename=str(cif_path))
        figure = _configure_figure({
            "input": str(cif_path),
            "style": "vesta_coordination",
            "display": {"intent": "coordination"},
            "bonds": {"strategy": "cutoff", "cutoff": 3.1},
            "polyhedra": {"centers": ["Na"], "strategy": "cutoff", "cutoff": 3.4},
        }, tmp_path)
        assert figure._scene_options.bond_strategy.cutoff == 3.1
        assert figure._scene_options.polyhedra_specs
        assert figure._scene_options.polyhedra_specs[0].strategy.cutoff == 3.4

    def test_nested_yaml_and_cli_flags_build_the_same_config(self, monkeypatch, tmp_path):
        cif_path = tmp_path / "input.cif"
        to_pymatgen(rocksalt_structure()).to(filename=str(cif_path))
        yaml_settings = {
            "input": str(cif_path),
            "display": {
                "framing": "local_coordination", "framing_site": 0,
                "bond_display_policy": "polyhedron_edges_only", "show_unit_cell": False,
            },
            "style": "journal_clean",
            "camera": {"mode": "auto"},
            "render": {"backend": "matplotlib", "quality": "preview", "dpi": 144},
            "show_axes": True,
        }
        yaml_figure = _configure_figure(yaml_settings, tmp_path)

        def compare(args):
            cli_figure = _configure_figure(vars(args), tmp_path)
            assert cli_figure.config == yaml_figure.config
            assert cli_figure._scene_options.show_axes is yaml_figure._scene_options.show_axes
            scene = cli_figure.build_scene()
            assert scene.metadata["framing"] == "local_coordination"
            assert scene.get_group("bonds") is None
            return 0

        monkeypatch.setattr(cli_module, "cmd_render", compare)
        assert main([
            "render", str(cif_path), "-o", str(tmp_path / "out.svg"),
            "--framing", "local_coordination", "--framing-site", "0",
            "--bond-display-policy", "polyhedron_edges_only", "--style", "journal_clean",
            "--backend", "matplotlib", "--quality", "preview", "--dpi", "144",
            "--no-cell", "--show-axes", "--view-preset", "auto",
        ]) == 0

    def test_doctor(self, capsys):
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "crystalfig doctor" in out

    def test_inspect(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "rocksalt.cif"
            pmg = to_pymatgen(rocksalt_structure())
            pmg.to(filename=str(cif_path))
            assert main(["inspect", str(cif_path)]) == 0
            out = capsys.readouterr().out
            assert "Formula" in out
            assert "angstrom^3" in out
            assert "alpha=" in out

    def test_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "rocksalt.cif"
            out_path = Path(tmpdir) / "out.svg"
            pmg = to_pymatgen(rocksalt_structure())
            pmg.to(filename=str(cif_path))
            assert main(["render", str(cif_path), "-o", str(out_path), "--fmt", "svg", "--bonds", "covalent"]) == 0
            assert out_path.exists()

    def test_render_flags_select_shared_figure_settings(self, tmp_path):
        cif_path = tmp_path / "input.cif"
        to_pymatgen(rocksalt_structure()).to(filename=str(cif_path))

        assert main([
            "render", str(cif_path), "-o", str(tmp_path / "render.svg"),
            "--backend", "svg", "--dpi", "144", "--height", "50",
            "--view-preset", "auto", "--no-axes", "--no-cell", "--quality", "preview",
        ]) == 0
        assert (tmp_path / "render.svg").is_file()

    def test_batch_resolves_paths_from_yaml_and_applies_shared_options(self, tmp_path):
        input_dir = tmp_path / "inputs"
        config_dir = tmp_path / "jobs"
        input_dir.mkdir()
        config_dir.mkdir()
        to_pymatgen(rocksalt_structure()).to(filename=str(input_dir / "input.cif"))
        config = config_dir / "batch.yaml"
        config.write_text(
            "figures:\n"
            "  - input: ../inputs/input.cif\n"
            "    output: output/figure.svg\n"
            "    style: ballstick_clean\n"
            "    backend: svg\n"
            "    dpi: 144\n"
            "    width: 89\n"
            "    height: 50\n"
            "    quality: preview\n"
            "    view_preset: auto\n"
            "    show_axes: false\n"
            "    show_cell: false\n"
            "    bonds:\n"
            "      strategy: cutoff\n"
            "      cutoff: 3.0\n",
            encoding="utf-8",
        )

        figure_settings = {
            "input": "../inputs/input.cif",
            "show_axes": False,
            "show_cell": False,
            "quality": "preview",
            "view_preset": "auto",
            "bonds": {"strategy": "cutoff", "cutoff": 3.0},
        }
        figure = _configure_figure(figure_settings, config_dir)
        scene = figure.build_scene()
        assert figure.theme.render_quality == "preview"
        assert not any(primitive.__class__.__name__ == "CellEdge" for primitive in scene.all_primitives())

        assert main(["batch", str(config)]) == 0
        output = config_dir / "output" / "figure.svg"
        assert output.is_file()
        assert 'width="89.000mm"' in output.read_text(encoding="utf-8")
        assert 'height="50.000mm"' in output.read_text(encoding="utf-8")
