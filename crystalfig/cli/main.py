"""crystalfig CLI entry point."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

import crystalfig
from crystalfig.export.preflight import preflight_pdf
from crystalfig.figure.builder import plot_structure
from crystalfig.io.loader import load_structure


def _add_render_args(parser: argparse.ArgumentParser):
    parser.add_argument("input", help="Input structure file (CIF, POSCAR, XYZ, ...)")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--fmt", default=None, help="Output format (pdf, svg, png, tex, ...)")
    parser.add_argument("--supercell", nargs=3, type=int, default=None, help="Supercell e.g. 2 2 1")
    parser.add_argument("--bonds", default=None, help="Bond strategy: crystalnn, cutoff, covalent, ase")
    parser.add_argument("--polyhedra", default=None, help="Polyhedra center species, e.g. Ti")
    parser.add_argument("--style", default="ballstick_clean", help="Style preset")
    parser.add_argument("--backend", default=None, help="Renderer backend (matplotlib, svg, tikz, pyvista)")
    parser.add_argument("--dpi", type=int, default=None, help="Raster resolution")
    parser.add_argument("--height", type=float, default=None, help="Figure height in mm; default follows projected structure")
    parser.add_argument("--quality", choices=("preview", "standard", "publication"), default="standard")
    views = parser.add_mutually_exclusive_group()
    views.add_argument("--view", nargs=3, type=float, default=None, help="View direction e.g. 1 1 0")
    views.add_argument("--view-preset", choices=("auto", "isometric", "along_a", "along_b", "along_c", "layer_top", "layer_oblique", "polyhedron_focus"))
    axes = parser.add_mutually_exclusive_group()
    axes.add_argument("--show-axes", dest="show_axes", action="store_true")
    axes.add_argument("--no-axes", dest="show_axes", action="store_false")
    parser.set_defaults(show_axes=False)
    cell = parser.add_mutually_exclusive_group()
    cell.add_argument("--show-cell", dest="show_cell", action="store_true")
    cell.add_argument("--no-cell", dest="show_cell", action="store_false")
    parser.set_defaults(show_cell=True)
    parser.add_argument("--transparent", action="store_true", help="Transparent background")
    parser.add_argument("--width", type=float, default=None, help="Figure width in mm")
    parser.add_argument("--latex", action="store_true", help="Compile TikZ via LaTeX to PDF")


def cmd_render(args):
    config = vars(args)
    fig = _configure_figure(config, Path.cwd())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.latex:
        result = fig.export_tikz_pdf(args.output)
    else:
        result = fig.export(
            args.output, fmt=args.fmt, width=args.width, height=args.height,
            dpi=args.dpi, transparent=args.transparent, backend=args.backend,
        )
    print(f"Exported: {result.path} ({result.format}, {result.vector_status})")
    for diagnostic in fig.diagnostics:
        print(f"Diagnostic: {diagnostic}")
    return 0


def _configure_figure(config: dict, base_dir: Path):
    input_path = Path(config["input"])
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    style = config.get("style") or "ballstick_clean"
    figure = plot_structure(input_path).style(style)
    figure.quality(config.get("quality") or "standard")
    figure.show_unit_cell(config.get("show_cell", True))
    figure.show_axes(config.get("show_axes", False))
    if config.get("supercell"):
        figure.supercell(tuple(config["supercell"]))
    if config.get("view_preset"):
        figure.view_preset(config["view_preset"])
    elif config.get("view"):
        figure.view(config["view"])
    if config.get("bonds"):
        bonds = config["bonds"]
        if isinstance(bonds, dict):
            figure.add_bonds(
                strategy=bonds.get("strategy", "crystalnn"),
                cutoff=bonds.get("cutoff", 2.5),
                pair_cutoffs=bonds.get("pair_cutoffs"),
            )
        else:
            figure.add_bonds(strategy=bonds)
    if config.get("polyhedra"):
        polyhedra = config["polyhedra"]
        if isinstance(polyhedra, dict):
            centers = polyhedra["centers"]
            strategy = polyhedra.get("strategy", "crystalnn")
        else:
            centers = polyhedra
            strategy = config.get("bonds") or "crystalnn"
        figure.add_polyhedra(centers=centers, strategy=strategy)
    return figure


def cmd_inspect(args):
    structure = load_structure(args.input)
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        pmg = crystalfig.io.pymatgen_adapter.to_pymatgen(structure)
        sga = SpacegroupAnalyzer(pmg)
        spg = sga.get_space_group_symbol()
        number = sga.get_space_group_number()
        system = sga.get_crystal_system()
    except Exception:
        spg = "unknown"
        number = "unknown"
        system = "unknown"

    print(f"Formula:   {structure.formula}")
    print(f"Sites:     {structure.num_sites}")
    print(f"Volume:    {structure.volume:.3f} Å³")
    print(f"Lattice:   a={structure.lattice.lengths[0]:.4f}, b={structure.lattice.lengths[1]:.4f}, c={structure.lattice.lengths[2]:.4f}")
    print(f"Angles:    α={structure.lattice.angles[0]:.2f}, β={structure.lattice.angles[1]:.2f}, γ={structure.lattice.angles[2]:.2f}")
    print(f"Spacegroup: {spg} ({number})")
    print(f"System:    {system}")
    print(f"Species:   {', '.join(structure.unique_species())}")
    return 0


def cmd_doctor(_args):
    print("crystalfig doctor")
    print(f"  package version: {crystalfig.__version__}")
    print(f"  python: {sys.version.split()[0]}")
    deps = ["numpy", "scipy", "matplotlib", "pymatgen", "spglib", "yaml"]
    for dep in deps:
        try:
            mod = __import__(dep)
            print(f"  {dep}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"  {dep}: NOT INSTALLED")
    optional = [("ase", "ase"), ("seekpath", "reciprocal"), ("pyvista", "render3d")]
    for dep, extra in optional:
        try:
            __import__(dep)
            print(f"  {dep} (optional): installed")
        except ImportError:
            print(f"  {dep} (optional): NOT INSTALLED (pip install crystalfig[{extra}])")
    for exe in ["pdflatex", "xelatex", "lualatex", "latexmk", "dvisvgm", "pdftocairo"]:
        print(f"  {exe}: {'found' if shutil.which(exe) else 'not found'}")
    if importlib.util.find_spec("pyvista") is not None:
        probe = (
            "import pyvista as pv; p=pv.Plotter(off_screen=True, window_size=(64,64)); "
            "p.add_mesh(pv.Sphere()); p.render(); "
            "print(p.ren_win.ReportCapabilities()); p.close()"
        )
        with tempfile.TemporaryDirectory(prefix="crystalfig-doctor-"):
            try:
                result = subprocess.run(
                    [sys.executable, "-c", probe], capture_output=True, text=True, timeout=20
                )
                print(f"  PyVista off-screen probe: {'available' if result.returncode == 0 else f'failed ({result.returncode})'}")
                if result.returncode == 0:
                    print("  OpenGL: " + next((line.strip() for line in result.stdout.splitlines() if "OpenGL" in line), "capabilities reported by VTK"))
                elif result.stderr:
                    print("  PyVista probe detail: " + result.stderr.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                print("  PyVista off-screen probe: timed out after 20 seconds")
    return 0


def cmd_preflight(args):
    report = preflight_pdf(args.input)
    print(f"File:      {report.path}")
    print(f"Exists:    {report.exists}")
    print(f"Page size: {report.page_size}")
    print(f"Fonts:     {report.fonts}")
    print(f"Type3:     {report.has_type3_fonts}")
    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    return 0


def cmd_batch(args):
    config_path = Path(args.input).resolve()
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    figures = config.get("figures", [config])
    for fig_cfg in figures:
        if "output" not in fig_cfg:
            raise ValueError("Each batch figure requires an output path.")
        figure = _configure_figure(fig_cfg, config_path.parent)
        output = Path(fig_cfg["output"])
        if not output.is_absolute():
            output = config_path.parent / output
        output.parent.mkdir(parents=True, exist_ok=True)
        result = figure.export(
            output, fmt=fig_cfg.get("fmt"), width=fig_cfg.get("width"),
            height=fig_cfg.get("height"), dpi=fig_cfg.get("dpi"),
            transparent=fig_cfg.get("transparent"), backend=fig_cfg.get("backend"),
        )
        print(f"Batch exported: {result.path} ({result.format}, {result.vector_status})")
        for diagnostic in figure.diagnostics:
            print(f"Diagnostic: {diagnostic}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crystalfig", description="Crystal structure figure toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_render = subparsers.add_parser("render", help="Render a structure to a figure")
    _add_render_args(p_render)

    p_inspect = subparsers.add_parser("inspect", help="Inspect a structure file")
    p_inspect.add_argument("input", help="Input structure file")

    subparsers.add_parser("doctor", help="Check environment and dependencies")

    p_preflight = subparsers.add_parser("preflight", help="Check an exported PDF")
    p_preflight.add_argument("input", help="PDF file to check")

    p_batch = subparsers.add_parser("batch", help="Run batch configuration")
    p_batch.add_argument("input", help="YAML batch configuration")

    args = parser.parse_args(argv)
    commands = {
        "render": cmd_render,
        "inspect": cmd_inspect,
        "doctor": cmd_doctor,
        "preflight": cmd_preflight,
        "batch": cmd_batch,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
