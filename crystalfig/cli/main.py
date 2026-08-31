"""crystalfig CLI entry point."""

from __future__ import annotations

import argparse
import shutil
import sys

import yaml

import crystalfig
from crystalfig.export.preflight import preflight_pdf
from crystalfig.figure.builder import plot_structure
from crystalfig.io.loader import load_structure


def _add_render_args(parser: argparse.ArgumentParser):
    parser.add_argument("input", help="Input structure file (CIF, POSCAR, XYZ, ...)")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--fmt", default=None, help="Output format (pdf, svg, png, tex, ...)")
    parser.add_argument("--view", nargs=3, type=float, default=None, help="View direction e.g. 1 1 0")
    parser.add_argument("--supercell", nargs=3, type=int, default=None, help="Supercell e.g. 2 2 1")
    parser.add_argument("--bonds", default=None, help="Bond strategy: crystalnn, cutoff, covalent, ase")
    parser.add_argument("--polyhedra", default=None, help="Polyhedra center species, e.g. Ti")
    parser.add_argument("--style", default="publication_muted", help="Style preset")
    parser.add_argument("--transparent", action="store_true", help="Transparent background")
    parser.add_argument("--width", type=float, default=None, help="Figure width in mm")
    parser.add_argument("--latex", action="store_true", help="Compile TikZ via LaTeX to PDF")


def cmd_render(args):
    fig = plot_structure(args.input).style(args.style).show_unit_cell().show_axes()
    if args.supercell:
        fig.supercell(tuple(args.supercell))
    if args.view:
        fig.view(list(args.view))
    if args.bonds:
        fig.add_bonds(strategy=args.bonds)
    if args.polyhedra:
        fig.add_polyhedra(centers=args.polyhedra, strategy=args.bonds or "covalent")
    if args.latex:
        result = fig.export_tikz_pdf(args.output)
    else:
        result = fig.export(args.output, fmt=args.fmt, width=args.width, transparent=args.transparent)
    print(f"Exported: {result.path} ({result.format}, {result.vector_status})")
    return 0


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
    optional = [("ase", "ase"), ("seekpath", "reciprocal")]
    for dep, extra in optional:
        try:
            __import__(dep)
            print(f"  {dep} (optional): installed")
        except ImportError:
            print(f"  {dep} (optional): NOT INSTALLED (pip install crystalfig[{extra}])")
    for exe in ["pdflatex", "xelatex", "lualatex", "latexmk", "dvisvgm", "pdftocairo"]:
        print(f"  {exe}: {'found' if shutil.which(exe) else 'not found'}")
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
    with open(args.input, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Simple batch runner
    figures = config.get("figures", [config])
    for fig_cfg in figures:
        input_file = fig_cfg["input"]
        output = fig_cfg["output"]
        style = fig_cfg.get("style", "publication_muted")
        view = fig_cfg.get("view")
        supercell = fig_cfg.get("supercell")
        bonds = fig_cfg.get("bonds")
        polyhedra = fig_cfg.get("polyhedra")
        fig = plot_structure(input_file).style(style).show_unit_cell().show_axes()
        if view:
            fig.view(view)
        if supercell:
            fig.supercell(tuple(supercell))
        if bonds:
            fig.add_bonds(strategy=bonds)
        if polyhedra:
            fig.add_polyhedra(centers=polyhedra)
        fig.export(output)
        print(f"Batch exported: {output}")
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
