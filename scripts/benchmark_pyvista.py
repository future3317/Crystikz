"""Measure the end-to-end PyVista cost for representative supercell sizes."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

# Run from a checkout without requiring crystalfig to be installed first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if os.name != "nt":
    import resource

from crystalfig.export.exporter import RenderOptions
from crystalfig.figure.builder import CrystalFigure
from crystalfig.model.lattice import Lattice
from crystalfig.model.site import Site
from crystalfig.model.structure import CrystalStructure


def peak_rss_bytes() -> int:
    """Return the process peak resident set size on Windows and Linux."""
    if os.name == "nt":
        size_type = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size_type),
                ("WorkingSetSize", size_type),
                ("QuotaPeakPagedPoolUsage", size_type),
                ("QuotaPagedPoolUsage", size_type),
                ("QuotaPeakNonPagedPoolUsage", size_type),
                ("QuotaNonPagedPoolUsage", size_type),
                ("PagefileUsage", size_type),
                ("PeakPagefileUsage", size_type),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if os.uname().sysname == "Darwin" else peak * 1024)


def benchmark(target_sites: int, output: Path) -> dict:
    """Render one periodic copper supercell and report its peak process cost."""
    task_started = time.perf_counter()
    repeats = math.ceil(target_sites ** (1 / 3))
    structure = CrystalStructure(
        lattice=Lattice.cubic(3.6),
        sites=[Site([0.0, 0.0, 0.0], "Cu")],
    )
    figure = (
        CrystalFigure(structure)
        .supercell((repeats, repeats, repeats))
        .style("ballstick_clean")
        .quality("preview")
        .show_unit_cell(False)
        .view([1, 1, 1])
        .add_bonds("cutoff", cutoff=3.7)
    )
    options = RenderOptions(width=25.4, height=25.4, dpi=200, max_pixels=4_000_000)
    started = time.perf_counter()
    scene = figure.build_scene()
    scene_seconds = time.perf_counter() - started
    bond_count = sum(primitive.__class__.__name__ == "Bond" for primitive in scene.all_primitives())

    from crystalfig.renderers.pyvista_renderer import PyVistaRenderer

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    PyVistaRenderer(camera=figure.camera).export(
        scene,
        str(output),
        figure.theme,
        options,
    )
    render_seconds = time.perf_counter() - started
    return {
        "target_sites": target_sites,
        "actual_sites": scene.metadata["num_sites"],
        "replication": [repeats] * 3,
        "bonds": bond_count,
        "scene_seconds": round(scene_seconds, 3),
        "render_seconds": round(render_seconds, 3),
        "total_seconds": round(time.perf_counter() - task_started, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "png_pixels": [200, 200],
        "png_size_bytes": output.stat().st_size,
        "benchmark_bond_model": "synthetic 3.7 Angstrom cutoff for simple cubic Cu",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-sites", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="crystalfig-benchmark-") as temp:
        result = benchmark(args.target_sites, Path(temp) / "benchmark.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
