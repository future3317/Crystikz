"""Build and smoke-test the wheel in a temporary, clean virtual environment."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    run([sys.executable, "-m", "build", "--no-isolation", "--wheel"])
    wheel = next((ROOT / "dist").glob("crystalfig-0.2.0-*.whl"), None)
    if wheel is None:
        raise SystemExit("The crystalfig 0.2.0 wheel was not produced.")

    with tempfile.TemporaryDirectory(prefix="crystalfig-wheel-") as temp:
        temp_path = Path(temp)
        env_path = temp_path / "venv"
        venv.EnvBuilder(with_pip=True).create(env_path)
        scripts = env_path / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        cli = scripts / ("crystalfig.exe" if os.name == "nt" else "crystalfig")
        run([str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)])

        smoke = temp_path / "smoke.py"
        smoke.write_text(
            """import sys
from pathlib import Path
from crystalfig import CrystalFigure, __version__
from crystalfig.examples.presets import rocksalt_structure
from crystalfig.io.pymatgen_adapter import to_pymatgen

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
structure = rocksalt_structure(conventional=True)
to_pymatgen(structure).to(filename=str(out / 'input.cif'))
figure = CrystalFigure.from_structure(structure).publication()
figure.export(out / 'figure.pdf', width=89, dpi=300)
figure.export(out / 'figure.svg', backend='svg', width=89)
figure.export(out / 'figure.png', backend='matplotlib', width=89, dpi=300, transparent=True)
assert __version__ == '0.2.0'
""",
            encoding="utf-8",
        )
        output = temp_path / "outputs"
        run([str(python), str(smoke), str(output)])
        run([str(cli), "--help"])
        run([
            str(cli), "render", str(output / "input.cif"), "-o", str(output / "cli.svg"),
            "--backend", "svg", "--bonds", "crystalnn", "--width", "89",
        ])

        preflight = temp_path / "preflight.py"
        preflight.write_text(
            """import sys
from crystalfig.export.preflight import preflight_export

root = sys.argv[1]
checks = [
    ('figure.pdf', 'pure', {'expected_width_mm': 89}),
    ('figure.svg', 'pure', {'expected_width_mm': 89}),
    ('figure.png', 'raster', {'expected_width_mm': 89, 'expected_dpi': 300, 'transparent': True}),
    ('cli.svg', 'pure', {'expected_width_mm': 89}),
]
for filename, status, expected in checks:
    report = preflight_export(f'{root}/{filename}', vector_status=status, **expected)
    assert report['exists'], report
    assert not report['warnings'], report
""",
            encoding="utf-8",
        )
        run([str(python), str(preflight), str(output)])
    print(f"Wheel smoke passed: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
