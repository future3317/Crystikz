"""Load structures from files and strings."""

from __future__ import annotations

from pathlib import Path

from crystalfig.exceptions import StructureParseError
from crystalfig.io.pymatgen_adapter import from_pymatgen
from crystalfig.model.structure import CrystalStructure


def guess_format(path: str | Path) -> str:
    """Guess file format from extension."""
    ext = Path(path).suffix.lower()
    mapping = {
        ".cif": "cif",
        ".poscar": "poscar",
        ".vasp": "poscar",
        ".cssr": "cssr",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".extxyz": "extxyz",
        ".xyz": "xyz",
    }
    return mapping.get(ext, "unknown")


def load_structure(path: str | Path, fmt: str | None = None) -> CrystalStructure:
    """Load a crystal structure from a file.

    Supports CIF, POSCAR, CSSR, JSON, YAML, and (if ASE is installed) XYZ/extxyz.
    """
    path = Path(path)
    if not path.exists():
        raise StructureParseError(f"File not found: {path}")

    fmt = fmt or guess_format(path)

    if fmt == "cif":
        from pymatgen.io.cif import CifParser
        parser = CifParser(str(path))
        structures = parser.parse_structures(primitive=False)
        if not structures:
            raise StructureParseError(f"No structures found in {path}")
        return from_pymatgen(structures[0])

    if fmt in ("poscar", "vasp"):
        from pymatgen.io.vasp import Poscar
        poscar = Poscar.from_file(str(path))
        return from_pymatgen(poscar.structure)

    if fmt == "cssr":
        from pymatgen.io.cssr import Cssr
        cssr = Cssr.from_file(str(path))
        return from_pymatgen(cssr.structure)

    if fmt == "json":
        from pymatgen.core import Structure
        pmg = Structure.from_file(str(path))
        return from_pymatgen(pmg)

    if fmt in ("yaml", "yml"):
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        from pymatgen.core import Structure
        pmg = Structure.from_dict(data)
        return from_pymatgen(pmg)

    # Fallback to ASE if available
    try:
        from ase.io import read

        from crystalfig.io.ase_adapter import from_ase
        atoms = read(str(path), format=fmt if fmt != "unknown" else None)
        if atoms is None:
            raise StructureParseError(f"ASE could not parse {path}")
        return from_ase(atoms)
    except ImportError as exc:
        raise StructureParseError(
            f"Format '{fmt}' requires ASE which is not installed. "
            "Install with: pip install crystalfig[ase]"
        ) from exc
    except Exception as exc:
        raise StructureParseError(f"Could not parse {path} as {fmt}: {exc}") from exc
