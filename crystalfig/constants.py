"""Physical and numerical constants used throughout crystalfig."""


# Numerical tolerances
DEFAULT_TOLERANCE = 1e-5
LATTICE_ANGLE_TOLERANCE = 1e-6
WRAP_TOLERANCE = 1e-8

# Default visual parameters
DEFAULT_ATOM_RADIUS = 0.25
DEFAULT_BOND_RADIUS = 0.08
DEFAULT_LINE_WIDTH = 1.0

# Coordinate conventions
# Lattice matrix stores column vectors: cart = lattice_matrix @ frac
# This matches pymatgen convention (Lattice.matrix rows are a, b, c cartesian).
