"""Coordination polyhedron construction using ConvexHull."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from crystalfig.exceptions import GeometryError


@dataclass
class CoordinationPolyhedron:
    """A coordination polyhedron around a central site."""

    center_index: int
    center_position: np.ndarray
    vertex_indices: list[int]
    vertex_positions: list[np.ndarray]
    faces: list[list[int]]
    volume: float = 0.0


def _triangulate_face(face: list[int]) -> list[list[int]]:
    """Triangulate a polygonal face using a fan from the first vertex."""
    if len(face) <= 3:
        return [face]
    return [[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)]


def build_polyhedron(
    center: np.ndarray,
    vertices: list[np.ndarray],
    vertex_indices: list[int] | None = None,
    center_index: int = -1,
) -> CoordinationPolyhedron:
    """Build a coordination polyhedron from center and vertex positions.

    Uses scipy.spatial.ConvexHull to obtain faces.  Faces are triangulated
    and oriented to point outward from the center.
    """
    vertices = [np.asarray(v, dtype=float) for v in vertices]
    if len(vertices) < 4:
        raise GeometryError("Need at least 4 vertices to build a polyhedron.")

    points = np.array([center] + vertices)
    try:
        hull = ConvexHull(points)
    except Exception as exc:
        raise GeometryError(f"ConvexHull failed: {exc}") from exc

    # Collect faces, skipping those that include the center vertex (index 0)
    raw_faces = []
    for simp in hull.simplices:
        if 0 in simp:
            continue
        # Shift indices by -1 to account for center at index 0
        face = [int(v - 1) for v in simp if v != 0]
        # Orient face outward
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        to_center = center - v0
        if np.dot(normal, to_center) > 0:
            face = face[::-1]
        raw_faces.append(face)

    # Triangulate any non-triangular faces (ConvexHull gives triangles,
    # but keep the step for robustness)
    tri_faces: list[list[int]] = []
    for face in raw_faces:
        tri_faces.extend(_triangulate_face(face))

    return CoordinationPolyhedron(
        center_index=center_index,
        center_position=np.asarray(center, dtype=float),
        vertex_indices=vertex_indices or list(range(len(vertices))),
        vertex_positions=vertices,
        faces=tri_faces,
        volume=float(hull.volume),
    )
