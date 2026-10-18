"""Coordination polyhedron construction using ConvexHull."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from crystalfig.exceptions import GeometryError


def visible_polyhedron_edges(vertices, faces, camera_rotation, edge_mode="silhouette"):
    """Classify convex-hull edges for a camera-facing 2D projection.

    Returns ``(vertex_a, vertex_b, front_facing)`` edges. Coplanar triangle
    diagonals are always omitted, so triangulation does not become visible.
    """
    if edge_mode == "none":
        return []
    camera_vertices = (np.asarray(vertices, dtype=float) @ np.asarray(camera_rotation).T)
    edge_faces: dict[tuple[int, int], list[tuple[np.ndarray, bool]]] = {}
    for face in faces:
        points = camera_vertices[face]
        normal = np.zeros(3)
        for i in range(len(points)):
            a, b = points[i], points[(i + 1) % len(points)]
            normal += np.cross(a, b)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        front = normal[2] >= 0
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            edge_faces.setdefault((min(a, b), max(a, b)), []).append((normal, front))

    result = []
    for edge, adjacent in edge_faces.items():
        if len(adjacent) > 1 and all(np.dot(adjacent[0][0], n) > 0.985 for n, _ in adjacent[1:]):
            continue
        fronts = [front for _, front in adjacent]
        has_front = any(fronts)
        is_silhouette = len(fronts) == 1 or any(fronts) and not all(fronts)
        if edge_mode == "silhouette" and not is_silhouette:
            continue
        if edge_mode == "visible" and not has_front:
            continue
        if edge_mode not in {"silhouette", "visible", "all"}:
            raise ValueError(f"Unknown polyhedron edge mode: {edge_mode}")
        result.append((*edge, has_front))
    return result


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
