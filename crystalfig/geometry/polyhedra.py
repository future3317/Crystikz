"""Coordination polyhedron construction using ConvexHull."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from crystalfig.exceptions import GeometryError


def coordination_shell_has_volume(vectors, relative_tolerance=0.05):
    """Return whether neighbor vectors define a genuinely three-dimensional shell."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) < 4:
        return False
    singular_values = np.linalg.svd(vectors - vectors.mean(axis=0), compute_uv=False)
    return singular_values[-1] > relative_tolerance * singular_values[0]


def orient_faces_outward(vertices, faces):
    """Return convex-hull faces with a consistent outward winding."""
    vertices = np.asarray(vertices, dtype=float)
    interior = vertices.mean(axis=0)
    oriented = []
    for face in faces:
        face = list(map(int, face))
        points = vertices[face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if np.dot(normal, points.mean(axis=0) - interior) < 0:
            face.reverse()
        oriented.append(face)
    return oriented


def validate_mesh_winding(vertices, faces, tolerance=1e-8):
    """Return whether every non-degenerate convex face points away from the interior."""
    vertices = np.asarray(vertices, dtype=float)
    interior = vertices.mean(axis=0)
    for face in faces:
        points = vertices[np.asarray(face, dtype=int)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if np.linalg.norm(normal) <= tolerance:
            return False
        if np.dot(normal, points.mean(axis=0) - interior) < -tolerance:
            return False
    return True


def merge_coplanar_faces(vertices, faces, tolerance=1e-7):
    """Collapse coplanar hull triangles into physical polygon faces."""
    vertices = np.asarray(vertices, dtype=float)
    faces = orient_faces_outward(vertices, faces)
    groups = []
    scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    for face in faces:
        face = list(map(int, face))
        points = vertices[face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = np.linalg.norm(normal)
        if norm <= tolerance:
            continue
        normal /= norm
        offset = float(np.dot(normal, points[0]))
        for group in groups:
            if np.dot(normal, group["normal"]) >= 1 - tolerance and abs(offset - group["offset"]) <= tolerance * scale:
                group["faces"].append(face)
                break
        else:
            groups.append({"normal": normal, "offset": offset, "faces": [face]})

    polygons = []
    for group in groups:
        directed = {}
        for face in group["faces"]:
            for start, end in zip(face, face[1:] + face[:1], strict=True):
                edge = (start, end)
                reverse = (end, start)
                if reverse in directed:
                    del directed[reverse]
                else:
                    directed[edge] = end
        if not directed:
            continue
        next_vertex = {edge[0]: edge[1] for edge in directed}
        start = next(iter(next_vertex))
        polygon = [start]
        while len(polygon) <= len(next_vertex):
            current = next_vertex.get(polygon[-1])
            if current is None or current == start:
                break
            polygon.append(current)
        if len(polygon) >= 3 and next_vertex.get(polygon[-1]) == start:
            polygons.append(polygon)
        else:
            polygons.extend(group["faces"])
    return orient_faces_outward(vertices, polygons)


def visible_polyhedron_edges(vertices, faces, camera_rotation, edge_mode="silhouette"):
    """Classify convex-hull edges for a camera-facing 2D projection.

    Returns ``(vertex_a, vertex_b, front_facing)`` physical hull edges.
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

    # The central atom is inside the ligand hull for ordinary coordination
    # geometries. Exclude the center-incident facets, then orient the remaining
    # ligand faces independently of Qhull's simplex ordering.
    raw_faces = [
        [int(vertex - 1) for vertex in face]
        for face in hull.simplices
        if 0 not in face
    ]
    raw_faces = merge_coplanar_faces(vertices, raw_faces)

    return CoordinationPolyhedron(
        center_index=center_index,
        center_position=np.asarray(center, dtype=float),
        vertex_indices=vertex_indices or list(range(len(vertices))),
        vertex_positions=vertices,
        faces=raw_faces,
        volume=float(hull.volume),
    )
