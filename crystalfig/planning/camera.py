"""Camera candidate ranking and renderer-independent visibility estimates."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from crystalfig.scene.camera import Camera
from crystalfig.scene.primitives import Bond, Polyhedron

_CENTER_LIGAND_COLLAPSE_WEIGHT = 0.75


def _projection_crowding_fraction(projected, positions, radii) -> float:
    if len(projected) < 2:
        return 0.0
    pairs = cKDTree(projected).query_pairs(
        3.0 * float(np.median(radii)), output_type="ndarray"
    )
    counts = np.zeros(len(projected), dtype=int)
    for i, j in pairs:
        if np.linalg.norm(positions[i] - positions[j]) < 1e-8:
            continue
        counts[i] += 1
        counts[j] += 1
    return float(np.count_nonzero(counts >= 4) / len(projected))


def _convex_polygon_intersection_area(subject, clipper):
    def cross2d(first, second):
        return first[0] * second[1] - first[1] * second[0]

    polygon = [np.asarray(point, dtype=float) for point in subject]
    clipper = [np.asarray(point, dtype=float) for point in clipper]
    for start, end in zip(clipper, clipper[1:] + clipper[:1], strict=True):
        clipped = []
        for current, following in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            edge = end - start
            side_current = cross2d(edge, current - start)
            side_following = cross2d(edge, following - start)
            if side_current >= -1e-9:
                clipped.append(current)
            if (side_current >= 0) != (side_following >= 0):
                direction = following - current
                denominator = cross2d(edge, direction)
                if abs(float(denominator)) > 1e-12:
                    fraction = cross2d(edge, start - current) / denominator
                    clipped.append(current + fraction * direction)
        polygon = clipped
        if not polygon:
            return 0.0
    points = np.asarray(polygon)
    twice_area = np.dot(points[:, 0], np.roll(points[:, 1], -1))
    twice_area -= np.dot(points[:, 1], np.roll(points[:, 0], -1))
    return abs(float(twice_area)) / 2


def _projected_bond_crossing_fraction(bonds, camera, projected=None) -> float:
    if len(bonds) < 2:
        return 0.0
    starts = np.asarray([bond.start for bond in bonds])
    ends = np.asarray([bond.end for bond in bonds])
    if projected is None:
        projected = camera.project(np.concatenate((starts, ends), axis=0))
    count = len(bonds)
    starts_2d, ends_2d = projected[:count], projected[count:]
    vectors = ends_2d - starts_2d
    lengths = np.linalg.norm(vectors, axis=1)
    positive = lengths[lengths > 1e-9]
    if not len(positive):
        return 0.0
    midpoints = (starts_2d + ends_2d) / 2
    pairs = cKDTree(midpoints).query_pairs(float(np.median(positive)), output_type="ndarray")
    crossings = 0
    for first, second in pairs:
        if min(
            np.linalg.norm(starts[first] - starts[second]),
            np.linalg.norm(starts[first] - ends[second]),
            np.linalg.norm(ends[first] - starts[second]),
            np.linalg.norm(ends[first] - ends[second]),
        ) < 1e-7:
            continue
        p, r = starts_2d[first], vectors[first]
        q, s = starts_2d[second], vectors[second]
        denominator = r[0] * s[1] - r[1] * s[0]
        if abs(float(denominator)) < 1e-10:
            continue
        offset = q - p
        t = (offset[0] * s[1] - offset[1] * s[0]) / denominator
        u = (offset[0] * r[1] - offset[1] * r[0]) / denominator
        if 0.08 < t < 0.92 and 0.08 < u < 0.92:
            crossings += 1
    return float(crossings / count)


class CameraPlanner:
    """Coordinate camera scoring while leaving renderers out of planning."""

    def __init__(self, structure, camera, intent="auto", layer_normal=None, layer_translations=None, centroid=None):
        self.structure = structure
        self.camera = camera
        self.intent = intent
        self.layer_normal = None if layer_normal is None else np.asarray(layer_normal, dtype=float)
        self.layer_translations = np.asarray([] if layer_translations is None else layer_translations, dtype=float).reshape(-1, 3)
        self.centroid = np.asarray(centroid, dtype=float) if centroid is not None else np.zeros(3)

    def score(self, scene):
        return self.refine_roll(scene, self.generate_candidates(scene), self.layer_normal, count=2)

    def generate_candidates(self, scene):
        world_bounds = scene.bounding_box()
        points = np.asarray(world_bounds) if world_bounds is not None else None
        target = points.mean(axis=0) if points is not None else self.centroid
        primitives = scene.all_primitives()
        spheres = [p for p in primitives if p.__class__.__name__ == "Sphere"]
        sphere_positions = np.asarray([p.position for p in spheres], dtype=float).reshape((-1, 3))
        radii = np.asarray([p.radius for p in spheres], dtype=float)
        bonds = [p for p in primitives if isinstance(p, Bond)]
        bond_starts = np.asarray([bond.start for bond in bonds], dtype=float).reshape((-1, 3))
        bond_ends = np.asarray([bond.end for bond in bonds], dtype=float).reshape((-1, 3))
        bond_distances = np.asarray([bond.distance for bond in bonds], dtype=float)
        bond_world_endpoints = np.concatenate((bond_starts, bond_ends), axis=0)
        polyhedra = [p for p in primitives if isinstance(p, Polyhedron)]
        polyhedron_vertices = [np.asarray(poly.vertices, dtype=float) for poly in polyhedra]
        sphere_by_identity = {
            (sphere.metadata.get("site_index"), tuple(sphere.metadata.get("image_offset", (0, 0, 0)))): sphere.position
            for sphere in spheres
        }
        polyhedron_centers = []
        for poly, vertices in zip(polyhedra, polyhedron_vertices, strict=True):
            center = poly.metadata.get("center_position")
            if center is None:
                center = sphere_by_identity.get((poly.center_site, (0, 0, 0)))
            polyhedron_centers.append(
                np.asarray(center, dtype=float) if center is not None else vertices.mean(axis=0)
            )
        if self.intent == "coordination":
            target = (
                np.mean(polyhedron_centers, axis=0)
                if polyhedron_centers else self.centroid
            )
        polyhedron_faces = []
        for vertices, poly in zip(polyhedron_vertices, polyhedra, strict=True):
            center = vertices.mean(axis=0)
            normals = []
            for face in poly.faces:
                face_vertices = vertices[np.asarray(face, dtype=int)]
                if len(face_vertices) < 3:
                    continue
                normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                if np.dot(normal, face_vertices.mean(axis=0) - center) < 0:
                    normal = -normal
                normals.append(normal / norm)
            polyhedron_faces.append(np.asarray(normals, dtype=float).reshape((-1, 3)))

        directions = []
        labels = []
        for label, vector in (
            [(f"direct_{axis}", row) for axis, row in zip("abc", self.structure.lattice.matrix, strict=True)]
            + [(f"plane_{axis}", row) for axis, row in zip("abc", self.structure.lattice.reciprocal_matrix, strict=True)]
        ):
            directions.append(np.asarray(vector, dtype=float))
            labels.append(label)

        for coefficients in product((-2, -1, 0, 1, 2), repeat=3):
            if not any(coefficients):
                continue
            first_nonzero = next(value for value in coefficients if value)
            if first_nonzero < 0:
                continue
            direction = np.asarray(coefficients, dtype=float) @ self.structure.lattice.matrix
            if np.linalg.norm(direction) > 1e-8:
                directions.append(direction)
                direction_label = "".join(
                    "+" if value > 0 else "-" if value < 0 else "0" for value in coefficients
                )
                labels.append(f"lattice_{direction_label}")

        geometry_points = []
        for primitive in scene.all_primitives():
            if getattr(primitive, "layer", "geometry") == "annotation":
                continue
            if hasattr(primitive, "position"):
                geometry_points.append(np.asarray(primitive.position, dtype=float))
            elif hasattr(primitive, "vertices"):
                geometry_points.extend(np.asarray(p, dtype=float) for p in primitive.vertices)
            elif hasattr(primitive, "points"):
                geometry_points.extend(np.asarray(p, dtype=float) for p in primitive.points)
        if len(geometry_points) >= 3:
            _, _, axes = np.linalg.svd(np.asarray(geometry_points) - target, full_matrices=False)
            for index, direction in enumerate(axes[:3]):
                directions.append(direction)
                labels.append(f"principal_{index + 1}")

        base = directions[:6]
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                vector = base[i] / np.linalg.norm(base[i]) + base[j] / np.linalg.norm(base[j])
                if np.linalg.norm(vector) > 1e-8:
                    directions.append(vector)
                    labels.append(f"tilt_{labels[i]}_{labels[j]}")

        camera_candidates = []
        seen = []
        for label, direction in zip(labels, directions, strict=True):
            direction = direction / np.linalg.norm(direction)
            if any(np.dot(direction, previous) > 0.9995 for previous in seen):
                continue
            seen.append(direction)
            camera_candidates.extend(
                (label, direction, roll) for roll in (0.0, 90.0)
            )

        scored = []
        settings = self.camera
        layer_normal = self.layer_normal if self.intent == "layered" else None
        layer_translations = self.layer_translations if layer_normal is not None else np.empty((0, 3))
        if layer_normal is not None and len(layer_translations) >= 2:
            in_plane = layer_translations[0] + 0.37 * layer_translations[1]
            in_plane /= np.linalg.norm(in_plane)
            side = in_plane - np.dot(in_plane, layer_normal) * layer_normal
            side /= max(np.linalg.norm(side), 1e-12)
            for label, direction in (
                ("layered_side", side),
                ("layered_oblique", side + 0.55 * layer_normal),
            ):
                direction = direction / np.linalg.norm(direction)
                camera_candidates.extend((label, direction, roll) for roll in (0.0, 90.0))
        for label, direction, roll in camera_candidates:
            layer_separation = 0.0
            layer_normal_vertical = 0.0
            projected_layer_spacing = 0.0
            layer_spacing_angstrom = 0.0
            camera = Camera.along_direction(
                direction, target=target, projection=settings.projection, roll=roll
            )
            camera.distance = settings.distance
            camera.fov = settings.fov
            camera.padding = settings.padding
            camera.auto_fit = settings.auto_fit
            frame = camera.scene_frame(scene)
            if frame is None:
                score = 0.0
                bond_visibility = 0.0
                atom_overlap = 0.0
                projection_crowding = 0.0
                polyhedron_area = 0.0
                polyhedron_depth = 0.0
                polyhedron_overlap = 0.0
                polyhedron_face_visibility = 0.0
                polyhedron_ligand_collapse = 0.0
                center_ligand_collapse = 0.0
                layer_separation = 0.0
                bond_crossings = 0.0
            else:
                diagonal = float(np.hypot(frame.width, frame.height))
                projected = camera.project(sphere_positions) if len(spheres) else np.empty((0, 2))
                overlapping_atoms: set[int] = set()
                if len(spheres) > 1:
                    candidate_pairs = cKDTree(projected).query_pairs(
                        1.56 * float(radii.max()), output_type="ndarray"
                    )
                    for i, j in candidate_pairs:
                        if np.linalg.norm(sphere_positions[i] - sphere_positions[j]) < 1e-8:
                            continue
                        if np.linalg.norm(projected[i] - projected[j]) < 0.78 * (radii[i] + radii[j]):
                            overlapping_atoms.update((int(i), int(j)))
                atom_overlap = len(overlapping_atoms) / max(len(spheres), 1)
                projection_crowding = _projection_crowding_fraction(
                    projected, sphere_positions, radii
                )
                projected_bonds = camera.project(bond_world_endpoints) if len(bonds) else np.empty((0, 2))
                if bonds:
                    projected_lengths = np.linalg.norm(
                        projected_bonds[len(bonds):] - projected_bonds[:len(bonds)], axis=1
                    )
                    bond_visibility = float(np.mean(np.minimum(projected_lengths / np.maximum(bond_distances, 1e-9), 1.0)))
                else:
                    bond_visibility = 0.5
                bond_crossings = _projected_bond_crossing_fraction(bonds, camera, projected_bonds)
                polyhedron_area = 0.0
                polyhedron_depth = 0.0
                polyhedron_overlap = 0.0
                polyhedron_face_visibility = 0.0
                polyhedron_ligand_collapse = 0.0
                center_ligand_collapse = 0.0
                if polyhedra:
                    areas = []
                    depth_ratios = []
                    projected_polygons = []
                    projected_polygon_bounds = []
                    projected_polygon_areas = []
                    face_visibility = []
                    for poly_index, _poly in enumerate(polyhedra):
                        vertices = polyhedron_vertices[poly_index]
                        view_vertices = vertices @ camera.rotation_matrix().T
                        uv = camera.project(vertices)
                        if len(uv) >= 3:
                            try:
                                hull = ConvexHull(uv)
                                areas.append(float(hull.volume))
                                hull_points = uv[hull.vertices]
                                projected_polygons.append(hull_points)
                                projected_polygon_bounds.append((hull_points.min(axis=0), hull_points.max(axis=0)))
                                projected_polygon_areas.append(float(hull.volume))
                                projected_span = float(np.linalg.norm(np.ptp(uv, axis=0)))
                                depth_ratios.append(
                                    float(np.ptp(view_vertices[:, 2])) / max(projected_span, 1e-9)
                                )
                            except QhullError:
                                continue
                        vertex_array = vertices
                        center = vertex_array.mean(axis=0)
                        toward_camera = -direction
                        normals = polyhedron_faces[poly_index]
                        if len(normals):
                            face_visibility.append(float(np.maximum(normals @ toward_camera, 0.0).mean()))
                    polyhedron_area = min(float(np.mean(areas)) / max(diagonal * diagonal, 1e-9), 1.0) if areas else 0.0
                    polyhedron_depth = min(float(np.mean(depth_ratios)), 1.0) if depth_ratios else 0.0
                    polyhedron_face_visibility = float(np.mean(face_visibility)) if face_visibility else 0.0
                    polyhedron_ligand_collapse, center_ligand_collapse = self.shell_collapse_metrics(
                        camera, polyhedron_vertices, polyhedron_centers
                    )
                    overlaps = []
                    for index, first in enumerate(projected_polygons):
                        first_area = projected_polygon_areas[index]
                        first_min, first_max = projected_polygon_bounds[index]
                        for second_index in range(index + 1, len(projected_polygons)):
                            second = projected_polygons[second_index]
                            second_min, second_max = projected_polygon_bounds[second_index]
                            if np.any(first_max < second_min) or np.any(second_max < first_min):
                                continue
                            second_area = projected_polygon_areas[second_index]
                            intersection = _convex_polygon_intersection_area(first, second)
                            overlaps.append(intersection / max(min(first_area, second_area), 1e-9))
                    polyhedron_overlap = float(np.mean(overlaps)) if overlaps else 0.0
                aspect_penalty = abs(np.log(frame.aspect)) + (0.025 if frame.aspect < 1 else 0.0)
                score = (
                    1.0 - atom_overlap + 0.35 * bond_visibility
                    + 0.35 * polyhedron_area + 0.22 * polyhedron_depth
                    + 0.18 * polyhedron_face_visibility
                    - 0.35 * polyhedron_ligand_collapse
                    - _CENTER_LIGAND_COLLAPSE_WEIGHT * center_ligand_collapse
                    - 0.45 * polyhedron_overlap - 0.08 * aspect_penalty
                    - 1.4 * bond_crossings
                    - 0.25 * projection_crowding
                )
                if layer_normal is not None and frame is not None:
                    cosine = abs(float(np.dot(direction, layer_normal)))
                    layer_separation = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
                    layer_normal_vertical = abs(float(np.dot(layer_normal, camera.rotation_matrix()[1])))
                    heights = np.unique(np.round(sphere_positions @ layer_normal, 2)) if len(spheres) else np.empty(0)
                    gaps = np.diff(heights)
                    gaps = gaps[gaps > 0.05]
                    layer_spacing_angstrom = float(np.min(gaps)) if len(gaps) else 0.0
                    projected_layer_spacing = min(
                        layer_spacing_angstrom * layer_separation / max(diagonal, 1e-9), 1.0
                    )
                    score += (
                        1.8 * layer_separation + 0.35 * (1.0 - cosine)
                        + 1.5 * projected_layer_spacing + 2.5 * layer_normal_vertical
                    )
            report = {
                "label": label if roll == 0 else f"{label}_roll90",
                "direction_cartesian": direction.tolist(),
                "roll_degrees": roll,
                "score": float(score),
                "atom_overlap_fraction": float(atom_overlap),
                "projection_crowding_fraction": float(projection_crowding),
                "atom_visibility_fraction": None,
                "atom_occlusion_fraction": None,
                "bond_visibility": float(bond_visibility),
                "projected_bond_crossing_fraction": float(bond_crossings),
                "polyhedron_projected_area": float(polyhedron_area),
                "polyhedron_depth_ratio": float(polyhedron_depth),
                "polyhedron_overlap_fraction": float(polyhedron_overlap),
                "polyhedron_face_visibility_fraction": float(polyhedron_face_visibility),
                "polyhedron_ligand_collapse_fraction": float(polyhedron_ligand_collapse),
                "center_ligand_collapse_fraction": float(center_ligand_collapse),
                "projected_aspect": frame.aspect if frame is not None else None,
                "layer_separation": float(layer_separation),
                "layer_normal_vertical_fraction": float(layer_normal_vertical),
                "layer_spacing_angstrom": float(layer_spacing_angstrom),
                "projected_layer_spacing_fraction": float(projected_layer_spacing),
            }
            scored.append((report, camera))
        scored.sort(key=lambda item: (-item[0]["score"], item[0]["label"]))
        # The inexpensive projected geometry score narrows the candidates before
        # the low-resolution depth visibility estimate is evaluated.
        for report, camera in scored[:5]:
            frame = camera.scene_frame(scene)
            visibility = CameraPlanner.sphere_visibility(camera, spheres, frame) if frame else 1.0
            report["atom_visibility_fraction"] = visibility
            report["atom_occlusion_fraction"] = 1.0 - visibility
            report["score"] += 0.35 * visibility
        return sorted(scored, key=lambda item: (-item[0]["score"], item[0]["label"]))

    @staticmethod
    def shell_collapse_metrics(camera, vertices_list, centers):
        """Measure ligand pairs and center–ligand rods that collapse in projection."""
        ligand_pair_scores = []
        center_ligand_scores = []
        for vertices, center in zip(vertices_list, centers, strict=True):
            if len(vertices) < 2:
                continue
            uv = camera.project(vertices)
            center_uv = camera.project(center)[0]
            diameter = float(np.max(np.linalg.norm(uv[:, None, :] - uv[None, :, :], axis=2)))
            if diameter <= 1e-9:
                ligand_pair_scores.append(1.0)
                center_ligand_scores.append(1.0)
                continue
            world_distances = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)
            projected_distances = np.linalg.norm(uv[:, None, :] - uv[None, :, :], axis=2)
            distinct_pairs = np.triu(world_distances > 1e-8, k=1)
            if distinct_pairs.any():
                ligand_pair_scores.append(float(np.mean(
                    projected_distances[distinct_pairs] < 0.33 * diameter
                )))
            center_distances = np.linalg.norm(uv - center_uv, axis=1)
            distinct_ligands = np.linalg.norm(vertices - center, axis=1) > 1e-8
            if distinct_ligands.any():
                center_ligand_scores.append(float(np.mean(
                    center_distances[distinct_ligands] < 0.33 * diameter
                )))
        return (
            float(np.mean(ligand_pair_scores)) if ligand_pair_scores else 0.0,
            float(np.mean(center_ligand_scores)) if center_ligand_scores else 0.0,
        )


    def refine_roll(self, scene, candidates, layer_normal=None, count: int = 5):
        """Search small view tilts around the strongest coarse candidates."""
        from crystalfig.scene.primitives import Bond

        refined = list(candidates)
        primitives = scene.all_primitives()
        spheres = [p for p in primitives if p.__class__.__name__ == "Sphere"]
        bonds = [p for p in primitives if isinstance(p, Bond)]
        polyhedra = [p for p in primitives if isinstance(p, Polyhedron)]
        vertices = [np.asarray(poly.vertices, dtype=float) for poly in polyhedra]
        sphere_by_identity = {
            (sphere.metadata.get("site_index"), tuple(sphere.metadata.get("image_offset", (0, 0, 0)))): sphere.position
            for sphere in spheres
        }
        centers = []
        for poly, poly_vertices in zip(polyhedra, vertices, strict=True):
            center = poly.metadata.get("center_position")
            if center is None:
                center = sphere_by_identity.get((poly.center_site, (0, 0, 0)))
            centers.append(
                np.asarray(center, dtype=float) if center is not None else poly_vertices.mean(axis=0)
            )
        for report, base_camera in candidates[:count]:
            direction = np.asarray(report["direction_cartesian"], dtype=float)
            old_visibility = report["atom_visibility_fraction"] or 0.0
            old_crossing = report["projected_bond_crossing_fraction"]
            old_ligand_collapse = report["polyhedron_ligand_collapse_fraction"]
            old_center_collapse = report["center_ligand_collapse_fraction"]
            old_vertical = report["layer_normal_vertical_fraction"]
            old_aspect = report["projected_aspect"] or 1.0
            old_aspect_penalty = abs(np.log(max(old_aspect, 1e-12)))
            rotation = base_camera.rotation_matrix()
            for axis_name, axis in (("right", rotation[0]), ("up", rotation[1])):
                for angle in (-10.0, -5.0, 5.0, 10.0):
                    # Tilt the view around screen right/up; preserve roll as its
                    # own camera degree of freedom.
                    tangent = np.cross(direction, axis)
                    tangent /= max(np.linalg.norm(tangent), 1e-12)
                    radians = np.radians(angle)
                    refined_direction = direction * np.cos(radians) + tangent * np.sin(radians)
                    self._score_tilt(
                        scene, report, base_camera, refined_direction,
                        layer_normal, spheres, bonds, vertices, centers,
                        old_visibility,
                        old_crossing, old_vertical, old_aspect_penalty,
                        old_ligand_collapse, old_center_collapse,
                        refined, axis_name, angle,
                    )
        return sorted(refined, key=lambda item: (-item[0]["score"], item[0]["label"]))

    def _score_tilt(
        self, scene, report, base_camera, direction, layer_normal,
        spheres, bonds, vertices, centers, old_visibility, old_crossing,
        old_vertical, old_aspect_penalty, old_ligand_collapse,
        old_center_collapse, refined, axis_name, angle,
    ):
        from crystalfig.scene.camera import Camera

        camera = Camera.along_direction(
            direction, target=base_camera.target,
            projection=base_camera.projection, roll=base_camera.roll,
        )
        camera.distance = base_camera.distance
        camera.fov = base_camera.fov
        camera.padding = base_camera.padding
        camera.auto_fit = base_camera.auto_fit
        frame = camera.scene_frame(scene)
        if frame is None:
            return
        visibility = self.sphere_visibility(camera, spheres, frame)
        old_crowding = report["projection_crowding_fraction"]
        atom_positions = np.asarray([sphere.position for sphere in spheres], dtype=float).reshape((-1, 3))
        projected_atoms = camera.project(atom_positions) if len(spheres) else np.empty((0, 2))
        crowding = _projection_crowding_fraction(
            projected_atoms,
            atom_positions,
            np.asarray([sphere.radius for sphere in spheres]),
        )
        crossing = _projected_bond_crossing_fraction(bonds, camera)
        ligand_collapse, center_collapse = self.shell_collapse_metrics(camera, vertices, centers)
        vertical = (
            abs(float(np.dot(layer_normal, camera.rotation_matrix()[1])))
            if layer_normal is not None else 0.0
        )
        aspect_penalty = abs(np.log(max(frame.aspect, 1e-12)))
        score = report["score"] + 0.35 * (visibility - old_visibility)
        score -= 0.25 * (crowding - old_crowding)
        score += 1.4 * (old_crossing - crossing)
        score -= 0.35 * (ligand_collapse - old_ligand_collapse)
        score -= _CENTER_LIGAND_COLLAPSE_WEIGHT * (center_collapse - old_center_collapse)
        score -= 0.08 * (aspect_penalty - old_aspect_penalty)
        score += 2.5 * (vertical - old_vertical)
        adjusted = dict(report)
        adjusted.update(
            label=f"{report['label']}_tilt_{axis_name}{angle:+g}",
            score=float(score),
            roll_degrees=camera.roll,
            atom_visibility_fraction=visibility,
            atom_occlusion_fraction=1.0 - visibility,
            projection_crowding_fraction=crowding,
            projected_bond_crossing_fraction=crossing,
            polyhedron_ligand_collapse_fraction=ligand_collapse,
            center_ligand_collapse_fraction=center_collapse,
            projected_aspect=frame.aspect,
            layer_normal_vertical_fraction=vertical,
            direction_cartesian=np.asarray(direction).tolist(),
        )
        refined.append((adjusted, camera))

    @staticmethod
    def sphere_visibility(camera, spheres, frame, resolution: int = 32) -> float:
        """Estimate atom visibility by rasterizing projected sphere depth at low resolution."""
        if not spheres:
            return 1.0
        x = np.linspace(frame.xmin, frame.xmax, resolution)
        y = np.linspace(frame.ymin, frame.ymax, resolution)
        xx, yy = np.meshgrid(x, y)
        depth_buffer = np.full((resolution, resolution), -np.inf)
        samples = []
        for sphere in spheres:
            uv, depth = camera.project(sphere.position, return_depth=True)
            uv = uv[0] / max(camera.scale, 1e-12)
            radius = float(sphere.radius)
            distance2 = (xx - uv[0]) ** 2 + (yy - uv[1]) ** 2
            inside = distance2 <= radius * radius
            front_depth = float(depth[0]) + np.sqrt(np.maximum(radius * radius - distance2, 0.0))
            missed = not bool(inside.any())
            samples.append((inside, front_depth, missed))
            if not missed:
                depth_buffer[inside] = np.maximum(depth_buffer[inside], front_depth[inside])
        visible = 0.0
        total = 0
        for inside, front_depth, missed in samples:
            if missed:
                visible += 1
                total += 1
                continue
            total += int(inside.sum())
            visible += int((inside & (front_depth >= depth_buffer - 1e-8)).sum())
        return float(visible / max(total, 1))
