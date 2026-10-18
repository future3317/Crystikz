"""Camera and projection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from crystalfig.model.lattice import Lattice


@dataclass(frozen=True)
class SceneFrame:
    """Camera-plane bounds for visible world-space geometry."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(self.xmax - self.xmin, 1e-9)

    @property
    def height(self) -> float:
        return max(self.ymax - self.ymin, 1e-9)

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass
class Camera:
    """Orthographic or perspective camera for 3D scene projection.

    Convention:
        The camera looks along the negative z-axis of the camera coordinate
        system.  Points are transformed from world to camera coordinates via
        ``R @ (p - target)`` where ``R`` is the rotation matrix.  Projection
        then drops the z coordinate for orthographic or applies perspective
        division for perspective.
    """

    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 10.0
    elevation: float = 25.0
    azimuth: float = 45.0
    roll: float = 0.0
    projection: Literal["orthographic", "perspective"] = "orthographic"
    fov: float = 30.0
    scale: float = 1.0
    auto_fit: bool = True
    padding: float = 0.1
    _rotation: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.target = np.asarray(self.target, dtype=float)
        if self._rotation is not None:
            self._rotation = np.asarray(self._rotation, dtype=float)

    @classmethod
    def along_direction(
        cls,
        direction: np.ndarray,
        target: np.ndarray | None = None,
        projection: Literal["orthographic", "perspective"] = "orthographic",
        roll: float = 0.0,
    ) -> Camera:
        """Create a camera looking along a direction, with an optional screen roll."""
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("Direction vector must be non-zero.")
        direction = direction / norm

        # Camera looks along negative z; build orthonormal basis via Gram-Schmidt.
        z_axis = -direction
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        roll_radians = np.radians(roll)
        x_rolled = np.cos(roll_radians) * x_axis - np.sin(roll_radians) * y_axis
        y_rolled = np.sin(roll_radians) * x_axis + np.cos(roll_radians) * y_axis
        rotation = np.vstack([x_rolled, y_rolled, z_axis])

        return cls(
            target=target if target is None else np.asarray(target, dtype=float),
            roll=roll,
            projection=projection,
            _rotation=rotation,
        )

    @classmethod
    def normal_to_plane(
        cls,
        hkl: np.ndarray,
        lattice: Lattice,
        target: np.ndarray | None = None,
        projection: Literal["orthographic", "perspective"] = "orthographic",
    ) -> Camera:
        """Create a camera looking along the normal of a Miller plane (hkl)."""
        normal = lattice.reciprocal_vector(hkl)
        return cls.along_direction(normal, target=target, projection=projection)

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------
    def rotation_matrix(self) -> np.ndarray:
        """Return the world-to-camera rotation matrix."""
        if self._rotation is not None:
            return self._rotation

        elev = np.radians(self.elevation)
        azim = np.radians(self.azimuth)
        roll = np.radians(self.roll)

        # Rotation around Z by -azimuth
        r_z = np.array([
            [np.cos(azim), np.sin(azim), 0.0],
            [-np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ])
        # Rotation around X by -elevation
        r_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev), np.sin(elev)],
            [0.0, -np.sin(elev), np.cos(elev)],
        ])
        r_roll = np.array([
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ])
        return r_roll @ r_x @ r_z

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project(
        self,
        points: np.ndarray,
        return_depth: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Project world points to 2D camera coordinates.

        Returns (N, 2) array of (u, v) coordinates, optionally with (N,) depth.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        rot = self.rotation_matrix()
        cam = (points - self.target) @ rot.T

        if self.projection == "perspective":
            # Simple perspective: camera at distance along -z
            z0 = self.distance
            denom = np.maximum(z0 - cam[:, 2], 1e-6)
            uv = cam[:, :2] * (z0 / denom)
        else:
            uv = cam[:, :2]

        uv = uv * self.scale
        if return_depth:
            return uv, cam[:, 2]
        return uv

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Return camera-space depth (z) for world points."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        rot = self.rotation_matrix()
        cam = (points - self.target) @ rot.T
        return cam[:, 2]

    def fit_to_bounding_box(self, bbox: np.ndarray) -> Camera:
        """Adjust scale so the bounding box fits in a [-1, 1] view."""
        if bbox is None or bbox.size == 0:
            return self
        corners = np.array([
            [bbox[0, 0], bbox[0, 1], bbox[0, 2]],
            [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
            [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
            [bbox[0, 0], bbox[1, 1], bbox[1, 2]],
            [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
            [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
            [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
            [bbox[1, 0], bbox[1, 1], bbox[1, 2]],
        ])
        uv = self.project(corners)
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        size = max(umax - umin, vmax - vmin)
        if size > 0:
            self.scale = 2.0 * (1.0 - self.padding) / size
        return self

    def fit_to_scene(self, scene) -> Camera:
        """Center and scale the camera against all visible scene geometry."""
        frame = self.scene_frame(scene)
        if frame is None:
            return self
        rotation = self.rotation_matrix()
        offset = (frame.xmin + frame.xmax) / 2 * rotation[0]
        offset += (frame.ymin + frame.ymax) / 2 * rotation[1]
        if np.linalg.norm(offset) > 1e-12:
            self.target = self.target + offset
        frame = self.scene_frame(scene)
        if frame is not None:
            self.scale = 2.0 * (1.0 - self.padding) / max(frame.width, frame.height)
        return self

    def scene_frame(self, scene) -> SceneFrame | None:
        """Return the shared unscaled frame used by automatic layout."""
        bounds = self._scene_camera_bounds(scene)
        if bounds is None:
            return None
        (xmin, ymin), (xmax, ymax) = bounds
        return SceneFrame(float(xmin), float(ymin), float(xmax), float(ymax))

    def scene_bounding_sphere(self, scene) -> tuple[np.ndarray, float] | None:
        """Return a world-space sphere enclosing visible geometry and atom radii."""
        points = []
        radii = []
        for primitive in scene.all_primitives():
            if getattr(primitive, "layer", "geometry") == "annotation":
                continue
            if getattr(primitive, "coordinate_space", "world") == "screen":
                continue
            if hasattr(primitive, "position"):
                points.append(np.asarray(primitive.position, dtype=float))
                radii.append(float(getattr(primitive, "radius", 0.0)))
            elif hasattr(primitive, "start") and hasattr(primitive, "end"):
                points.extend([np.asarray(primitive.start, dtype=float), np.asarray(primitive.end, dtype=float)])
                radii.extend([float(getattr(primitive, "radius", 0.0))] * 2)
            elif hasattr(primitive, "vertices"):
                points.extend(np.asarray(point, dtype=float) for point in primitive.vertices)
            elif hasattr(primitive, "points"):
                points.extend(np.asarray(point, dtype=float) for point in primitive.points)
            elif hasattr(primitive, "start") and hasattr(primitive, "direction"):
                points.extend([
                    np.asarray(primitive.start, dtype=float),
                    np.asarray(primitive.start, dtype=float) + np.asarray(primitive.direction, dtype=float),
                ])
        if not points:
            return None
        points = np.asarray(points)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(float(np.linalg.norm(points - center, axis=1).max()), max(radii, default=0.0))
        return center, radius

    def _scene_camera_bounds(self, scene):
        """Return ((umin, vmin), (umax, vmax)) in unscaled camera coordinates."""
        pts = []
        rot = self.rotation_matrix()
        x_axis = rot[0]
        y_axis = rot[1]

        def add_point(point, radius: float = 0.0):
            p = np.asarray(point, dtype=float) - self.target
            ux = float(np.dot(p, x_axis))
            uy = float(np.dot(p, y_axis))
            if radius > 0.0:
                pts.extend([(ux + radius, uy), (ux - radius, uy), (ux, uy + radius), (ux, uy - radius)])
            else:
                pts.append((ux, uy))

        for p in scene.all_primitives():
            if getattr(p, "layer", "geometry") == "annotation" or getattr(p, "coordinate_space", "world") == "screen":
                continue
            if hasattr(p, "position"):  # Sphere, Text
                r = getattr(p, "radius", 0.0)
                add_point(p.position, r)
            elif hasattr(p, "start") and hasattr(p, "end"):  # Bond, Line, CellEdge, Cylinder
                radius = float(getattr(p, "radius", 0.0))
                add_point(p.start, radius)
                add_point(p.end, radius)
            elif hasattr(p, "vertices"):  # Polyhedron
                for v in p.vertices:
                    add_point(v)
            elif hasattr(p, "points"):  # Polygon, Polyline
                for v in p.points:
                    add_point(v)
            elif hasattr(p, "start") and hasattr(p, "direction"):  # Arrow, Axis
                radius = max(float(getattr(p, "shaft_radius", 0.0)), float(getattr(p, "head_radius", 0.0)))
                add_point(p.start, radius)
                add_point(p.start + p.direction, radius)

        if not pts:
            return None
        arr = np.array(pts)
        return arr.min(axis=0), arr.max(axis=0)
