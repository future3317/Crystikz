"""Camera and projection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from crystalfig.model.lattice import Lattice


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
    ) -> Camera:
        """Create a camera looking along a direct-space direction [uvw]."""
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
        rotation = np.vstack([x_axis, y_axis, z_axis])

        return cls(
            target=target if target is None else np.asarray(target, dtype=float),
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
        """Adjust scale so every projected primitive fits in a [-1, 1] view.

        Uses unscaled camera coordinates so degenerate views (e.g. looking along
        a body diagonal) do not collapse to zero extent.
        """
        bounds = self._scene_camera_bounds(scene)
        if bounds is None:
            return self
        (umin, vmin), (umax, vmax) = bounds
        size = max(umax - umin, vmax - vmin)
        if size > 0:
            self.scale = 2.0 * (1.0 - self.padding) / size
        return self

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
                add_point(p.start)
                add_point(p.end)
            elif hasattr(p, "vertices"):  # Polyhedron
                for v in p.vertices:
                    add_point(v)
            elif hasattr(p, "points"):  # Polygon, Polyline
                for v in p.points:
                    add_point(v)
            elif hasattr(p, "start") and hasattr(p, "direction"):  # Arrow, Axis
                add_point(p.start)
                add_point(p.start + p.direction)

        if not pts:
            return None
        arr = np.array(pts)
        return arr.min(axis=0), arr.max(axis=0)
