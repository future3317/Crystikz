"""Scene graph: a collection of primitives ready for rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crystalfig.scene.primitives import Group


@dataclass
class Scene:
    """Backend-independent scene graph."""

    primitives: list[Any] = field(default_factory=list)
    groups: dict[str, Group] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, primitive: Any, group: str | None = None) -> Scene:
        """Add a primitive or group to the scene."""
        if group is not None:
            if group not in self.groups:
                self.groups[group] = Group(name=group)
            self.groups[group].add(primitive)
        else:
            self.primitives.append(primitive)
        return self

    def extend(self, primitives: list[Any], group: str | None = None) -> Scene:
        for p in primitives:
            self.add(p, group=group)
        return self

    def get_group(self, name: str) -> Group | None:
        """Return a group by name, or None if it does not exist."""
        return self.groups.get(name)

    def all_primitives(self) -> list[Any]:
        """Flatten groups and return all visible primitives in deterministic order."""
        out = [p for p in self.primitives if getattr(p, "visible", True)]
        for name in sorted(self.groups.keys()):
            group = self.groups[name]
            if not getattr(group, "visible", True):
                continue
            out.extend(p for p in group.primitives if getattr(p, "visible", True))
        return out

    def filter(self, predicate) -> Scene:
        """Return a new scene containing only primitives matching predicate."""
        new = Scene(metadata=self.metadata.copy())
        for p in self.all_primitives():
            if predicate(p):
                new.add(p)
        return new

    def bounding_box(self) -> np.ndarray | None:
        """Return axis-aligned bounding box as [[xmin,ymin,zmin], [xmax,ymax,zmax]]."""
        positions = []
        for p in self.all_primitives():
            if hasattr(p, "position"):
                positions.append(p.position)
            elif hasattr(p, "start"):
                if hasattr(p, "end"):
                    positions.extend([p.start, p.end])
                elif hasattr(p, "direction"):
                    positions.extend([p.start, p.start + p.direction])
            elif hasattr(p, "vertices"):
                positions.extend(p.vertices)
            elif hasattr(p, "points"):
                positions.extend(p.points)
        if not positions:
            return None
        positions = np.array(positions)
        return np.array([positions.min(axis=0), positions.max(axis=0)])

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "metadata": self.metadata,
            "primitives": [self._primitive_to_dict(p) for p in self.primitives],
            "groups": {k: [self._primitive_to_dict(p) for p in v.primitives] for k, v in self.groups.items()},
        }

    @staticmethod
    def _primitive_to_dict(p: Any) -> dict[str, Any]:
        d = {"type": type(p).__name__}
        if isinstance(p, Group):
            d["primitives"] = [Scene._primitive_to_dict(x) for x in p.primitives]
            return d
        for key, value in p.__dict__.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
                d[key] = [v.tolist() for v in value]
            else:
                d[key] = value
        return d
