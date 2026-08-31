"""Site property container with typed accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SiteProperties:
    """Container for arbitrary site properties.

    Properties are stored as a dictionary and can be scalar, vector, or string.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def has(self, key: str) -> bool:
        return key in self.data

    @property
    def magnetic_moment(self) -> np.ndarray | None:
        mag = self.data.get("magmom")
        if mag is None:
            return None
        return np.asarray(mag, dtype=float)

    @magnetic_moment.setter
    def magnetic_moment(self, value: np.ndarray | float) -> None:
        self.data["magmom"] = value

    @property
    def force(self) -> np.ndarray | None:
        f = self.data.get("force")
        if f is None:
            return None
        return np.asarray(f, dtype=float)

    @force.setter
    def force(self, value: np.ndarray) -> None:
        self.data["force"] = value

    @property
    def displacement(self) -> np.ndarray | None:
        d = self.data.get("displacement")
        if d is None:
            return None
        return np.asarray(d, dtype=float)

    @displacement.setter
    def displacement(self, value: np.ndarray) -> None:
        self.data["displacement"] = value

    def as_dict(self) -> dict[str, Any]:
        out = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SiteProperties:
        return cls(data=d.copy())
