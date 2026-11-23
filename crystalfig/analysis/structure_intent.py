"""Structure-level display classification, independent of rendering."""

from __future__ import annotations

import numpy as np

from crystalfig.analysis.coordination import CoordinationAnalyzer
from crystalfig.model.structure import CrystalStructure


class StructureIntentAnalyzer:
    def __init__(self, structure: CrystalStructure):
        self.structure = structure

    def classify(self, intent: str, bonds) -> tuple[str, float]:
        allowed = {"auto", "network", "coordination", "layered", "ionic", "minimal"}
        if intent not in allowed:
            raise ValueError(f"intent must be one of {sorted(allowed)}")
        if intent != "auto":
            return intent, 1.0
        coordination = np.zeros(self.structure.num_sites, dtype=int)
        contrasts = []
        for bond in bonds:
            coordination[bond.i] += 1
            coordination[bond.j] += 1
            first = self._electronegativity(self.structure.sites[bond.i].dominant_element)
            second = self._electronegativity(self.structure.sites[bond.j].dominant_element)
            if first is not None and second is not None:
                contrasts.append(abs(first - second))
        translations = self.periodic_translation_vectors(bonds)
        dimension = int(np.linalg.matrix_rank(translations)) if len(translations) else 0
        centers = self.automatic_center_candidates(bonds)
        if dimension == 2:
            return "layered", 0.84
        if contrasts and np.mean(contrasts) >= 0.75 and np.median(coordination) >= 5 and np.std(coordination) < 1.0:
            return "ionic", 0.82
        if centers:
            return "hybrid", 0.80
        if np.mean(coordination) >= 2.5:
            return "network", 0.72
        return "minimal", 0.58

    def periodic_translation_vectors(self, bonds) -> np.ndarray:
        adjacency = [[] for _ in self.structure.sites]
        for bond in bonds:
            image = np.asarray(bond.jimage, dtype=int)
            adjacency[bond.i].append((bond.j, image))
            adjacency[bond.j].append((bond.i, -image))
        offsets: dict[int, np.ndarray] = {}
        translations = []
        for root in range(len(adjacency)):
            if root in offsets:
                continue
            offsets[root] = np.zeros(3, dtype=int)
            pending = [root]
            while pending:
                site = pending.pop()
                for neighbor, image in adjacency[site]:
                    candidate = offsets[site] + image
                    if neighbor not in offsets:
                        offsets[neighbor] = candidate
                        pending.append(neighbor)
                    else:
                        translation = candidate - offsets[neighbor]
                        if np.any(translation):
                            translations.append(translation)
        independent = []
        for translation in translations:
            if np.linalg.matrix_rank(np.asarray(independent + [translation])) > len(independent):
                independent.append(translation)
        if not independent:
            return np.empty((0, 3))
        return np.asarray(independent, dtype=float) @ self.structure.lattice.matrix

    def automatic_center_candidates(self, bonds) -> list[int]:
        return CoordinationAnalyzer().automatic_center_candidates(self.structure, bonds)

    @staticmethod
    def _electronegativity(element: str) -> float | None:
        from pymatgen.core import Element
        return Element(element).X
