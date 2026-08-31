"""Color palettes for publication-ready figures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColorPalette:
    """A named color palette for elements and figure accents."""

    name: str
    element_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    accents: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    def color(self, name: str, default: tuple[int, int, int] = (128, 128, 128)) -> tuple[int, int, int]:
        if name in self.element_colors:
            return self.element_colors[name]
        if name in self.accents:
            return self.accents[name]
        return default

    def rgb(self, name: str, default: tuple[int, int, int] = (128, 128, 128)) -> tuple[int, int, int]:
        return self.color(name, default)

    def hex(self, name: str, default: str = "#808080") -> str:
        rgb = self.color(name, (128, 128, 128))
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def normalized(self, name: str, default: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> tuple[float, float, float]:
        rgb = self.color(name, tuple(int(c * 255) for c in default))
        return tuple(c / 255.0 for c in rgb)

    def copy(self) -> ColorPalette:
        """Return a deep copy of the palette."""
        from copy import deepcopy

        return ColorPalette(
            name=self.name,
            element_colors=deepcopy(self.element_colors),
            accents=deepcopy(self.accents),
        )


def _mute(rgb: tuple[int, int, int], factor: float = 0.35, target: tuple[int, int, int] = (150, 150, 155)) -> tuple[int, int, int]:
    """Blend an RGB color toward a neutral gray to lower saturation."""
    return tuple(int(rgb[i] * (1.0 - factor) + target[i] * factor) for i in range(3))


def _desaturate_element(rgb: tuple[int, int, int], factor: float = 0.55) -> tuple[int, int, int]:
    """Desaturate an element color while preserving its hue identity.

    Uses HSL-style luminance-preserving desaturation toward the perceptual gray.
    """
    r, g, b = rgb
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return tuple(int(rgb[i] * (1.0 - factor) + gray * factor) for i in range(3))


def _publication_from_jmol(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Derive a publication-friendly color from a raw Jmol color.

    Lower saturation and mute toward the theme gray so the color harmonises
    with the hand-tuned publication palette.
    """
    return _mute(_desaturate_element(rgb, factor=0.45), factor=0.25)


# Base colors carry element identity; the muted palette blends them toward gray
# so the figure reads as a single publication theme rather than a Jmol screenshot.
_MUTED_BASE = {
    "H": (255, 255, 255), "He": (217, 255, 255),
    "Li": (204, 128, 255), "Be": (194, 255, 0),
    "B": (255, 181, 181), "C": (128, 128, 128),
    "N": (48, 80, 248), "O": (255, 13, 13),
    "F": (144, 224, 80), "Ne": (179, 227, 245),
    "Na": (171, 92, 242), "Mg": (138, 255, 0),
    "Al": (191, 166, 166), "Si": (240, 200, 160),
    "P": (255, 128, 0), "S": (255, 255, 48),
    "Cl": (31, 240, 31), "Ar": (128, 209, 227),
    "K": (143, 64, 212), "Ca": (61, 255, 0),
    "Sc": (230, 230, 230), "Ti": (191, 194, 199),
    "V": (166, 166, 171), "Cr": (138, 153, 199),
    "Mn": (156, 122, 199), "Fe": (224, 102, 51),
    "Co": (240, 144, 160), "Ni": (80, 208, 80),
    "Cu": (200, 128, 51), "Zn": (125, 128, 176),
    "Ga": (194, 143, 143), "Ge": (102, 143, 143),
    "As": (189, 128, 227), "Se": (255, 161, 0),
    "Br": (166, 41, 41), "Kr": (92, 184, 209),
    "Rb": (112, 46, 176), "Sr": (0, 255, 0),
    "Y": (148, 255, 255), "Zr": (148, 224, 224),
    "Nb": (115, 194, 201), "Mo": (84, 181, 181),
    "Tc": (59, 158, 158), "Ru": (36, 143, 143),
    "Rh": (10, 125, 140), "Pd": (0, 105, 133),
    "Ag": (192, 192, 192), "Cd": (255, 217, 143),
    "In": (166, 117, 115), "Sn": (102, 128, 128),
    "Sb": (158, 99, 181), "Te": (212, 122, 0),
    "I": (148, 0, 148), "Xe": (66, 158, 176),
    "Cs": (87, 23, 143), "Ba": (0, 201, 0),
    "La": (112, 212, 255), "Ce": (255, 255, 199),
    "Pb": (87, 89, 97),
}

MUTED = ColorPalette(
    name="muted",
    element_colors={k: _mute(v) for k, v in _MUTED_BASE.items()},
    accents={
        "primary": (44, 95, 138),
        "secondary": (196, 90, 74),
        "accent": (67, 147, 108),
        "amber": (217, 131, 36),
        "purple": (117, 112, 179),
        "gray": (140, 150, 160),
        "dark": (40, 44, 52),
        "light": (248, 249, 250),
    },
)

# Hand-tuned publication palette: keeps element identity but avoids neon / pure RGB.
# Colors are chosen to work together in multi-panel Nature-style figures.
_PUBLICATION_BASE: dict[str, tuple[int, int, int]] = {
    "H": (245, 245, 245),
    "C": (90, 90, 90),
    "N": (80, 110, 175),
    "O": (195, 75, 70),          # muted red, not #ff0d0d
    "F": (110, 170, 90),
    "Na": (160, 100, 195),
    "Mg": (135, 175, 90),
    "Al": (165, 145, 145),
    "Si": (195, 165, 130),
    "P": (195, 120, 60),
    "S": (195, 175, 75),         # muted yellow
    "Cl": (95, 200, 95),
    "K": (150, 90, 180),
    "Ca": (120, 185, 95),
    "Sc": (200, 200, 200),
    "Ti": (155, 160, 165),       # soft gray metal
    "V": (145, 145, 150),
    "Cr": (130, 140, 175),
    "Mn": (150, 125, 170),
    "Fe": (175, 95, 65),
    "Co": (185, 120, 135),
    "Ni": (105, 175, 105),
    "Cu": (175, 125, 75),
    "Zn": (130, 135, 175),       # muted blue-gray
    "Ga": (175, 130, 130),
    "Ge": (115, 145, 145),
    "As": (175, 125, 195),
    "Se": (195, 140, 70),
    "Br": (155, 80, 80),
    "Kr": (120, 170, 185),
    "Rb": (140, 80, 165),
    "Sr": (120, 180, 120),
    "Y": (135, 200, 200),
    "Zr": (140, 190, 190),
    "Nb": (120, 175, 180),
    "Mo": (110, 170, 170),
    "Tc": (95, 155, 155),
    "Ru": (80, 140, 140),
    "Rh": (70, 125, 135),
    "Pd": (60, 110, 130),
    "Ag": (180, 180, 180),
    "Cd": (195, 175, 135),
    "In": (165, 120, 120),
    "Sn": (115, 140, 140),
    "Sb": (150, 100, 165),
    "Te": (185, 120, 65),
    "I": (145, 70, 145),
    "Xe": (125, 155, 170),
    "Cs": (120, 70, 150),
    "Ba": (105, 165, 105),       # muted green, not neon
    "La": (130, 180, 210),
    "Ce": (210, 210, 170),
    "Pb": (110, 110, 115),
}

# Raw Jmol/CPK colors for elements H (Z=1) through Bi (Z=83).  These are used
# only to derive muted publication colors for elements not present above.
_JMOL_BASE: dict[str, tuple[int, int, int]] = {
    "H": (255, 255, 255), "He": (217, 255, 255), "Li": (204, 128, 255),
    "Be": (194, 255, 0), "B": (255, 181, 181), "C": (128, 128, 128),
    "N": (48, 80, 248), "O": (255, 13, 13), "F": (144, 224, 80),
    "Ne": (179, 227, 245), "Na": (171, 92, 242), "Mg": (138, 255, 0),
    "Al": (191, 166, 166), "Si": (240, 200, 160), "P": (255, 128, 0),
    "S": (255, 255, 48), "Cl": (31, 240, 31), "Ar": (128, 209, 227),
    "K": (143, 64, 212), "Ca": (61, 255, 0), "Sc": (230, 230, 230),
    "Ti": (191, 194, 199), "V": (166, 166, 171), "Cr": (138, 153, 199),
    "Mn": (156, 122, 199), "Fe": (224, 102, 51), "Co": (240, 144, 160),
    "Ni": (80, 208, 80), "Cu": (200, 128, 51), "Zn": (125, 128, 176),
    "Ga": (194, 143, 143), "Ge": (102, 143, 143), "As": (189, 128, 227),
    "Se": (255, 161, 0), "Br": (166, 41, 41), "Kr": (92, 184, 209),
    "Rb": (112, 46, 176), "Sr": (0, 255, 0), "Y": (148, 255, 255),
    "Zr": (148, 224, 224), "Nb": (115, 194, 201), "Mo": (84, 181, 181),
    "Tc": (59, 158, 158), "Ru": (36, 143, 143), "Rh": (10, 125, 140),
    "Pd": (0, 105, 133), "Ag": (192, 192, 192), "Cd": (255, 217, 143),
    "In": (166, 117, 115), "Sn": (102, 128, 128), "Sb": (158, 99, 181),
    "Te": (212, 122, 0), "I": (148, 0, 148), "Xe": (66, 158, 176),
    "Cs": (87, 23, 143), "Ba": (0, 201, 0), "La": (112, 212, 255),
    "Ce": (255, 255, 199), "Pr": (217, 255, 199), "Nd": (199, 255, 199),
    "Pm": (163, 255, 199), "Sm": (143, 255, 199), "Eu": (97, 255, 199),
    "Gd": (69, 255, 199), "Tb": (48, 255, 199), "Dy": (31, 255, 199),
    "Ho": (0, 255, 156), "Er": (0, 230, 117), "Tm": (0, 212, 82),
    "Yb": (0, 191, 56), "Lu": (0, 171, 36), "Hf": (77, 194, 255),
    "Ta": (77, 166, 255), "W": (33, 148, 214), "Re": (38, 125, 171),
    "Os": (38, 102, 150), "Ir": (23, 84, 135), "Pt": (208, 208, 224),
    "Au": (255, 209, 35), "Hg": (184, 184, 208), "Tl": (166, 84, 77),
    "Pb": (87, 89, 97), "Bi": (158, 79, 181),
}

# Fill any gaps in the hand-tuned publication palette with muted Jmol colors.
_PUBLICATION_BASE = {
    **_PUBLICATION_BASE,
    **{
        el: _publication_from_jmol(_JMOL_BASE[el])
        for el in _JMOL_BASE
        if el not in _PUBLICATION_BASE
    },
}

PUBLICATION = ColorPalette(
    name="publication",
    element_colors=_PUBLICATION_BASE.copy(),
    accents={
        "primary": (44, 95, 138),
        "secondary": (196, 90, 74),
        "accent": (67, 147, 108),
        "amber": (217, 131, 36),
        "purple": (117, 112, 179),
        "gray": (140, 150, 160),
        "dark": (40, 44, 52),
        "light": (248, 249, 250),
    },
)

# Colorblind-safe Okabe-Ito-like palette
OKABE_ITO = ColorPalette(
    name="okabe_ito",
    element_colors={},
    accents={
        "primary": (0, 114, 178),
        "secondary": (230, 159, 0),
        "accent": (0, 158, 115),
        "amber": (240, 228, 66),
        "purple": (204, 121, 167),
        "gray": (128, 128, 128),
        "dark": (0, 0, 0),
        "light": (255, 255, 255),
    },
)

# Monochrome palette
MONOCHROME = ColorPalette(
    name="monochrome",
    element_colors={},
    accents={
        "primary": (0, 0, 0),
        "secondary": (80, 80, 80),
        "accent": (150, 150, 150),
        "amber": (100, 100, 100),
        "purple": (60, 60, 60),
        "gray": (180, 180, 180),
        "dark": (0, 0, 0),
        "light": (255, 255, 255),
    },
)

_JMOL_OVERRIDES = {
    "O": (255, 13, 13), "N": (48, 80, 248), "C": (128, 128, 128),
    "H": (255, 255, 255), "S": (255, 255, 48), "P": (255, 128, 0),
    "Fe": (224, 102, 51), "Cu": (200, 128, 51), "Zn": (125, 128, 176),
    "Ti": (191, 194, 199), "Ba": (0, 201, 0), "Sr": (0, 255, 0),
    "Si": (240, 200, 160),
}

JMOL = ColorPalette(
    name="jmol",
    element_colors=_JMOL_OVERRIDES,
    accents=MUTED.accents.copy(),
)

_PALETTES: dict[str, ColorPalette] = {
    "muted": MUTED,
    "publication_muted": MUTED,
    "publication": PUBLICATION,
    "journal_clean": PUBLICATION,
    "okabe_ito": OKABE_ITO,
    "colorblind_safe": OKABE_ITO,
    "monochrome": MONOCHROME,
    "jmol": JMOL,
}


def get_palette(name: str) -> ColorPalette:
    if name not in _PALETTES:
        raise ValueError(f"Unknown palette '{name}'. Available: {list(_PALETTES.keys())}")
    return _PALETTES[name]


def list_palettes() -> list[str]:
    return list(_PALETTES.keys())
