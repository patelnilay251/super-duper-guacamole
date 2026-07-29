"""Genre presets.

The default wall is deliberately heterogeneous -- it exists to compare
algorithms. A genre does the opposite: it constrains palette, screen and finish
so every tile reads as one process, and the variation between tiles becomes
variation *within* a style rather than between styles.

Each genre supplies its own labelled effect list, a canvas colour, and an
optional finish applied to every tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from effects import (
    dither, dither_ordered,
    bayer, bloom, chromatic, crosshatch, crt, crystallize, dot_screen, duotone,
    edges, error_diffuse, gradient_map, grain, halftone, kuwahara, luma,
    misregister, normalize_tone, ordered_dither, palette_array, paper_finish,
    phosphor, pixel_sort, pixelate, posterize, riso, scanline_slice, xerox,
)


@dataclass
class Genre:
    name: str
    blurb: str
    background: tuple
    effects: dict[str, Callable] = field(default_factory=dict)
    finish: Callable | None = None
    label_ink: tuple = (255, 255, 255)


def _ink(rgb, dark, light):
    """Force a monochrome result onto a two-colour ink/stock pair."""
    return duotone(rgb, dark, light)


# ---------------------------------------------------------------------------
# newsprint -- black ink on cheap warm stock, everything screened
# ---------------------------------------------------------------------------

INK, STOCK = (18, 16, 14), (247, 241, 226)

NEWSPRINT = Genre(
    name="newsprint",
    blurb="black ink on warm stock; rotated screens and diffusion",
    background=(228, 220, 202),
    label_ink=(40, 34, 28),
    finish=lambda img, rng: paper_finish(img, rng, (252, 246, 232)),
    effects={
        "screen 60°": lambda img, rng: _ink(halftone(normalize_tone(img), 6, 1.05), INK, STOCK),
        "screen 45°": lambda img, rng: _ink(halftone(normalize_tone(img), 7, 0.79), INK, STOCK),
        "screen 15°": lambda img, rng: _ink(halftone(normalize_tone(img), 8, 0.26), INK, STOCK),
        "coarse screen": lambda img, rng: _ink(halftone(normalize_tone(img), 12, 0.79), INK, STOCK),
        "fine screen": lambda img, rng: _ink(halftone(normalize_tone(img), 4, 0.52), INK, STOCK),
        "engraving": lambda img, rng: _ink(crosshatch(normalize_tone(img), 7), INK, STOCK),
        "engraving fine": lambda img, rng: _ink(crosshatch(normalize_tone(img), 4), INK, STOCK),
        "floyd–steinberg": lambda img, rng: _ink(dither(img, "mono", "floyd_steinberg"), INK, STOCK),
        "atkinson": lambda img, rng: _ink(dither(img, "mono", "atkinson"), INK, STOCK),
        "stucki": lambda img, rng: _ink(dither(img, "mono", "stucki"), INK, STOCK),
        "bayer 8×8": lambda img, rng: _ink(dither_ordered(img, "mono", bayer(8)), INK, STOCK),
        "line cut": lambda img, rng: _ink(edges(img) > 0.35, INK, STOCK),
        "overprint": lambda img, rng: _ink(halftone(normalize_tone(kuwahara(img, 5)), 6, 0.79), INK, STOCK),
    },
)


# ---------------------------------------------------------------------------
# handheld -- four greens, hard quantisation, chunky pixels
# ---------------------------------------------------------------------------

HANDHELD = Genre(
    name="handheld",
    blurb="four-tone LCD greens, chunky pixels, hard quantisation",
    background=(24, 42, 20),
    label_ink=(200, 230, 150),
    effects={
        "floyd": lambda img, rng: dither(img, "gameboy", "floyd_steinberg"),
        "atkinson": lambda img, rng: dither(img, "gameboy", "atkinson"),
        "burkes": lambda img, rng: dither(img, "gameboy", "burkes"),
        "sierra lite": lambda img, rng: dither(img, "gameboy", "sierra_lite"),
        "bayer 2×2": lambda img, rng: dither_ordered(img, "gameboy", bayer(2)),
        "bayer 4×4": lambda img, rng: dither_ordered(img, "gameboy", bayer(4)),
        "bayer 8×8": lambda img, rng: dither_ordered(img, "gameboy", bayer(8)),
        "8px sprite": lambda img, rng: dither_ordered(pixelate(img, 8), "gameboy", bayer(4)),
        "5px sprite": lambda img, rng: dither(pixelate(img, 5), "gameboy", "floyd_steinberg"),
        "12px sprite": lambda img, rng: dither_ordered(pixelate(img, 12), "gameboy", bayer(2)),
        "flat 4-tone": lambda img, rng: dither_ordered(kuwahara(img, 6), "gameboy", bayer(8)),
        "screened": lambda img, rng: dither_ordered(halftone(normalize_tone(img), 5), "gameboy", bayer(4)),
    },
)


# ---------------------------------------------------------------------------
# risograph -- two spot inks, visible screens, imperfect registration
# ---------------------------------------------------------------------------

RISO_INKS = {
    "fluoro pink · blue": [(255, 72, 176), (0, 120, 191)],
    "yellow · blue": [(255, 232, 0), (0, 120, 191)],
    "red · teal": [(255, 102, 94), (0, 169, 157)],
    "purple · green": [(118, 82, 205), (0, 169, 92)],
    "orange · navy": [(255, 108, 47), (26, 44, 124)],
    "pink · yellow": [(255, 72, 176), (255, 232, 0)],
}


def _riso_variant(inks, cell, slip):
    return lambda img, rng: riso(img, inks, rng, cell=cell, slip=slip)


RISOGRAPH = Genre(
    name="risograph",
    blurb="two spot inks, visible screens, deliberate misregistration",
    background=(238, 233, 222),
    label_ink=(40, 34, 40),
    finish=lambda img, rng: paper_finish(img, rng, (250, 247, 238)),
    effects={
        **{f"{name}": _riso_variant(inks, 5, 2) for name, inks in RISO_INKS.items()},
        **{f"{name} · coarse": _riso_variant(inks, 9, 3) for name, inks in list(RISO_INKS.items())[:3]},
        **{f"{name} · tight": _riso_variant(inks, 4, 0) for name, inks in list(RISO_INKS.items())[3:]},
    },
)


# ---------------------------------------------------------------------------
# nightcity -- neon duotones, signal damage, glow
# ---------------------------------------------------------------------------

def _neon(dark, light):
    return lambda img, rng: np.clip(bloom(duotone(normalize_tone(img), dark, light), 0.78, 0.35), 0, 1)


NIGHTCITY = Genre(
    name="nightcity",
    blurb="neon duotones, signal damage, bloom",
    background=(8, 6, 16),
    effects={
        "magenta/cyan": _neon((12, 4, 40), (255, 88, 220)),
        "cyan/violet": _neon((6, 10, 44), (96, 240, 255)),
        "acid/indigo": _neon((14, 6, 38), (198, 255, 64)),
        "ember/plum": _neon((20, 4, 28), (255, 138, 72)),
        "chromatic": lambda img, rng: chromatic(normalize_tone(img), 0.028),
        "scanlines": lambda img, rng: crt(normalize_tone(img)),
        "datamosh": lambda img, rng: scanline_slice(normalize_tone(img), rng),
        "pixel sort": lambda img, rng: pixel_sort(normalize_tone(img), 0.5),
        "sort + shift": lambda img, rng: chromatic(pixel_sort(normalize_tone(img), 0.45), 0.02),
        "bloom": lambda img, rng: bloom(normalize_tone(img), 0.58, 1.1),
        "crush": lambda img, rng: posterize(chromatic(normalize_tone(img), 0.02), 4),
        "misregister": lambda img, rng: misregister(normalize_tone(img), rng, 6),
        "glitch stack": lambda img, rng: scanline_slice(chromatic(pixel_sort(normalize_tone(img), 0.5), 0.03), rng),
    },
)


# ---------------------------------------------------------------------------
# blueprint -- cyanotype and drafting
# ---------------------------------------------------------------------------

BLUE_DARK, BLUE_LIGHT = (8, 32, 78), (214, 236, 252)

BLUEPRINT = Genre(
    name="blueprint",
    blurb="cyanotype stock, drafting lines, edge extraction",
    background=(10, 38, 88),
    label_ink=(220, 238, 255),
    effects={
        "edges": lambda img, rng: _ink(edges(img), BLUE_DARK, BLUE_LIGHT),
        "edges · hard": lambda img, rng: _ink((edges(img) > 0.3).astype(float), BLUE_DARK, BLUE_LIGHT),
        "hatching": lambda img, rng: _ink(1 - crosshatch(normalize_tone(img), 6), BLUE_DARK, BLUE_LIGHT),
        "hatching fine": lambda img, rng: _ink(1 - crosshatch(normalize_tone(img), 4), BLUE_DARK, BLUE_LIGHT),
        "cyanotype floyd": lambda img, rng: dither(img, "cyanotype", "floyd_steinberg"),
        "cyanotype stucki": lambda img, rng: dither(img, "cyanotype", "stucki"),
        "cyanotype bayer": lambda img, rng: dither_ordered(img, "cyanotype", bayer(8)),
        "negative screen": lambda img, rng: _ink(1 - halftone(normalize_tone(img), 6, 0.79), BLUE_DARK, BLUE_LIGHT),
        "contour": lambda img, rng: _ink(posterize(normalize_tone(img), 5), BLUE_DARK, BLUE_LIGHT),
        "wash": lambda img, rng: duotone(normalize_tone(kuwahara(img, 6)), BLUE_DARK, BLUE_LIGHT),
    },
)


# ---------------------------------------------------------------------------
# thermal -- false-colour heat ramps
# ---------------------------------------------------------------------------

RAMPS = {
    "ironbow": [(0, 0, 12), (46, 8, 92), (148, 20, 108), (232, 92, 44), (255, 190, 40), (255, 252, 220)],
    "inferno": [(2, 2, 10), (60, 12, 90), (170, 40, 70), (240, 110, 30), (250, 220, 90)],
    "arctic": [(4, 8, 28), (18, 70, 130), (60, 160, 200), (170, 226, 240), (250, 252, 255)],
    "viridis-ish": [(38, 12, 68), (32, 90, 132), (34, 148, 116), (140, 200, 68), (250, 232, 60)],
    "magma": [(0, 0, 8), (70, 16, 90), (168, 42, 90), (244, 106, 62), (252, 226, 176)],
    "sodium": [(6, 4, 2), (72, 26, 4), (176, 76, 8), (244, 160, 28), (255, 240, 190)],
}


def _ramp(stops, dither=None):
    def run(img, rng):
        out = gradient_map(img, stops)
        if dither == "bayer":
            out = ordered_dither(out, np.array(stops, dtype=float) / 255.0, bayer(8))
        elif dither == "floyd":
            out = error_diffuse(out, np.array(stops, dtype=float) / 255.0, "floyd_steinberg")
        return out
    return run


THERMAL = Genre(
    name="thermal",
    blurb="false-colour heat ramps, banded and dithered",
    background=(6, 4, 10),
    effects={
        **{name: _ramp(stops) for name, stops in RAMPS.items()},
        **{f"{name} · dithered": _ramp(stops, "floyd") for name, stops in list(RAMPS.items())[:3]},
        **{f"{name} · banded": _ramp(stops, "bayer") for name, stops in list(RAMPS.items())[3:]},
        "ironbow · posterised": lambda img, rng: posterize(gradient_map(img, RAMPS["ironbow"]), 6),
    },
)


# ---------------------------------------------------------------------------
# phosphor -- terminal CRTs
# ---------------------------------------------------------------------------

PHOSPHOR = Genre(
    name="phosphor",
    blurb="monochrome terminal tubes: green, amber, ice",
    background=(4, 8, 6),
    label_ink=(150, 255, 180),
    effects={
        "p1 green": lambda img, rng: phosphor(img, (110, 255, 140)),
        "p3 amber": lambda img, rng: phosphor(img, (255, 182, 66)),
        "ice": lambda img, rng: phosphor(img, (150, 220, 255)),
        "green · dithered": lambda img, rng: phosphor(dither_ordered(img, "mono", bayer(4)), (110, 255, 140)),
        "amber · dithered": lambda img, rng: phosphor(dither(img, "mono", "atkinson"), (255, 182, 66)),
        "green · pixelated": lambda img, rng: phosphor(pixelate(img, 6), (120, 255, 150)),
        "amber · coarse": lambda img, rng: phosphor(pixelate(img, 10), (255, 176, 60)),
        "green · screened": lambda img, rng: phosphor(halftone(normalize_tone(img), 5), (110, 255, 140)),
        "ice · edges": lambda img, rng: phosphor(edges(img), (150, 220, 255)),
        "burn-in": lambda img, rng: np.clip(phosphor(img, (120, 255, 150)) + bloom(normalize_tone(img), 0.7, 0.5) * 0.25, 0, 1),
    },
)


# ---------------------------------------------------------------------------
# xerox -- generation-loss photocopies
# ---------------------------------------------------------------------------

XEROX = Genre(
    name="xerox",
    blurb="crushed tone, toner speckle, generation loss",
    background=(196, 194, 188),
    label_ink=(30, 30, 32),
    finish=lambda img, rng: grain(img, rng, 0.035),
    effects={
        "1st gen": lambda img, rng: xerox(img, rng, 0.50),
        "2nd gen": lambda img, rng: xerox(xerox(img, rng, 0.50), rng, 0.46),
        "3rd gen": lambda img, rng: xerox(xerox(xerox(img, rng, 0.52), rng, 0.48), rng, 0.44),
        "overexposed": lambda img, rng: xerox(img, rng, 0.66),
        "underexposed": lambda img, rng: xerox(img, rng, 0.34),
        "screened copy": lambda img, rng: xerox(halftone(normalize_tone(img), 6), rng, 0.5),
        "hatched copy": lambda img, rng: xerox(crosshatch(normalize_tone(img), 5), rng, 0.5),
        "dithered copy": lambda img, rng: xerox(dither(img, "mono", "atkinson"), rng, 0.5),
        "skewed": lambda img, rng: xerox(misregister(img, rng, 4), rng, 0.5),
        "edge copy": lambda img, rng: xerox(1 - edges(img), rng, 0.55),
    },
)


GENRES: dict[str, Genre] = {
    g.name: g
    for g in (NEWSPRINT, HANDHELD, RISOGRAPH, NIGHTCITY, BLUEPRINT, THERMAL, PHOSPHOR, XEROX)
}
