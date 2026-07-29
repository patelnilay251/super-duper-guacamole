"""Dithering and shader-style effects.

Every effect takes and returns a float RGB array in [0, 1] with shape (h, w, 3),
so they compose freely. The registry at the bottom is what the wall renders from.

Error diffusion is genuinely sequential, so those kernels run on Python lists --
counterintuitively faster than numpy here, because scalar indexing into an
ndarray costs more per access than plain float arithmetic.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance."""
    return rgb @ np.array([0.299, 0.587, 0.114])


def box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via summed-area accumulation along each axis."""
    if radius < 1:
        return img
    out = img
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        cum = np.cumsum(padded, axis=axis)
        zero = np.zeros_like(np.take(cum, [0], axis=axis))
        cum = np.concatenate([zero, cum], axis=axis)
        hi = np.take(cum, range(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
        lo = np.take(cum, range(0, n), axis=axis)
        out = (hi - lo) / (2 * radius + 1)
    return out


def gaussian_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Three box blurs approximate a Gaussian closely enough for this work."""
    r = max(1, radius // 3)
    for _ in range(3):
        img = box_blur(img, r)
    return img


def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = kx.T
    return convolve(gray, kx), convolve(gray, ky)


def convolve(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small-kernel 2D convolution by shifted accumulation."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(gray, dtype=float)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j]:
                out += kernel[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
    return out


def contrast(rgb: np.ndarray, amount: float) -> np.ndarray:
    return np.clip((rgb - 0.5) * amount + 0.5, 0, 1)


def normalize_tone(rgb: np.ndarray, target: float = 0.48) -> np.ndarray:
    """Stretch to full range, then gamma-shift the mean toward mid grey.

    Dithering is unforgiving about exposure: a bright photo quantised to two
    levels clips to near-solid white and the pattern disappears. Every photo
    therefore gets put on the same tonal footing before it is quantised, which
    is what makes the kernels comparable across tiles.
    """
    g = luma(rgb)
    lo, hi = np.percentile(g, 1.5), np.percentile(g, 98.5)
    out = np.clip((rgb - lo) / (hi - lo), 0, 1) if hi - lo > 1e-3 else np.clip(rgb, 0, 1)

    mean = float(luma(out).mean())
    if 0.02 < mean < 0.98:
        gamma = np.clip(np.log(target) / np.log(mean), 0.5, 2.0)
        out = out ** gamma
    return out


# --------------------------------------------------------------------------
# palettes
# --------------------------------------------------------------------------

PALETTES = {
    "mono": [(0, 0, 0), (255, 255, 255)],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "cga": [(0, 0, 0), (85, 255, 255), (255, 85, 255), (255, 255, 255)],
    "c64": [
        (0, 0, 0), (255, 255, 255), (136, 57, 50), (103, 182, 189),
        (139, 63, 150), (85, 160, 73), (64, 49, 141), (191, 206, 114),
    ],
    "ember": [(20, 12, 28), (94, 30, 40), (196, 84, 46), (240, 176, 96), (255, 240, 205)],
    "cyanotype": [(8, 22, 48), (24, 62, 110), (68, 124, 176), (150, 196, 224), (235, 245, 252)],
    "sepia": [(32, 22, 16), (94, 68, 46), (168, 130, 92), (226, 202, 168), (252, 244, 230)],
}


def palette_array(name: str) -> np.ndarray:
    return np.array(PALETTES[name], dtype=float) / 255.0


def nearest(pixel: tuple[float, float, float], pal: np.ndarray) -> np.ndarray:
    d = pal - np.array(pixel)
    return pal[int(np.argmin(np.einsum("ij,ij->i", d, d)))]


def quantize_to(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Nearest-palette-colour with no dithering, vectorised."""
    flat = rgb.reshape(-1, 3)
    d = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    return pal[np.argmin(d, axis=1)].reshape(rgb.shape)


# --------------------------------------------------------------------------
# error-diffusion dithering
# --------------------------------------------------------------------------

# (dx, dy, weight) with a shared divisor.
KERNELS = {
    "floyd_steinberg": ([(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16),
    "atkinson": ([(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8),
    "jarvis": (
        [(1, 0, 7), (2, 0, 5), (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
         (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)], 48),
    "stucki": (
        [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
         (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)], 42),
    "burkes": (
        [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)], 32),
    "sierra": (
        [(1, 0, 5), (2, 0, 3), (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
         (-1, 2, 2), (0, 2, 3), (1, 2, 2)], 32),
    "sierra_lite": ([(1, 0, 2), (-1, 1, 1), (0, 1, 1)], 4),
}


def error_diffuse(rgb: np.ndarray, pal: np.ndarray, kernel: str, serpentine: bool = True) -> np.ndarray:
    """Classic error-diffusion dithering against an arbitrary palette."""
    offsets, divisor = KERNELS[kernel]
    h, w = rgb.shape[:2]
    # Python lists beat ndarray scalar indexing for this access pattern.
    buf = rgb.astype(float).tolist()
    pal_list = [tuple(c) for c in pal]

    for y in range(h):
        row = range(w) if (not serpentine or y % 2 == 0) else range(w - 1, -1, -1)
        flip = serpentine and y % 2 == 1
        for x in row:
            old = buf[y][x]
            best, best_d = pal_list[0], 1e9
            for c in pal_list:
                d = (old[0] - c[0]) ** 2 + (old[1] - c[1]) ** 2 + (old[2] - c[2]) ** 2
                if d < best_d:
                    best, best_d = c, d
            buf[y][x] = list(best)
            er, eg, eb = old[0] - best[0], old[1] - best[1], old[2] - best[2]
            for dx, dy, wgt in offsets:
                sx = x - dx if flip else x + dx
                sy = y + dy
                if 0 <= sx < w and 0 <= sy < h:
                    f = wgt / divisor
                    t = buf[sy][sx]
                    t[0] += er * f
                    t[1] += eg * f
                    t[2] += eb * f
    return np.clip(np.array(buf), 0, 1)


# --------------------------------------------------------------------------
# ordered / threshold dithering
# --------------------------------------------------------------------------


def bayer(n: int) -> np.ndarray:
    """Recursive Bayer threshold matrix.

    Normalised as (m + 0.5) / n^2 rather than m / n^2 so the thresholds are
    symmetric about 0.5. Dividing without the half-step leaves the matrix mean
    below 0.5, which biases every comparison and darkens the whole image --
    badly at 2x2, where a 50% grey renders as 25% white.
    """
    m = np.array([[0.0]])
    size = 1
    while size < n:
        m = np.block([
            [4 * m, 4 * m + 2],
            [4 * m + 3, 4 * m + 1],
        ])
        size *= 2
    return (m + 0.5) / (size * size)


def blue_noise(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Approximate blue noise: high-pass white noise, then rank to uniform.

    Not void-and-cluster, but the spectral character is right and the visual
    result is the clean, grain-free texture blue noise is prized for.
    """
    h, w = shape
    noise = rng.random((h, w))
    f = np.fft.fftshift(np.fft.fft2(noise))
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy - cy, xx - cx)
    f *= radius / (radius.max() + 1e-9)          # suppress low frequencies
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(f)))
    ranks = filtered.argsort(axis=None).argsort()  # rank transform -> uniform
    return (ranks / ranks.size).reshape(h, w)


def ordered_dither(rgb: np.ndarray, pal: np.ndarray, thresh: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Tile a threshold matrix over the image and quantise with the offset."""
    h, w = rgb.shape[:2]
    tiled = np.tile(thresh, (h // thresh.shape[0] + 1, w // thresh.shape[1] + 1))[:h, :w]
    spread = strength / max(len(pal) - 1, 1)
    nudged = np.clip(rgb + (tiled[..., None] - 0.5) * spread, 0, 1)
    return quantize_to(nudged, pal)


# --------------------------------------------------------------------------
# halftone & line-based
# --------------------------------------------------------------------------


def halftone(rgb: np.ndarray, cell: int = 7, angle: float = 0.4) -> np.ndarray:
    """Rotated dot screen -- dot area tracks local darkness."""
    g = luma(rgb)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    ca, sa = np.cos(angle), np.sin(angle)
    u, v = xx * ca - yy * sa, xx * sa + yy * ca
    # distance from the centre of the nearest screen cell
    du = (u % cell) - cell / 2
    dv = (v % cell) - cell / 2
    dist = np.hypot(du, dv) / (cell / 2)
    coarse = box_blur(g[..., None], cell // 2)[..., 0]
    dots = (dist > (1.0 - coarse) ** 0.7).astype(float)
    return np.repeat(dots[..., None], 3, axis=2)


def crosshatch(rgb: np.ndarray, spacing: int = 6) -> np.ndarray:
    """Engraving-style hatching: darker regions accumulate more line sets."""
    g = box_blur(luma(rgb)[..., None], 2)[..., 0]
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    out = np.ones_like(g)
    for level, angle in enumerate((0.7, -0.7, 0.0, 1.57)):
        cut = 0.82 - level * 0.20
        proj = xx * np.cos(angle) - yy * np.sin(angle)
        lines = (proj % spacing) < 1.35
        out[np.logical_and(lines, g < cut)] = 0.0
    return np.repeat(out[..., None], 3, axis=2)


# --------------------------------------------------------------------------
# shader-style filters
# --------------------------------------------------------------------------


def kuwahara(rgb: np.ndarray, radius: int = 4) -> np.ndarray:
    """Painterly edge-preserving filter: take the mean of the flattest quadrant."""
    g = luma(rgb)
    mean = box_blur(rgb, radius)
    sq = box_blur((g ** 2)[..., None], radius)[..., 0]
    var = sq - box_blur(g[..., None], radius)[..., 0] ** 2

    best_var = None
    best_mean = None
    for dy in (-radius, radius):
        for dx in (-radius, radius):
            v = np.roll(np.roll(var, dy, axis=0), dx, axis=1)
            m = np.roll(np.roll(mean, dy, axis=0), dx, axis=1)
            if best_var is None:
                best_var, best_mean = v, m
            else:
                take = v < best_var
                best_var = np.where(take, v, best_var)
                best_mean = np.where(take[..., None], m, best_mean)
    return np.clip(best_mean, 0, 1)


def chromatic(rgb: np.ndarray, amount: float = 0.012) -> np.ndarray:
    """Radial lens dispersion -- channels scale slightly differently."""
    h, w = rgb.shape[:2]
    out = np.empty_like(rgb)
    for ch, scale in enumerate((1.0 + amount, 1.0, 1.0 - amount)):
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        sy = np.clip(((yy - h / 2) / scale + h / 2).astype(int), 0, h - 1)
        sx = np.clip(((xx - w / 2) / scale + w / 2).astype(int), 0, w - 1)
        out[..., ch] = rgb[sy, sx, ch]
    return out


def crt(rgb: np.ndarray) -> np.ndarray:
    """Scanlines plus an RGB aperture mask, the way a shadow-mask tube looks."""
    h, w = rgb.shape[:2]
    out = contrast(chromatic(rgb, 0.006), 1.15)
    scan = 0.72 + 0.28 * (np.arange(h) % 3 != 0)
    out = out * scan[:, None, None]
    mask = np.ones((1, w, 3))
    for ch in range(3):
        mask[0, ch::3, ch] = 1.35
    return np.clip(out * mask + 0.02, 0, 1)


def bloom(rgb: np.ndarray, threshold: float = 0.68, strength: float = 0.9) -> np.ndarray:
    bright = np.clip(rgb - threshold, 0, 1) / max(1e-6, 1 - threshold)
    return np.clip(rgb + gaussian_blur(bright, 14) * strength, 0, 1)


def edges(rgb: np.ndarray) -> np.ndarray:
    gx, gy = sobel(luma(rgb))
    mag = np.hypot(gx, gy)
    mag = mag / (np.percentile(mag, 99) + 1e-9)
    return np.repeat(np.clip(mag, 0, 1)[..., None], 3, axis=2)


def pixel_sort(rgb: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    """Glitch-art sorting: reorder contiguous bright spans by luminance."""
    out = rgb.copy()
    g = luma(rgb)
    h, w = g.shape
    for y in range(h):
        mask = g[y] > threshold
        x = 0
        while x < w:
            if mask[x]:
                end = x
                while end < w and mask[end]:
                    end += 1
                if end - x > 3:
                    span = out[y, x:end]
                    out[y, x:end] = span[np.argsort(luma(span))]
                x = end
            else:
                x += 1
    return out


def crystallize(rgb: np.ndarray, cell: int = 22, rng: np.random.Generator | None = None) -> np.ndarray:
    """Jittered-grid Voronoi: each cell takes the colour at its seed point."""
    rng = rng or np.random.default_rng(0)
    h, w = rgb.shape[:2]
    gy, gx = h // cell + 2, w // cell + 2
    jitter = rng.random((gy, gx, 2))
    seeds_y = (np.arange(gy)[:, None] + jitter[..., 0]) * cell
    seeds_x = (np.arange(gx)[None, :] + jitter[..., 1]) * cell

    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cy = np.clip((yy / cell).astype(int), 0, gy - 1)
    cx = np.clip((xx / cell).astype(int), 0, gx - 1)

    best_d = np.full((h, w), 1e18)
    best_y = np.zeros((h, w), dtype=int)
    best_x = np.zeros((h, w), dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny = np.clip(cy + dy, 0, gy - 1)
            nx = np.clip(cx + dx, 0, gx - 1)
            sy, sx = seeds_y[ny, nx], seeds_x[ny, nx]
            d = (sy - yy) ** 2 + (sx - xx) ** 2
            take = d < best_d
            best_d = np.where(take, d, best_d)
            best_y = np.where(take, np.clip(sy.astype(int), 0, h - 1), best_y)
            best_x = np.where(take, np.clip(sx.astype(int), 0, w - 1), best_x)
    return rgb[best_y, best_x]


def duotone(rgb: np.ndarray, dark: tuple, light: tuple) -> np.ndarray:
    g = contrast(luma(rgb)[..., None], 1.25)
    a = np.array(dark, dtype=float) / 255
    b = np.array(light, dtype=float) / 255
    return a + (b - a) * g


def posterize(rgb: np.ndarray, levels: int = 5) -> np.ndarray:
    return np.round(rgb * (levels - 1)) / (levels - 1)


def displace(rgb: np.ndarray, rng: np.random.Generator, amount: float = 18.0) -> np.ndarray:
    """Warp along a smooth noise field -- liquid, heat-haze distortion."""
    h, w = rgb.shape[:2]
    field = gaussian_blur(rng.random((h, w, 2)), 26)
    field = (field - field.mean()) / (field.std() + 1e-9)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sy = np.clip((yy + field[..., 0] * amount).astype(int), 0, h - 1)
    sx = np.clip((xx + field[..., 1] * amount).astype(int), 0, w - 1)
    return rgb[sy, sx]


def scanline_slice(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Horizontal datamosh -- rows shifted in blocks."""
    out = rgb.copy()
    h = rgb.shape[0]
    y = 0
    while y < h:
        band = rng.integers(4, 26)
        shift = int(rng.normal(0, 22))
        out[y : y + band] = np.roll(out[y : y + band], shift, axis=1)
        y += band
    return out


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def _ed(kernel: str, pal: str):
    return lambda img, rng: error_diffuse(normalize_tone(img), palette_array(pal), kernel)


def _ord(matrix, pal: str, strength: float = 1.0):
    def run(img, rng):
        m = matrix(img, rng) if callable(matrix) else matrix
        return ordered_dither(normalize_tone(img), palette_array(pal), m, strength)
    return run


EFFECTS: dict[str, tuple] = {
    # --- error diffusion, monochrome -------------------------------------
    "floyd–steinberg": (_ed("floyd_steinberg", "mono"), "diffusion"),
    "atkinson": (_ed("atkinson", "mono"), "diffusion"),
    "jarvis–judice–ninke": (_ed("jarvis", "mono"), "diffusion"),
    "stucki": (_ed("stucki", "mono"), "diffusion"),
    "burkes": (_ed("burkes", "mono"), "diffusion"),
    "sierra": (_ed("sierra", "mono"), "diffusion"),
    "sierra lite": (_ed("sierra_lite", "mono"), "diffusion"),
    # --- error diffusion, palettes ---------------------------------------
    "floyd–steinberg · game boy": (_ed("floyd_steinberg", "gameboy"), "diffusion"),
    "atkinson · ember": (_ed("atkinson", "ember"), "diffusion"),
    "stucki · cyanotype": (_ed("stucki", "cyanotype"), "diffusion"),
    "burkes · c64": (_ed("burkes", "c64"), "diffusion"),
    "sierra · sepia": (_ed("sierra", "sepia"), "diffusion"),
    "jarvis · cga": (_ed("jarvis", "cga"), "diffusion"),
    # --- ordered / threshold ---------------------------------------------
    "bayer 2×2": (_ord(bayer(2), "mono"), "ordered"),
    "bayer 4×4": (_ord(bayer(4), "mono"), "ordered"),
    "bayer 8×8": (_ord(bayer(8), "mono"), "ordered"),
    "bayer 8×8 · ember": (_ord(bayer(8), "ember"), "ordered"),
    "bayer 4×4 · game boy": (_ord(bayer(4), "gameboy"), "ordered"),
    "blue noise": (_ord(lambda img, rng: blue_noise((64, 64), rng), "mono"), "ordered"),
    "blue noise · cyanotype": (_ord(lambda img, rng: blue_noise((64, 64), rng), "cyanotype"), "ordered"),
    "white noise": (_ord(lambda img, rng: rng.random((64, 64)), "mono"), "ordered"),
    # --- line & screen ----------------------------------------------------
    "halftone": (lambda img, rng: halftone(normalize_tone(img)), "screen"),
    "halftone fine": (lambda img, rng: halftone(normalize_tone(img), cell=4, angle=0.9), "screen"),
    "crosshatch": (lambda img, rng: crosshatch(normalize_tone(img)), "screen"),
    "sobel edges": (lambda img, rng: edges(img), "screen"),
    # --- shader-style -----------------------------------------------------
    "kuwahara": (lambda img, rng: kuwahara(img, 5), "shader"),
    "kuwahara · heavy": (lambda img, rng: kuwahara(kuwahara(img, 4), 6), "shader"),
    "crt": (lambda img, rng: crt(img), "shader"),
    "chromatic aberration": (lambda img, rng: chromatic(img, 0.02), "shader"),
    "bloom": (lambda img, rng: bloom(img), "shader"),
    "crystallize": (lambda img, rng: crystallize(img, 20, rng), "shader"),
    "crystallize · fine": (lambda img, rng: crystallize(img, 9, rng), "shader"),
    "pixel sort": (lambda img, rng: pixel_sort(img, 0.52), "shader"),
    "displace": (lambda img, rng: displace(img, rng), "shader"),
    "datamosh": (lambda img, rng: scanline_slice(img, rng), "shader"),
    "posterize": (lambda img, rng: posterize(img, 5), "shader"),
    "duotone · ink": (lambda img, rng: duotone(img, (16, 24, 52), (236, 232, 220)), "shader"),
    "duotone · rust": (lambda img, rng: duotone(img, (28, 14, 18), (244, 186, 122)), "shader"),
    # --- composites -------------------------------------------------------
    "bloom → atkinson": (lambda img, rng: error_diffuse(normalize_tone(bloom(img, 0.75, 0.55)), palette_array("mono"), "atkinson"), "composite"),
    "kuwahara → bayer": (lambda img, rng: ordered_dither(normalize_tone(kuwahara(img, 5)), palette_array("ember"), bayer(4)), "composite"),
    "crt → halftone": (lambda img, rng: halftone(normalize_tone(crt(img)), cell=5), "composite"),
    "crystallize → floyd": (lambda img, rng: error_diffuse(normalize_tone(crystallize(img, 14, rng)), palette_array("c64"), "floyd_steinberg"), "composite"),
}
