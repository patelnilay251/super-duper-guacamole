"""Differential test: GPU shaders against the NumPy reference.

The two implementations will never match pixel for pixel -- different tone
paths, different sampling, floating point. So this does not compare pixels. It
compares the properties that actually break when a shader is wrong:

  structure   both downsampled to 32x32 and correlated. Collapses when the
              shader samples the wrong region, or renders a flat field.
  exposure    mean luminance. Catches clipping to black or white.
  ink         fraction of dark pixels. Catches a dither losing its tone curve.

Every bug found by eye in this project would have been caught here: the Bayer
matrix darkening, the Game Boy palette collapsing to a flat field, and
crystallize sampling outside its texture.

    python3 tools/diff_harness.py --url http://127.0.0.1:8123/diff.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ditherwall"))

from effects import (  # noqa: E402
    bayer, crosshatch, dither_ordered, duotone, edges, gradient_map, halftone,
    kuwahara, lay, luma, normalize_tone, palette_array, riso, srgb_to_linear,
    xerox,
)


def inked(grey: np.ndarray, stock: tuple, ink: tuple) -> np.ndarray:
    """Re-lay a black-on-white screen in the given ink, the way the shader does.

    The screens return a linear-light composite of ink over paper, so coverage
    comes back out of the grey exactly. Routing it through `duotone` instead
    would apply that effect's contrast curve, which the shader does not.
    """
    return lay(stock, ink, 1.0 - srgb_to_linear(grey[..., 0]))

SIZE = 256

# Tolerances. Loose on absolutes, tight on structure: the failure mode that
# matters is a tile that stops resembling its source, not a few percent of tone.
MIN_STRUCTURE = 0.70
STOCHASTIC = {"xerox"}          # independent noise; judged on a looser bar
MIN_STRUCTURE_STOCHASTIC = 0.45
# Set from what the two implementations actually achieve when both are correct
# (worst observed: 0.050 exposure, 0.072 ink), with roughly 2x headroom. The
# original values were guesses made before they had ever agreed, and were loose
# enough that moving Python to linear light -- a 38x jump in exposure error on
# the mono dither -- still passed. A tolerance that cannot fail is not a test.
MAX_EXPOSURE_DELTA = 0.10
MAX_INK_DELTA = 0.13


def rgbf(*c):
    return [v / 255 for v in c]


PALETTES_JS = {
    "mono": [rgbf(0, 0, 0), rgbf(255, 255, 255)],
    "gameboy": [rgbf(15, 56, 15), rgbf(48, 98, 48), rgbf(139, 172, 15), rgbf(155, 188, 15)],
    "cyanotype": [rgbf(8, 22, 48), rgbf(24, 62, 110), rgbf(68, 124, 176), rgbf(150, 196, 224), rgbf(235, 245, 252)],
    "ember": [rgbf(20, 12, 28), rgbf(94, 30, 40), rgbf(196, 84, 46), rgbf(240, 176, 96), rgbf(255, 240, 205)],
}

INK, STOCK = rgbf(18, 16, 14), rgbf(247, 241, 226)
IRONBOW = [rgbf(0, 0, 12), rgbf(46, 8, 92), rgbf(148, 20, 108),
           rgbf(232, 92, 44), rgbf(255, 190, 40), rgbf(255, 252, 220)]


# name -> (gpu spec fragment, python reference callable)
CASES = {
    "bayer 8×8 · mono": (
        {"program": "ordered", "kind": 2, "palette": PALETTES_JS["mono"], "uniforms": {"uScale": 1}},
        lambda img: dither_ordered(img, "mono", bayer(8)),
    ),
    "bayer 4×4 · game boy": (
        {"program": "ordered", "kind": 1, "palette": PALETTES_JS["gameboy"], "uniforms": {"uScale": 1}},
        lambda img: dither_ordered(img, "gameboy", bayer(4)),
    ),
    "bayer 8×8 · cyanotype": (
        {"program": "ordered", "kind": 2, "palette": PALETTES_JS["cyanotype"], "uniforms": {"uScale": 1}},
        lambda img: dither_ordered(img, "cyanotype", bayer(8)),
    ),
    "bayer 2×2 · ember": (
        {"program": "ordered", "kind": 0, "palette": PALETTES_JS["ember"], "uniforms": {"uScale": 1}},
        lambda img: dither_ordered(img, "ember", bayer(2)),
    ),
    "halftone": (
        {"program": "halftone", "uniforms": {"uCell": 7, "uInk": INK, "uStock": STOCK, "uDotGain": 0.35}},
        lambda img: inked(halftone(normalize_tone(img), 7, 0.5, gain=0.35), (247, 241, 226), (18, 16, 14)),
    ),
    "gradient · ironbow": (
        {"program": "gradient", "uniforms": {"uRamp": IRONBOW}},
        lambda img: gradient_map(img, [(0, 0, 12), (46, 8, 92), (148, 20, 108),
                                       (232, 92, 44), (255, 190, 40), (255, 252, 220)]),
    ),
    "kuwahara": (
        {"program": "kuwahara", "uniforms": {"uRadius": 5}},
        lambda img: kuwahara(normalize_tone(img), 5),
    ),
    "edges": (
        {"program": "edges", "uniforms": {"uGain": 0.85}},
        lambda img: edges(normalize_tone(img)),
    ),
    "crosshatch": (
        {"program": "crosshatch", "uniforms": {"uSpacing": 7, "uInk": INK, "uStock": STOCK, "uDotGain": 0.15}},
        lambda img: inked(crosshatch(normalize_tone(img), 7, gain=0.15), (247, 241, 226), (18, 16, 14)),
    ),
    "duotone": (
        {"program": "duotone", "uniforms": {"uDark": rgbf(16, 24, 52), "uLight": rgbf(236, 232, 220), "uLevels": 0}},
        lambda img: duotone(normalize_tone(img), (16, 24, 52), (236, 232, 220)),
    ),
    "xerox": (
        {"program": "xerox", "uniforms": {"uBias": 0.5}},
        lambda img: xerox(img, np.random.default_rng(0), 0.5),
    ),
    "riso": (
        {"program": "riso", "uniforms": {"uInkA": rgbf(255, 72, 176), "uInkB": rgbf(0, 120, 191),
                                         "uCell": 5, "uSlip": 0, "uDotGain": 0.5}},
        lambda img: riso(img, [(255, 72, 176), (0, 120, 191)], np.random.default_rng(0), cell=5, slip=0, gain=0.5),
    ),
}


# ---------------------------------------------------------------- metrics


def small_gray(arr: np.ndarray, n: int = 32) -> np.ndarray:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
    return np.asarray(img.convert("L").resize((n, n), Image.BOX), dtype=float) / 255.0


def structure(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of coarse structure. 1 is identical, 0 unrelated."""
    x, y = small_gray(a).ravel(), small_gray(b).ravel()
    x, y = x - x.mean(), y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 1e-9 else 0.0


def measure(a: np.ndarray, b: np.ndarray) -> dict:
    la, lb = luma(a), luma(b)
    return {
        "structure": structure(a, b),
        "exposure": abs(float(la.mean()) - float(lb.mean())),
        "ink": abs(float((la < 0.5).mean()) - float((lb < 0.5).mean())),
    }


# ------------------------------------------------------------------ driver


def reference(photo_path: Path, crop, fn) -> np.ndarray:
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    box = (int(crop[0] * w), int(crop[1] * h),
           int((crop[0] + crop[2]) * w), int((crop[1] + crop[3]) * h))
    tile = img.crop(box).resize((SIZE, SIZE), Image.LANCZOS)
    return fn(np.asarray(tile, dtype=np.float64) / 255.0)


def stats_for(photo_path: Path, crop) -> tuple:
    """Everything the shader needs that it cannot compute for itself."""
    ref_img = Image.open(photo_path).convert("RGB")
    w, h = ref_img.size
    box = (int(crop[0] * w), int(crop[1] * h), int((crop[0] + crop[2]) * w), int((crop[1] + crop[3]) * h))
    crop_arr = np.asarray(ref_img.crop(box), dtype=np.float64) / 255.0
    g = luma(crop_arr)
    lo, hi = float(np.percentile(g, 1.5)), float(np.percentile(g, 98.5))
    post = np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1)
    mean = float(np.clip(post.mean(), 0.02, 0.98))
    gamma = float(np.clip(np.log(0.48) / np.log(mean), 0.5, 2.0))

    # riso's second separation is stretched across the chroma range this crop
    # occupies. The shader cannot take percentiles, so it receives them -- the
    # same route uTone takes. Measured after the tone map and in linear light,
    # which is where the shader takes the difference.
    toned = np.power(np.clip((crop_arr - lo) / max(hi - lo, 1e-3), 0, 1), gamma)
    lin = srgb_to_linear(toned)
    ch = lin[..., 0] - lin[..., 2]
    chroma = [float(np.percentile(ch, 3)), float(np.percentile(ch, 97))]
    return [lo, hi], gamma, chroma


def run(url: str, photo_indices: list[int]) -> int:
    """Compare every case against the reference, over several photographs.

    Sweeping matters more than it looks. Two real divergences hid for months
    behind a single pinned photograph: duotone was missing the reference's 1.25
    contrast, and riso's chroma separation used a fixed affine mapping where the
    reference stretches by percentiles. Both are invisible on a photograph whose
    tone happens to suit them, and both blow out on one that does not -- riso's
    ink fraction was off by 0.76 on the third photograph tried.
    """
    from playwright.sync_api import sync_playwright

    manifest = json.loads((ROOT / "gpu" / "manifest.json").read_text())
    crop = [0.12, 0.10, 0.62, 0.62]

    # This sandbox ships a browser at a fixed path; a CI runner lets Playwright
    # manage its own. Fall back to Playwright's resolution when the former is absent.
    chromium = next(iter(sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))), None)
    launch = {"args": ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]}
    if chromium:
        launch["executable_path"] = str(chromium)

    worst: dict[str, dict] = {}
    failures = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch)
        page = browser.new_page()
        page.goto(url, wait_until="load", timeout=60_000)
        page.wait_for_function("window.__ready === true", timeout=60_000)

        for idx in photo_indices:
            rec = manifest["photos"][idx]
            photo_path = ROOT / "gpu" / rec["src"]
            tone, gamma, chroma = stats_for(photo_path, crop)

            for name, (spec, ref_fn) in CASES.items():
                payload = {**spec, "photo": rec["src"], "size": SIZE, "crop": crop,
                           "tone": tone, "gamma": gamma, "chroma": chroma}
                data_url = page.evaluate("s => window.renderCase(s)", payload)
                raw = base64.b64decode(data_url.split(",", 1)[1])
                gpu = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float64) / 255.0

                ref = np.clip(reference(photo_path, crop, ref_fn), 0, 1)
                m = measure(gpu, ref)
                m["photo"] = idx

                # Keep each case's worst photograph, so one bad pairing cannot
                # be averaged away by the others.
                prev = worst.get(name)
                if prev is None or m["structure"] < prev["structure"] \
                        or m["ink"] > prev["ink"] or m["exposure"] > prev["exposure"]:
                    worst[name] = m if prev is None else {
                        "structure": min(m["structure"], prev["structure"]),
                        "exposure": max(m["exposure"], prev["exposure"]),
                        "ink": max(m["ink"], prev["ink"]),
                        "photo": idx if m["structure"] < prev["structure"] else prev["photo"],
                    }

        browser.close()

    rows = []
    for name, m in worst.items():
        floor = MIN_STRUCTURE_STOCHASTIC if name.split(" ")[0] in STOCHASTIC else MIN_STRUCTURE
        ok = (m["structure"] >= floor and m["exposure"] <= MAX_EXPOSURE_DELTA
              and m["ink"] <= MAX_INK_DELTA)
        failures += 0 if ok else 1
        rows.append((name, m, ok))

    print(f"worst of {len(photo_indices)} photographs, per case")
    print(f"{'case':<24} {'structure':>9} {'exposure':>9} {'ink':>7}   verdict")
    print("-" * 62)
    for name, m, ok in rows:
        print(f"{name:<24} {m['structure']:>9.3f} {m['exposure']:>9.3f} "
              f"{m['ink']:>7.3f}   {'ok' if ok else 'FAIL'}")
    print("-" * 62)
    print(f"{len(rows) - failures}/{len(rows)} within tolerance "
          f"(structure ≥ {MIN_STRUCTURE}, exposure ≤ {MAX_EXPOSURE_DELTA}, ink ≤ {MAX_INK_DELTA})")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8123/diff.html")
    ap.add_argument("--photos", type=int, default=4,
                    help="how many photographs to sweep; one is not enough")
    a = ap.parse_args()
    sys.exit(run(a.url, list(range(a.photos))))
