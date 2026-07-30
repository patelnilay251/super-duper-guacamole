"""Does a dithered tile actually reflect the light its source did?

A dither is a physical claim: cover x% of the paper in ink and the tile, seen
from far enough away that the dots blur together, reflects what the original
reflected. Blurring is an average of *light*, so the claim can only hold if the
quantiser decides on light too. Deciding in sRGB -- the storage encoding, which
is roughly a square root -- makes midtones systematically too pale.

This measures the claim directly, with no reference implementation involved:
render a tile, average it in linear light, and compare against the source crop
averaged the same way. Small residual is correct; a large positive residual
means the dither is holding back ink it owes.

    python3 tools/tone_fidelity.py --url http://127.0.0.1:8123/diff.html
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

from effects import light, luma, normalize_tone  # noqa: E402

SIZE = 256
CROP = [0.12, 0.10, 0.62, 0.62]

# Worst observed once every process quantises on light is 0.020 (halftone, whose
# dot has a hard edge and so lands a little dark). Set with the same ~2.5x
# headroom the differential harness uses. For scale, the failures this replaced
# were +0.19 on the dithers and +0.38 on the hatch.
MAX_RESIDUAL = 0.05

# Only the processes that make a tonal claim: they render the photo as coverage
# of a fixed set of inks. Effects that recolour continuously have nothing to
# reconstruct and are not measured here.
CASES = {
    "bayer 8×8 · mono": {"program": "ordered", "kind": 2, "uniforms": {"uScale": 1},
                         "palette": [[0, 0, 0], [1, 1, 1]]},
    "bayer 4×4 · mono": {"program": "ordered", "kind": 1, "uniforms": {"uScale": 1},
                         "palette": [[0, 0, 0], [1, 1, 1]]},
    "bayer 2×2 · mono": {"program": "ordered", "kind": 0, "uniforms": {"uScale": 1},
                         "palette": [[0, 0, 0], [1, 1, 1]]},
    "halftone": {"program": "halftone",
                 "uniforms": {"uCell": 7, "uInk": [0, 0, 0], "uStock": [1, 1, 1]}},
    "crosshatch": {"program": "crosshatch",
                   "uniforms": {"uSpacing": 7, "uInk": [0, 0, 0], "uStock": [1, 1, 1]}},
}


def run(url: str, photo_index: int, tag: str) -> int:
    from playwright.sync_api import sync_playwright

    manifest = json.loads((ROOT / "gpu" / "manifest.json").read_text())
    rec = manifest["photos"][photo_index]
    photo_path = ROOT / "gpu" / rec["src"]

    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    box = (int(CROP[0] * w), int(CROP[1] * h),
           int((CROP[0] + CROP[2]) * w), int((CROP[1] + CROP[3]) * h))
    crop_arr = np.asarray(img.crop(box), dtype=np.float64) / 255.0

    # The wall's tone pass, replayed so the shader and this script agree on what
    # "the source" means -- the shader never sees the raw photo either.
    g = luma(crop_arr)
    lo, hi = float(np.percentile(g, 1.5)), float(np.percentile(g, 98.5))
    post = np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1)
    mean = float(np.clip(post.mean(), 0.02, 0.98))
    gamma = float(np.clip(np.log(0.48) / np.log(mean), 0.5, 2.0))

    target = float(light(normalize_tone(
        np.asarray(img.crop(box).resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float64) / 255.0
    )).mean())

    chromium = next(iter(sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))), None)
    launch = {"args": ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]}
    if chromium:
        launch["executable_path"] = str(chromium)

    rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch)
        page = browser.new_page()
        page.goto(url, wait_until="load", timeout=60_000)
        page.wait_for_function("window.__ready === true", timeout=60_000)

        for name, spec in CASES.items():
            payload = {**spec, "photo": rec["src"], "size": SIZE,
                       "crop": CROP, "tone": [lo, hi], "gamma": gamma}
            data_url = page.evaluate("s => window.renderCase(s)", payload)
            raw = base64.b64decode(data_url.split(",", 1)[1])
            out = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float64) / 255.0
            rows.append((name, float(light(out).mean())))

        browser.close()

    if tag:
        print(tag)
    print(f"source mean light = {target:.4f}")
    print(f"{'case':<20} {'rendered':>9} {'residual':>9}   verdict")
    print("-" * 51)
    worst, failures = 0.0, 0
    for name, got in rows:
        d = got - target
        worst = max(worst, abs(d))
        ok = abs(d) <= MAX_RESIDUAL
        failures += 0 if ok else 1
        print(f"{name:<20} {got:>9.4f} {d:>+9.4f}   {'ok' if ok else 'FAIL'}")
    print("-" * 51)
    print(f"{len(rows) - failures}/{len(rows)} within {MAX_RESIDUAL} "
          f"(worst |residual| = {worst:.4f})")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8123/diff.html")
    ap.add_argument("--photo", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sys.exit(run(a.url, a.photo, a.tag))
