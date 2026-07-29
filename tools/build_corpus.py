"""Build the web corpus: optimised photographs plus a manifest.

The GPU gallery reads photographs as WebGL textures, which means they must be
same-origin or CORS-enabled. Serving them from the site itself sidesteps the
question entirely -- and removes any runtime dependency on a third-party CDN or
its rate limits.

    python3 tools/build_corpus.py --count 36 --edge 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ditherwall"))

from fetch import fetch  # noqa: E402

OUT = ROOT / "gpu" / "photos"


def blue_noise_texture(size: int = 64) -> None:
    """Ship the blue-noise threshold matrix as a texture.

    Generating it in the shader is not practical -- it needs an FFT and a rank
    transform -- and it is only 4kB, so it is baked here instead.
    """
    import numpy as np

    from effects import blue_noise

    rng = np.random.default_rng(7)
    noise = blue_noise((size, size), rng)
    img = Image.fromarray((noise * 255).astype("uint8"), mode="L")
    img.save(ROOT / "gpu" / "bluenoise.png", optimize=True)


def tone_stats(img: Image.Image) -> dict:
    """Percentile range the shader needs, since it cannot compute percentiles.

    Mirrors normalize_tone's stretch. The gamma half of that function depends on
    the crop rather than the whole frame, so the shader derives it at runtime
    from coarse mip samples instead.
    """
    import numpy as np

    small = img.copy()
    small.thumbnail((256, 256), Image.BILINEAR)
    g = np.asarray(small.convert("L"), dtype=float) / 255.0
    lo, hi = np.percentile(g, 1.5), np.percentile(g, 98.5)
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def build(count: int, edge: int, quality: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    photos = fetch(count)

    manifest = []
    total = 0
    for rec in photos:
        img = Image.open(rec["path"]).convert("RGB")
        img.thumbnail((edge, edge), Image.LANCZOS)

        name = f"{rec['id']}.jpg"
        path = OUT / name
        img.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
        size = path.stat().st_size
        total += size

        manifest.append({
            "src": f"photos/{name}",
            "w": img.width,
            "h": img.height,
            "author": rec["author"],
            "link": rec.get("url", ""),
            **tone_stats(img),
        })

    blue_noise_texture()
    (ROOT / "gpu" / "manifest.json").write_text(json.dumps({"photos": manifest}, indent=1))
    print(f"{len(manifest)} photos -> {OUT}")
    print(f"total {total / 1e6:.1f} MB, mean {total / max(1, len(manifest)) / 1e3:.0f} kB")
    print("blue-noise texture -> gpu/bluenoise.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=36)
    ap.add_argument("--edge", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=80)
    a = ap.parse_args()
    build(a.count, a.edge, a.quality)
