"""Compose the wall: every tile a different algorithm, all visible at once."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from effects import EFFECTS
from fetch import fetch
from layout import organic_crop, photo_detail, split_canvas

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def label_tile(draw: ImageDraw.ImageDraw, rect, text: str, credit: str) -> None:
    """Caption in the corner: algorithm name, plus photographer in smaller type."""
    size = max(11, min(15, rect.w // 26))
    font = load_font(size)
    small = load_font(max(9, size - 3))

    pad = 7
    x = rect.x + pad
    y = rect.y + rect.h - pad - size - (size - 2)

    box = draw.textbbox((x, y), text, font=font)
    plate = [box[0] - 5, box[1] - 4, box[2] + 6, y + size + (size - 2) + 3]
    draw.rectangle(plate, fill=(0, 0, 0, 190))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    draw.text((x, y + size + 2), credit, font=small, fill=(165, 165, 165))


def build(
    tiles: int = 40,
    width: int = 2600,
    height: int = 1700,
    seed: int = 11,
    gutter: int = 6,
    labels: bool = True,
    out_name: str = "wall.png",
) -> Path:
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    names = list(EFFECTS.keys())
    rng.shuffle(names)

    print(f"fetching photos ...")
    photos = fetch(tiles, seed=seed)
    if not photos:
        raise SystemExit("no photos could be fetched")

    rects = split_canvas(width, height, tiles, rng)
    print(f"laid out {len(rects)} tiles across {width}×{height}")

    # Give the biggest tiles the most detailed photos. A featureless sky
    # dithers into a dead grey field, and it is most conspicuous at large size.
    photos.sort(key=lambda p: photo_detail(p["path"]), reverse=True)
    order = sorted(range(len(rects)), key=lambda i: rects[i].area, reverse=True)
    assignment = [0] * len(rects)
    for slot, idx in enumerate(order):
        assignment[idx] = slot % len(photos)

    canvas = Image.new("RGB", (width, height), (10, 10, 12))
    started = time.time()

    for i, rect in enumerate(rects):
        name = names[i % len(names)]
        fn, _category = EFFECTS[name]
        photo = photos[assignment[i]]

        inner_w = max(8, rect.w - gutter)
        inner_h = max(8, rect.h - gutter)

        src = organic_crop(photo["path"], inner_w, inner_h, rng)
        try:
            result = fn(src, nprng)
        except Exception as exc:
            print(f"  ! {name}: {exc}")
            result = src

        arr = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(arr), (rect.x + gutter // 2, rect.y + gutter // 2))
        print(f"  [{i + 1:>2}/{len(rects)}] {name:<28} {inner_w}×{inner_h}")

    if labels:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for i, rect in enumerate(rects):
            label_tile(
                draw,
                rect,
                names[i % len(names)],
                f"photo · {photos[assignment[i]]['author']}",
            )
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / out_name
    canvas.save(path, quality=95)
    print(f"\nrendered in {time.time() - started:.1f}s -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a wall of dithering and shader effects.")
    ap.add_argument("--tiles", type=int, default=40)
    ap.add_argument("--width", type=int, default=2600)
    ap.add_argument("--height", type=int, default=1700)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--out", default="wall.png")
    args = ap.parse_args()

    build(
        tiles=args.tiles,
        width=args.width,
        height=args.height,
        seed=args.seed,
        labels=not args.no_labels,
        out_name=args.out,
    )


if __name__ == "__main__":
    main()
