"""Organic tiling.

A uniform grid reads as a contact sheet. To make the wall read as a composition
instead, two things vary: the tile shapes (recursive splitting rather than a
grid) and the framing inside each tile (every photo is cropped at its own zoom
and position, so no two tiles share a viewpoint).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h


def split_canvas(
    width: int,
    height: int,
    count: int,
    rng: random.Random,
    min_side: int = 150,
    balance: float = 0.62,
) -> list[Rect]:
    """Recursively split a canvas into `count` rectangles of varied size.

    Larger rectangles are more likely to be chosen for splitting (`balance`
    controls how strongly), which keeps the spread of tile sizes wide without
    letting any one tile swallow the canvas.
    """
    rects = [Rect(0, 0, width, height)]

    while len(rects) < count:
        candidates = [r for r in rects if r.w >= min_side * 2 or r.h >= min_side * 2]
        if not candidates:
            break

        weights = [r.area ** balance for r in candidates]
        target = rng.choices(candidates, weights=weights, k=1)[0]
        rects.remove(target)

        # Split across the longer axis so tiles trend toward usable aspects,
        # but allow the occasional cross-split for irregularity.
        vertical = target.aspect > 1.0
        if rng.random() < 0.18:
            vertical = not vertical
        if vertical and target.w < min_side * 2:
            vertical = False
        if not vertical and target.h < min_side * 2:
            vertical = True

        ratio = rng.uniform(0.36, 0.64)
        if vertical:
            cut = max(min_side, min(target.w - min_side, int(target.w * ratio)))
            rects.append(Rect(target.x, target.y, cut, target.h))
            rects.append(Rect(target.x + cut, target.y, target.w - cut, target.h))
        else:
            cut = max(min_side, min(target.h - min_side, int(target.h * ratio)))
            rects.append(Rect(target.x, target.y, target.w, cut))
            rects.append(Rect(target.x, target.y + cut, target.w, target.h - cut))

    # Repair pass: recursive splitting can leave full-height slivers, which read
    # as seams rather than tiles. Cut anything too elongated across its long axis.
    for _ in range(count):
        slivers = [r for r in rects if _too_thin(r, min_side)]
        if not slivers:
            break
        target = max(slivers, key=lambda r: r.area)
        rects.remove(target)
        if target.aspect < 1.0:  # tall and narrow -> cut horizontally
            cut = int(target.h * rng.uniform(0.42, 0.58))
            rects.append(Rect(target.x, target.y, target.w, cut))
            rects.append(Rect(target.x, target.y + cut, target.w, target.h - cut))
        else:                    # wide and short -> cut vertically
            cut = int(target.w * rng.uniform(0.42, 0.58))
            rects.append(Rect(target.x, target.y, cut, target.h))
            rects.append(Rect(target.x + cut, target.y, target.w - cut, target.h))

    return rects


MAX_ASPECT = 2.6


def _too_thin(r: Rect, min_side: int) -> bool:
    extreme = r.aspect > MAX_ASPECT or r.aspect < 1 / MAX_ASPECT
    splittable = (r.h >= min_side * 2) if r.aspect < 1.0 else (r.w >= min_side * 2)
    return extreme and splittable


def photo_detail(path) -> float:
    """Mean gradient energy of a whole photo -- used to rank photos by interest."""
    detail, _ = _detail_map(Image.open(path).convert("RGB"))
    return float(detail.mean())


SCORE_WIDTH = 220  # thumbnail width used for cheap detail scoring


def _detail_map(img: Image.Image) -> tuple[np.ndarray, float]:
    """Per-pixel detail proxy on a thumbnail: local gradient magnitude."""
    w, h = img.size
    thumb = img.resize((SCORE_WIDTH, max(1, int(h * SCORE_WIDTH / w))), Image.BILINEAR)
    g = np.asarray(thumb.convert("L"), dtype=np.float64) / 255.0
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1]))
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    return gx + gy, SCORE_WIDTH / w


def _candidate_box(src_w: int, src_h: int, aspect: float, zoom: float, rng: random.Random):
    if src_w / src_h > aspect:
        crop_h = src_h / zoom
        crop_w = crop_h * aspect
    else:
        crop_w = src_w / zoom
        crop_h = crop_w / aspect
    crop_w, crop_h = min(crop_w, src_w), min(crop_h, src_h)

    # Beta keeps the window off the extreme edges and slightly high in frame,
    # where photographic subjects usually sit.
    left = rng.betavariate(2.0, 2.0) * (src_w - crop_w)
    top = rng.betavariate(2.0, 2.6) * (src_h - crop_h)
    return left, top, crop_w, crop_h


def organic_crop(
    path,
    out_w: int,
    out_h: int,
    rng: random.Random,
    max_zoom: float = 2.3,
    candidates: int = 10,
) -> np.ndarray:
    """Crop to the tile's aspect at a varied zoom and position, seeking detail.

    Every tile gets its own framing, but a purely random window at high zoom
    regularly lands on blank sky or a plain wall -- which dithers into a flat,
    dead tile. So several candidate windows are proposed and the one with the
    most local gradient energy wins. Variety comes from the random proposals;
    the scoring only rejects the genuinely empty ones.
    """
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    aspect = out_w / out_h
    detail, scale = _detail_map(img)
    dh, dw = detail.shape

    best, best_score = None, -1.0
    for _ in range(candidates):
        left, top, crop_w, crop_h = _candidate_box(src_w, src_h, aspect, rng.uniform(1.0, max_zoom), rng)
        x0, y0 = int(left * scale), int(top * scale)
        x1, y1 = int((left + crop_w) * scale), int((top + crop_h) * scale)
        region = detail[max(0, y0) : min(dh, max(y0 + 1, y1)), max(0, x0) : min(dw, max(x0 + 1, x1))]
        score = float(region.mean()) if region.size else 0.0
        if score > best_score:
            best, best_score = (left, top, crop_w, crop_h), score

    left, top, crop_w, crop_h = best
    box = (int(left), int(top), int(left + crop_w), int(top + crop_h))
    tile = img.crop(box).resize((out_w, out_h), Image.LANCZOS)
    return np.asarray(tile, dtype=np.float64) / 255.0
