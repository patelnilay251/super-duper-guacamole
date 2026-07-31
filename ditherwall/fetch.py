"""Fetch a corpus of real photographs to run the effect wall against.

Uses Picsum's catalogue, which serves Unsplash photographs and exposes stable
ids plus photographer attribution. Images are cached on disk so repeated runs
stay offline.

Museum and archive APIs were evaluated as alternatives -- the Met, Library of
Congress, Wikimedia Featured Pictures, NASA -- and are worse for this. They
photograph *objects*: you get mount board, paper edges, pencil annotations,
29000-pixel panoramas and legible text. A tight crop lands on cream card stock.
This wall wants full-bleed photographs at ordinary proportions, which is what
stock photography is and what an archive is not.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"
LIST_URL = "https://picsum.photos/v2/list?page={page}&limit=100"
IMAGE_URL = "https://picsum.photos/id/{id}/{w}/{h}"

# Long edge cached from the source. Tiles crop down from this, and focus mode
# shows a single photograph at most of the viewport, so it needs headroom.
FETCH_EDGE = 1440

# Mechanical filters only. Scoring candidates for how well they *dither* was
# tried and abandoned: structure retention through a halftone came back 0.98 to
# 1.00 across the whole corpus -- no discrimination -- and ranked the flattest
# images best, since a flat field is trivial to reproduce. Tonal spread and
# detail density do vary widely but do not separate good from bad on
# inspection: the lowest-spread photograph in the corpus is a figure on asphalt,
# which dithers beautifully, and the highest-detail one is branches against sky,
# which also does. What makes a photograph work here is graphic legibility under
# two-level reduction, and that is semantic, not statistical.
MIN_SOURCE_EDGE = 1440
ASPECT_RANGE = (0.62, 2.20)   # reject panoramas and extreme portraits


def catalogue(pages: int = 10) -> list[dict]:
    """Return photo metadata records: id, author, url, dimensions."""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / f"catalogue_{pages}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    records: list[dict] = []
    for page in range(1, pages + 1):
        with urllib.request.urlopen(LIST_URL.format(page=page), timeout=30) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        records.extend(batch)
    meta_path.write_text(json.dumps(records, indent=2))
    return records


def usable(records: list[dict]) -> list[dict]:
    """Records big enough to crop into, at proportions a tile can use."""
    out = []
    for r in records:
        w, h = r.get("width", 0), r.get("height", 0)
        if not w or not h:
            continue
        if max(w, h) < MIN_SOURCE_EDGE:
            continue
        if not ASPECT_RANGE[0] <= w / h <= ASPECT_RANGE[1]:
            continue
        out.append(r)
    return out


def fetch(count: int, seed: int = 7, edge: int = FETCH_EDGE) -> list[dict]:
    """Download `count` photos, returning records with a local `path` added.

    Photos are sampled deterministically from the catalogue so a given seed
    always produces the same corpus.

    The request preserves each photograph's own proportions. Asking Picsum for a
    fixed width and height makes it *crop* to that shape, which is what the
    corpus used to do -- every photograph arrived as 4:3 with its framing thrown
    away before the wall ever saw it.
    """
    import random

    pool = usable(catalogue())
    rng = random.Random(seed)
    picked = rng.sample(pool, min(count, len(pool)))

    out = []
    for rec in picked:
        path = CACHE / f"{rec['id']}_{edge}.jpg"
        if not path.exists():
            w = min(edge, rec["width"])
            h = round(w * rec["height"] / rec["width"])
            url = IMAGE_URL.format(id=rec["id"], w=w, h=h)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:  # a dead id shouldn't sink the whole run
                print(f"  skip id={rec['id']}: {exc}")
                continue
        out.append({**rec, "path": path})
    return out


if __name__ == "__main__":
    records = catalogue()
    keep = usable(records)
    print(f"catalogue: {len(records)} photos, {len(keep)} usable "
          f"(>= {MIN_SOURCE_EDGE}px, aspect {ASPECT_RANGE[0]}-{ASPECT_RANGE[1]})")
