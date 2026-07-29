"""Fetch a small corpus of real photographs to run the effect wall against.

Uses Picsum's catalogue, which serves Unsplash photographs and exposes stable
ids plus photographer attribution. Images are cached on disk so repeated runs
stay offline.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"
LIST_URL = "https://picsum.photos/v2/list?page={page}&limit=100"
IMAGE_URL = "https://picsum.photos/id/{id}/{w}/{h}"

# Long edge we cache at. Tiles are cropped down from this, so it needs enough
# headroom for a tight crop to still look sharp.
FETCH_W, FETCH_H = 1400, 1050


def catalogue(pages: int = 2) -> list[dict]:
    """Return photo metadata records: id, author, dimensions."""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "catalogue.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    records: list[dict] = []
    for page in range(1, pages + 1):
        with urllib.request.urlopen(LIST_URL.format(page=page), timeout=30) as resp:
            records.extend(json.loads(resp.read()))
    meta_path.write_text(json.dumps(records, indent=2))
    return records


def fetch(count: int, seed: int = 7) -> list[dict]:
    """Download `count` photos, returning records with a local `path` added.

    Photos are sampled deterministically from the catalogue so a given seed
    always produces the same wall.
    """
    import random

    records = catalogue()
    # Landscape-ish originals crop more gracefully into arbitrary tile shapes.
    usable = [r for r in records if r["width"] >= r["height"]]
    rng = random.Random(seed)
    picked = rng.sample(usable, min(count, len(usable)))

    out = []
    for rec in picked:
        path = CACHE / f"{rec['id']}.jpg"
        if not path.exists():
            url = IMAGE_URL.format(id=rec["id"], w=FETCH_W, h=FETCH_H)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:  # a dead id shouldn't sink the whole run
                print(f"  skip id={rec['id']}: {exc}")
                continue
        out.append({**rec, "path": path})
    return out


if __name__ == "__main__":
    got = fetch(44)
    print(f"cached {len(got)} photos in {CACHE}")
    for r in got[:5]:
        print(f"  {r['id']:>5}  {r['author']}")
