"""Live gallery server.

Serves a grid whose tiles each pull an independent random photograph and a
random effect, rendered on the server. No framework -- the standard library's
threading HTTP server is enough, and it keeps the project dependency-free
beyond numpy and Pillow.

    python3 web/server.py --port 8000

Rendering a tile costs anywhere from 20ms to about a second depending on the
effect, which is too slow to sit inside a request on a phone. A pool of worker
threads therefore renders tiles ahead of demand into a per-size cache, and
requests normally just pop a finished one.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ditherwall"))

from effects import EFFECTS                      # noqa: E402
from fetch import fetch                          # noqa: E402
from genres import GENRES                        # noqa: E402
from layout import organic_crop                  # noqa: E402

STATIC = ROOT / "static"


def _env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Requested sizes are rounded to this grid so the cache has few distinct keys.
SIZE_BUCKET = 80
# Defaults are deliberately conservative: a free container may get a fraction
# of a core, and every one of these knobs trades image size for CPU per tile.
MAX_EDGE = _env("DW_MAX_EDGE", 720)
CACHE_DEPTH = _env("DW_CACHE_DEPTH", 4)


def effect_table(genre: str | None) -> dict:
    """Name -> callable, for a genre or for the full cross-genre registry."""
    if genre and genre in GENRES:
        return dict(GENRES[genre].effects)
    if genre == "everything" or not genre:
        table = {name: fn for name, (fn, _cat) in EFFECTS.items()}
        for g in GENRES.values():
            table.update({f"{g.name} · {k}": fn for k, fn in g.effects.items()})
        return table
    return {name: fn for name, (fn, _cat) in EFFECTS.items()}


def bucket(value: int) -> int:
    value = max(SIZE_BUCKET, min(MAX_EDGE, int(value)))
    return int(round(value / SIZE_BUCKET) * SIZE_BUCKET)


class TileFactory:
    """Renders tiles, and keeps a shallow queue of pre-rendered ones per key."""

    def __init__(self, photos: list[dict], workers: int = 3):
        self.photos = photos
        self.cache: dict[tuple, deque] = defaultdict(deque)
        self.hot: set[tuple] = set()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        for i in range(workers):
            threading.Thread(target=self._worker, name=f"render-{i}", daemon=True).start()

    # -- rendering ---------------------------------------------------------

    def render(self, key: tuple) -> tuple[bytes, dict]:
        genre, w, h = key
        rng = random.Random()
        nprng = np.random.default_rng(rng.randrange(1 << 30))

        table = effect_table(genre)
        name = rng.choice(list(table))
        photo = rng.choice(self.photos)

        src = organic_crop(photo["path"], w, h, rng)
        try:
            out = table[name](src, nprng)
        except Exception:
            out = src

        preset = GENRES.get(genre)
        if preset and preset.finish is not None:
            out = preset.finish(np.clip(out, 0, 1), nprng)

        arr = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=False)
        meta = {"effect": name, "author": photo["author"], "photo_id": str(photo["id"])}
        return buf.getvalue(), meta

    # -- cache -------------------------------------------------------------

    def take(self, key: tuple) -> tuple[bytes, dict]:
        with self.lock:
            self.hot.add(key)
            queue = self.cache[key]
            if queue:
                return queue.popleft()
        return self.render(key)  # cold start: render inline just this once

    def _worker(self) -> None:
        while not self.stop.is_set():
            target = None
            with self.lock:
                for key in list(self.hot):
                    if len(self.cache[key]) < CACHE_DEPTH:
                        target = key
                        break
            if target is None:
                time.sleep(0.15)
                continue
            try:
                tile = self.render(target)
            except Exception:
                time.sleep(0.2)
                continue
            with self.lock:
                if len(self.cache[target]) < CACHE_DEPTH:
                    self.cache[target].append(tile)


class Handler(BaseHTTPRequestHandler):
    factory: TileFactory = None  # set in main()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if url.path == "/style.css":
            return self._serve_static("style.css", "text/css; charset=utf-8")
        if url.path == "/app.js":
            return self._serve_static("app.js", "text/javascript; charset=utf-8")

        if url.path == "/healthz":
            ready = sum(len(q) for q in self.factory.cache.values())
            body = json.dumps({"ok": True, "photos": len(self.factory.photos),
                               "effects": len(effect_table("everything")),
                               "cached_tiles": ready}).encode()
            return self._send(200, body, "application/json")

        if url.path == "/api/genres":
            body = json.dumps({
                "genres": [
                    {"id": "everything", "label": "everything", "blurb": "all algorithms and all genres mixed"},
                    {"id": "default", "label": "algorithms", "blurb": "the raw effect registry, ungrouped"},
                    *[{"id": g.name, "label": g.name, "blurb": g.blurb} for g in GENRES.values()],
                ]
            }).encode()
            return self._send(200, body, "application/json")

        if url.path == "/api/tile.png":
            genre = (query.get("genre") or ["everything"])[0]
            w = bucket(int(float((query.get("w") or [320])[0])))
            h = bucket(int(float((query.get("h") or [240])[0])))
            png, meta = self.factory.take((genre, w, h))
            # Header values are encoded latin-1 by http.server, which cannot
            # represent the en-dashes in names like "floyd–steinberg" or
            # non-ASCII photographer names. Percent-encode; the client decodes.
            return self._send(200, png, "image/png", {
                "Cache-Control": "no-store",
                "X-Effect": quote(meta["effect"]),
                "X-Author": quote(meta["author"]),
                "X-Photo-Id": quote(meta["photo_id"]),
                "Access-Control-Expose-Headers": "X-Effect, X-Author, X-Photo-Id",
            })

        self._send(404, b"not found", "text/plain")

    def _serve_static(self, name: str, ctype: str) -> None:
        path = STATIC / name
        if not path.exists():
            return self._send(404, b"missing", "text/plain")
        self._send(200, path.read_bytes(), ctype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the live dither gallery.")
    # PORT and HOST come from the environment on most hosts.
    ap.add_argument("--port", type=int, default=_env("PORT", 8000))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--photos", type=int, default=_env("DW_PHOTOS", 44))
    ap.add_argument("--workers", type=int, default=_env("DW_WORKERS", 3))
    args = ap.parse_args()

    print(f"loading {args.photos} photos ...")
    photos = fetch(args.photos)
    if not photos:
        raise SystemExit("no photos available")

    Handler.factory = TileFactory(photos, workers=args.workers)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"gallery on http://{args.host}:{args.port}  ({len(photos)} photos, "
          f"{len(effect_table('everything'))} effects, {args.workers} workers, "
          f"max edge {MAX_EDGE}px)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
