"""Drive the gallery in a real browser and capture desktop and mobile views.

Doubles as a smoke test: it fails loudly on console errors, failed requests, or
tiles that never finish loading.

    python3 web/shoot.py --url http://127.0.0.1:8077
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def chromium_path() -> str:
    """Find the preinstalled browser; the layout is versioned, not fixed."""
    for exe in sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome")):
        return str(exe)
    raise SystemExit("no chromium under /opt/pw-browsers")

OUT = Path(__file__).resolve().parent.parent / "out"

VIEWPORTS = [
    ("desktop", {"width": 1440, "height": 900}, 1),
    ("mobile", {"width": 390, "height": 844}, 3),   # iPhone-ish, 3x DPR
]


def capture(url: str, settle: float) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium_path())
        for name, viewport, dpr in VIEWPORTS:
            ctx = browser.new_context(
                viewport=viewport,
                device_scale_factor=dpr,
                is_mobile=(name == "mobile"),
                has_touch=(name == "mobile"),
            )
            page = ctx.new_page()
            page.on("console", lambda m: problems.append(f"console.{m.type}: {m.text}")
                    if m.type == "error" else None)
            page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))

            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(int(settle * 1000))

            loaded = page.evaluate("""() => {
                const tiles = [...document.querySelectorAll('.tile')];
                return {
                    total: tiles.length,
                    ready: tiles.filter(t => t.querySelector('img.ready')).length,
                    stuck: tiles.filter(t => t.classList.contains('loading')).length,
                    status: document.getElementById('status').textContent,
                };
            }""")
            print(f"  {name:8} {loaded['total']:>2} tiles, {loaded['ready']:>2} ready, "
                  f"{loaded['stuck']} still loading — \"{loaded['status']}\"")
            if loaded["ready"] < max(1, loaded["total"] * 0.6):
                problems.append(f"{name}: only {loaded['ready']}/{loaded['total']} tiles rendered")

            page.screenshot(path=OUT / f"web_{name}.png")
            ctx.close()
        browser.close()

    if problems:
        print("\nproblems:")
        for p in dict.fromkeys(problems):
            print(f"  ! {p}")
        return 1
    print("\nno console errors, no failed requests")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--settle", type=float, default=9.0)
    args = ap.parse_args()
    sys.exit(capture(args.url, args.settle))
