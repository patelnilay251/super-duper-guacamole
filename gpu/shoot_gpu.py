"""Drive the GPU wall in real Chromium: capture frames and measure frame rate.

Headless Chromium needs the GPU explicitly enabled with SwiftShader as the
fallback rasteriser, otherwise WebGL2 is simply unavailable and the page fails
in a way that says nothing useful.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "out"

# Outbound traffic in this environment goes through a local CONNECT proxy that
# Chromium does not pick up from the environment the way curl does.
_PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

FLAGS = ([f"--proxy-server={_PROXY}"] if _PROXY else []) + [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
]


def chromium_path() -> str:
    for exe in sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome")):
        return str(exe)
    raise SystemExit("no chromium found")


def run(url: str, seconds: float, shots: int, genre: str | None = None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium_path(), args=FLAGS)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.on("console", lambda m: problems.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))

        page.goto(url, wait_until="load", timeout=60_000)

        renderer = page.evaluate("""() => {
            const c = document.createElement('canvas');
            const gl = c.getContext('webgl2');
            if (!gl) return 'NO WEBGL2';
            const d = gl.getExtension('WEBGL_debug_renderer_info');
            return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown renderer';
        }""")
        print(f"  renderer: {renderer}")

        page.wait_for_timeout(3500)   # textures + shader compile

        if genre:
            page.click(f'.chip:text-is("{genre}")')
            page.wait_for_timeout(2500)

        # Sweep the pointer so the interactive uniforms are exercised.
        for i in range(shots):
            page.mouse.move(200 + i * 340, 300 + i * 120)
            page.wait_for_timeout(int(seconds * 1000 / max(1, shots)))
            page.screenshot(path=OUT / f"gpu_{genre or 'all'}_{i}.png")

        stats = page.evaluate("""() => {
            const t = document.getElementById('status').textContent;
            return { status: t, tiles: document.querySelectorAll('.label').length };
        }""")
        print(f"  {stats['tiles']} tiles — \"{stats['status']}\"")
        if "fps" not in stats["status"]:
            problems.append("no frame-rate reported; render loop may not be running")

        browser.close()

    if problems:
        print("\nproblems:")
        for p in dict.fromkeys(problems):
            print(f"  ! {p}")
        return 1
    print("\nclean: no console errors, no failed requests")
    return 0


def capture_gif(url: str, frames: int, interval: float, width: int, height: int) -> None:
    """Record the wall in motion, sweeping the pointer to show interactivity."""
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    shots = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium_path(), args=FLAGS)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.goto(url, wait_until="load", timeout=60_000)
        page.wait_for_timeout(4000)

        tmp = OUT / "_frames"
        tmp.mkdir(exist_ok=True)
        for i in range(frames):
            # Sweep the pointer across the wall over the course of the capture.
            t = i / max(1, frames - 1)
            page.mouse.move(width * (0.1 + 0.8 * t), height * (0.25 + 0.5 * abs(0.5 - t) * 2))
            page.wait_for_timeout(int(interval * 1000))
            path = tmp / f"f{i:03d}.png"
            page.screenshot(path=path)
            shots.append(Image.open(path).convert("RGB"))
        browser.close()

    scaled = [s.resize((900, int(900 * s.height / s.width)), Image.LANCZOS) for s in shots]
    pal = [f.quantize(colors=200, method=Image.MEDIANCUT, dither=Image.NONE) for f in scaled]
    pal[0].save(OUT / "gpu_motion.gif", save_all=True, append_images=pal[1:],
                duration=int(interval * 1000), loop=0, optimize=True)
    print(f"gif -> {OUT / 'gpu_motion.gif'}  ({len(pal)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8123/")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--gif", action="store_true", help="record motion instead of stills")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--genre", help="click a genre chip before capturing")
    a = ap.parse_args()
    if a.gif:
        capture_gif(a.url, a.frames, 0.16, 1200, 760)
        sys.exit(0)
    sys.exit(run(a.url, a.seconds, a.shots, a.genre))
