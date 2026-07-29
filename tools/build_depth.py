"""Precompute a depth map for every photograph in the corpus.

Monocular depth estimation is far too slow to run per frame, but it only has to
happen once: the result ships as a greyscale image alongside the photograph and
becomes just another texture. The shaders then get distance for free, and can
vary a process across the depth of a scene rather than uniformly over the frame.

    python3 tools/build_depth.py

Uses Depth Anything V2 (small, quantised) via ONNX Runtime on CPU -- roughly a
second per image, which is fine for a corpus this size.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
GPU = ROOT / "gpu"
DEPTH_DIR = GPU / "depth"
MODEL = ROOT / "models" / "depth_anything_v2_small.onnx"
MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small"
    "/resolve/main/onnx/model_quantized.onnx"
)

# The network was trained on ImageNet-normalised input at a multiple of 14.
SIZE = 518
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_model() -> None:
    if MODEL.exists():
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading depth model ({MODEL_URL.rsplit('/', 1)[-1]}) ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL)
    print(f"  {MODEL.stat().st_size / 1e6:.1f} MB")


def preprocess(img: Image.Image) -> np.ndarray:
    resized = img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]           # NCHW


def estimate(session, img: Image.Image) -> Image.Image:
    depth = session.run(None, {"pixel_values": preprocess(img)})[0][0]

    # The model emits relative inverse depth on an arbitrary scale, so it has to
    # be normalised per image. Percentiles rather than min/max: a single stray
    # pixel would otherwise compress the whole range.
    lo, hi = np.percentile(depth, 1), np.percentile(depth, 99)
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)

    out = Image.fromarray((norm * 255).astype("uint8"), mode="L")
    out = out.resize(img.size, Image.BICUBIC)
    # A light blur hides the network's blocky artefacts, which would otherwise
    # show up as visible steps once a screen or dither is keyed to depth.
    return out.filter(ImageFilter.GaussianBlur(1.2))


def build(quality: int) -> None:
    import onnxruntime as ort

    ensure_model()
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = GPU / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    total = 0

    for rec in manifest["photos"]:
        src = GPU / rec["src"]
        img = Image.open(src)
        depth = estimate(session, img)

        name = Path(rec["src"]).stem + ".jpg"
        path = DEPTH_DIR / name
        depth.save(path, "JPEG", quality=quality, optimize=True)
        total += path.stat().st_size

        rec["depth"] = f"depth/{name}"
        print(f"  {rec['src']:<22} -> {rec['depth']}")

    manifest_path.write_text(json.dumps(manifest, indent=1))
    n = len(manifest["photos"])
    print(f"\n{n} depth maps, {total / 1e6:.1f} MB total, mean {total / max(1, n) / 1e3:.0f} kB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=82)
    build(ap.parse_args().quality)
