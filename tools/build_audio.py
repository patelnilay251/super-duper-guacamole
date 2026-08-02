"""Build the radio's audio: normalise loudness, re-encode, write a manifest.

Two things have to happen before music can sit under the wall.

Loudness. The collected tracks span roughly 24 dB of mean level, which is the
difference between "background" and "startling". Normalising is done the way
broadcast does it -- EBU R128 integrated loudness, measured in an analysis pass
and applied as a fixed gain -- rather than by dynamic compression, which pumps.
A fixed gain per track cannot change how a track breathes; it only decides how
loud it sits.

Tempo. The wall's motion is driven from a clock, not from listening to the
audio. Beat detection jitters and its mistakes are visible; a known BPM is
exact. ccMixter artists state their own tempo, so it is carried through here
rather than estimated.

    python3 tools/build_audio.py --src <dir with candidates.json>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "gpu" / "audio"

# Integrated loudness target. Quieter than a music service (-14) because this
# plays under something you are looking at, not listening to.
TARGET_LUFS = -19.0
BITRATE = "96k"          # stereo; ample for material this soft
PEAK_CEILING = -1.5      # dBTP, leaves room for the encoder

# Curated by hand from the collected pool. Chosen for tempo in the 65-90 range,
# length over two minutes, and spread of mood -- not by any automated score.
# Neither of us could audition the whole set, so expect to swap a few.
KEEP = [
    "ccmixter_Robbero_Sad-Night.mp3",
    "ccmixter_airtone_lostTrack.mp3",
    "ccmixter_airtone_reNovation.mp3",
    "ccmixter_Doxent-Zsigmond_Space-Jazz-78bpm.mp3",
    "ccmixter_Doxent-Zsigmond_Piano-Song-For-Goodnight.mp3",
    "ccmixter_Mana-Junkie_Reflections-in-the-Rain.mp3",
    "ccmixter_onlymeith_Truth.mp3",
    "ccmixter_oldDog_too-quiet-piano.mp3",
    "ccmixter_Wired-Ant_Eternal-Loop-of-Glue-Instrumental.mp3",
    "ccmixter_moscardo_Staying-together-as-the-first-time.mp3",
    "ccmixter_zikweb_ARK.mp3",
    "ccmixter_Jeris_3-pound-universe.mp3",
    "ccmixter_Wired-Ant_Rio-sleeping-Instrumental.mp3",
    "ccmixter_grapes_Ophelias-Song.mp3",
]


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def measure(exe: str, path: Path) -> dict:
    """EBU R128 analysis pass. Returns loudnorm's measured values."""
    r = subprocess.run(
        [exe, "-hide_banner", "-i", str(path), "-af",
         f"loudnorm=I={TARGET_LUFS}:TP={PEAK_CEILING}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True)
    blob = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not blob:
        raise RuntimeError(f"loudnorm gave nothing for {path.name}")
    return json.loads(blob.group(0))


def duration(exe: str, path: Path) -> float:
    r = subprocess.run([exe, "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", r.stderr)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def slug(artist: str, title: str) -> str:
    s = f"{artist}-{title}".lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60]


def build(src: Path) -> int:
    exe = ffmpeg()
    cands = {t["file"]: t for t in json.loads((src / "candidates.json").read_text())}

    missing = [f for f in KEEP if f not in cands]
    if missing:
        print("not in the candidate pool:")
        for f in missing:
            print(f"  ! {f}")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.mp3"):
        old.unlink()

    tracks, total = [], 0
    for name in KEEP:
        rec = cands[name]
        stats = measure(exe, src / name)
        # Pure gain, held back so nothing clips. A limiter would hit the target
        # every time, but it would do it by squashing peaks -- which changes how
        # the track breathes, and the whole point of a fixed gain is that it
        # cannot. A track with a wide crest factor sits quieter instead, which is
        # the honest outcome. (ffmpeg's alimiter also auto-levels to its ceiling
        # by default, so using it here both compressed the dynamics and pushed
        # the output a decibel hotter than asked -- two tracks reached +0.2 dBTP.)
        headroom = PEAK_CEILING - float(stats["input_tp"])
        gain = min(TARGET_LUFS - float(stats["input_i"]), headroom)

        out_name = f"{slug(rec['artist'], rec['title'])}.mp3"
        dest = OUT / out_name
        subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src / name),
             "-af", f"volume={gain:.2f}dB",
             "-c:a", "libmp3lame", "-b:a", BITRATE, "-ac", "2", "-map_metadata", "-1",
             str(dest)],
            check=True, capture_output=True)

        size = dest.stat().st_size
        total += size
        secs = duration(exe, dest)
        tracks.append({
            "src": f"audio/{out_name}",
            "title": rec["title"],
            "artist": rec["artist"],
            "bpm": rec.get("bpm"),
            "seconds": round(secs, 2),
            "bytes": size,
            "license": rec["license"],
            "license_url": rec["license_url"],
            "source": rec["source_page"],
            "gain_db": round(gain, 2),
        })
        print(f"  {rec['artist'][:18]:<18} {rec['title'][:28]:<28} "
              f"{gain:>+6.1f}dB  {secs / 60:>4.1f}min  {size / 1e6:>4.1f}MB")

    known = [t for t in tracks if t["bpm"]]
    (ROOT / "gpu" / "audio.json").write_text(
        json.dumps({"tracks": tracks}, indent=1) + "\n")
    print(f"\n{len(tracks)} tracks -> {OUT}")
    print(f"total {total / 1e6:.1f} MB, {sum(t['seconds'] for t in tracks) / 60:.0f} min")
    print(f"tempo known for {len(known)}/{len(tracks)}")
    print(f"manifest -> gpu/audio.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding candidates.json")
    a = ap.parse_args()
    sys.exit(build(Path(a.src)))
