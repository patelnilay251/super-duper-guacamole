# Audio provenance

Two manifests, kept because the expensive part of collecting music was not the
downloading — it was verifying that every track could legally be shipped.

## `candidates.json`

All 61 tracks collected, of which 14 ship in `gpu/audio/`. For each: title,
artist, licence, the page that *states* that licence, the download URL, and a
measured duration and loudness.

**Every licence was confirmed against the source page's served HTML**, not
against API metadata. That distinction turned out to matter: archive.org's
`licenseurl` field is uploader-supplied, and items tagged CC0 there included
Nine Inch Nails, a Jason Mraz single and the Skyrim soundtrack. Nothing was
taken on metadata alone.

Durations are measured too, not read from metadata — ccMixter's API reported one
track as 202s when it is 388s.

The 47 tracks that are collected but not shipped are re-downloadable from this
file. Two things make that harder than it looks, both worth knowing before
trying:

- ccMixter's content server returns 403 without a `Referer` header pointing at
  the track's own page.
- It also sends HTTP header lines over 64 kB, which exceeds Python's default
  `http.client._MAXLINE` and fails the request.

## `flagged_not_downloaded.json`

Material found but deliberately left alone, so the decision does not have to be
made twice. Mostly **172 shortlisted Jamendo instrumentals** (65 CC-BY, 107
CC-BY-SA): Openverse reports their licence and the MP3 URLs work, but Jamendo's
track pages are fully JS-rendered and their served HTML contains no licence text
at all. The licence could not be confirmed *at the source*, so nothing was
taken. Shipping those needs either a Jamendo API key or a decision to accept
Openverse as licence-of-record — a call for a human, not a script.

Also recorded: NC-licensed ccMixter uploads (excluded at query time), and
Holizna's archive.org presence, which is CC-BY-NC-ND there even though the same
artist's material on Freesound is CC0.

## Rebuilding

```bash
python3 tools/build_audio.py --src <dir holding candidates.json and the audio>
```

`KEEP` at the top of that script is the shipped selection.
