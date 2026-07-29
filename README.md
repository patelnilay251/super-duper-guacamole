# ditherwall

Forty-odd dithering and shader-style algorithms, applied to real photographs and
rendered onto a single canvas so they can be compared at a glance.

![the wall](out/wall.png)

Every tile is a different algorithm on a different photograph, cropped at its own
zoom and framing. The layout is a recursive split rather than a grid, so the tile
shapes vary and the result reads as a composition instead of a contact sheet.

## Genres

The default wall is deliberately heterogeneous — it exists to compare algorithms
against each other. A genre does the opposite: it fixes the palette, screen and
finish so every tile reads as one process, and the variation becomes variation
*within* a style.

![genres](out/genres.jpg)

| Genre | Process |
|---|---|
| `newsprint` | Black ink on warm stock, rotated screens at 15°/45°/60°, engraving hatch |
| `handheld` | Four-tone LCD greens, chunky sprite pixels, hard quantisation |
| `risograph` | Two spot inks, separate screen angles, deliberate misregistration |
| `nightcity` | Neon duotones, chromatic aberration, pixel sort, signal damage |
| `blueprint` | Cyanotype stock, drafting lines, edge extraction |
| `thermal` | False-colour heat ramps — ironbow, magma, arctic, sodium |
| `phosphor` | Monochrome terminal tubes in green, amber and ice, with glow bleed |
| `xerox` | Crushed tone curve, toner speckle, generation loss over three passes |

## Run it

```bash
pip install numpy pillow
cd ditherwall
python3 render.py                                  # 2600×1700, 40 tiles
python3 render.py --genre risograph --tiles 30     # a single style
python3 render.py --tiles 24 --seed 3 --no-labels  # a different wall
```

Photographs are pulled from Picsum's Unsplash-backed catalogue and cached in
`cache/`, so only the first run needs network. `--seed` controls photo choice,
crops, and layout together; the same seed always reproduces the same wall.

## What's in it

| Family | Count | Examples |
|---|---|---|
| Error diffusion | 13 | Floyd–Steinberg, Atkinson, Jarvis–Judice–Ninke, Stucki, Burkes, Sierra |
| Ordered / threshold | 8 | Bayer 2×2/4×4/8×8, blue noise, white noise |
| Screen & line | 4 | Rotated halftone, crosshatch engraving, Sobel edges |
| Shader-style | 13 | Kuwahara, CRT aperture mask, crystallize, pixel sort, bloom, displace |
| Composites | 4 | `bloom → atkinson`, `kuwahara → bayer`, `crt → halftone` |

Palettes include monochrome, Game Boy, CGA, C64, and three duotones. Every error
diffusion kernel can target any palette.

Everything is plain NumPy — no GPU, no shader compiler. The "shader" effects are
the same operations a fragment shader would do, computed on the CPU.

## Live gallery

A web front end over the same engine. Every tile independently pulls its own
random photograph and its own random effect, rendered server-side, and refreshes
on its own jittered timer — so the wall changes continuously and unevenly rather
than blinking over all at once.

```bash
python3 web/server.py --port 8000     # then open http://127.0.0.1:8000
```

| | |
|---|---|
| ![desktop](out/web_desktop.png) | ![mobile](out/web_mobile.png) |

No framework and no build step — `http.server` with a thread pool on the back,
one HTML/CSS/JS file each on the front. Genre chips switch the whole wall between
the eight presets or the full 135-effect mix.

**Responsive.** A CSS grid of `auto-fill / minmax(min(--tile, 46vw), 1fr))`, with
tiles spanning a random number of 10px rows so heights stay uneven. Tile count is
derived from the viewport. Captions reveal on hover on desktop but are pinned on
under `@media (hover: none)`, since a touch screen has no hover.

**Tiles are requested at device pixels.** A tile asks for `CSS size × devicePixelRatio`
and the image is `image-rendering: pixelated`. Dithering is a per-pixel pattern —
letting the browser resample it turns the whole point of the exercise to mush.

**Pre-rendering.** An effect costs 20ms to about a second, too slow to sit inside
a request on a phone. Worker threads render ahead of demand into a per-size cache,
so a request normally pops a finished tile — measured 0.55s cold, under 5ms warm.
Requested dimensions are rounded to an 80px grid to keep the number of cache keys
small; `object-fit: cover` absorbs the difference.

**Percent-encoded metadata headers.** The effect name and photographer ride back
on `X-Effect` / `X-Author`. `http.server` encodes headers as latin-1, which cannot
represent the en-dash in `floyd–steinberg` or the arrow in `bloom → atkinson`;
before this was fixed roughly a third of requests died with `UnicodeEncodeError`.
Values are percent-encoded on the way out and decoded in the client.

`web/shoot.py` drives the page in real Chromium at desktop and mobile viewports,
and doubles as a smoke test — it fails on console errors, failed requests, or
tiles that never finish loading.

## GPU wall

A second front end that renders the effects as WebGL2 fragment shaders on the
viewer's own GPU. No server compute at all, so it deploys as static files — and
because it runs at frame rate, the effects can *move*: screen angles rotate,
dither grain crawls, riso registration drifts, and everything responds to the
pointer.

![gpu wall in motion](out/gpu_motion.gif)

```bash
python3 tools/build_corpus.py     # optimised photos + manifest + blue noise
cd gpu && python3 -m http.server 8123
```

**One context, not one per tile.** Browsers cap concurrent WebGL contexts at
roughly 8–16, so a canvas per tile fails outright past a dozen. The canvas covers
the page instead, and each tile is drawn by setting `gl.viewport` to its rectangle
and running a full-screen quad through that tile's program.

**Photographs ship with the site.** WebGL textures must be same-origin or
CORS-enabled, and Picsum sends no CORS header. Serving a bundled corpus sidesteps
the question and removes any runtime dependency on a third-party CDN or its rate
limits — 36 photographs at 1024px come to 3.9 MB.

**Statistics the shader cannot compute are precomputed.** `normalize_tone` needs
luminance percentiles over the image; those go in the manifest at build time. The
per-crop exposure gamma is measured on the CPU once per tile by drawing the crop
into an 8×8 canvas and reading it back. Doing that in the shader instead cost
nine texture fetches on *every pixel* to recompute a value constant across the
tile — removing it took the software-rendered frame rate from 5 fps to 13.

Error diffusion does not port: it is sequential by construction. It stays on the
CPU and is not part of the GPU wall.

## Deploying

The repo carries a `Dockerfile` and a Render Blueprint. Deployment is a pull, not
a push — connect the repo in the Render dashboard (**New → Blueprint**), and it
reads `render.yaml`, builds the image, and serves it. No API token is needed.

```bash
docker build -t ditherwall .
docker run -p 8000:8000 ditherwall
```

The photo corpus is baked into the image at build time. Fetching at boot would
make every cold start wait on ~30 HTTP round trips, and a free instance cold-starts
often.

**Tuning knobs**, all environment variables, all read at startup:

| Variable | Default | Effect |
|---|---|---|
| `DW_WORKERS` | 3 | Render threads. More than the core count just adds contention |
| `DW_MAX_EDGE` | 720 | Longest tile edge. Cost is roughly quadratic in this |
| `DW_PHOTOS` | 44 | Corpus size — memory and image size |
| `DW_CACHE_DEPTH` | 4 | Pre-rendered tiles held per size bucket |

The Blueprint sets 480px / 2 workers, because a free instance gets a fraction of
a core and cost scales with area. On a paid plan, raise both.

**The client throttles itself.** Refresh interval is derived from measured
latency rather than fixed: N tiles refreshing every T seconds demand `N/T`
renders per second against a capacity of about `workers/latency`. So on a fast
machine tiles turn over every ~5s, and on a slow host the interval stretches
toward its 90s ceiling instead of piling up a queue the server can never drain.
The status line reports both the measured latency and the current interval.

## Notes on the implementation

Three things turned out to matter more than the algorithms themselves.

**Tone normalization before quantization.** Dithering is unforgiving about
exposure: a bright photo reduced to two levels clips to near-solid white and the
pattern vanishes. `normalize_tone` stretches each tile to full range and
gamma-shifts its mean toward mid grey first, which is what makes the kernels
comparable to each other rather than to their source photos.

**Bayer matrix normalization.** The recursive matrix has to be scaled as
`(m + 0.5) / n²`, not `m / n²`. Without the half-step the matrix mean sits below
0.5, which biases every threshold and darkens the output — at 2×2 a 50% grey
renders as 25% white. Verified against uniform patches: each matrix now
reproduces input tone to within its own quantization limit.

**Ramp palettes must be dithered in luminance.** Nearest-colour matching in RGB
is correct for a scattered palette like C64, but wrong for a tonal ramp: the Game
Boy palette is four greens, so a green-ish photograph lands near the light end for
every pixel and the tile comes out a flat field. `RAMP_PALETTES` routes those
through `ramp_error_diffuse` / `ramp_ordered`, which quantise luminance into
level indices and diffuse the error per step.

**Crops that seek detail.** A random window at high zoom regularly lands on blank
sky, which dithers into a dead grey field. Each tile proposes several candidate
crops and keeps the one with the most local gradient energy. The largest tiles
also get the most detailed photographs, since an empty tile is most conspicuous
at size.

Error diffusion runs on Python lists rather than NumPy arrays — the access
pattern is inherently sequential, and scalar indexing into an `ndarray` costs
more per access than plain float arithmetic. A full 2600×1700 wall renders in
about 11 seconds.

## Layout

```
ditherwall/
  fetch.py     photo corpus + caching
  effects.py   all algorithms, plus the EFFECTS registry
  layout.py    recursive canvas splitting, detail-seeking crops
  render.py    composition and labelling
```

Photographs are from Unsplash via Picsum and are credited in each tile's caption.
