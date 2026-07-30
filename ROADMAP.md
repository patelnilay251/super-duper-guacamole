# Roadmap

Rendering and compute work that has not been done yet, in roughly the order it
is worth doing. Numbering continues from the items already shipped, so the
labels stay stable in conversation:

1. ~~Linear-light colour~~ — done. Every process that maps tone to ink decides
   on light rather than on the sRGB encoding.
2. ~~Antialiased screens~~ — done, along with `lay()`, which composites partial
   coverage in linear light.
3. ~~Dot gain~~ — done. Modelled as ink spread, so the TVI curve shape and the
   screen-ruling dependence fall out rather than being tuned.

Two harnesses gate every change: `tools/diff_harness.py` asks whether the GPU
and the NumPy reference **agree**, `tools/tone_fidelity.py` asks whether they
are **right**. Anything below should land with a number moving in one of them,
or with a reason why neither applies.

---

## 4. Oklab ramp interpolation

The ramp palettes now bracket correctly in luminance — `rampDither` picks the
pair of entries a tone falls between and chooses by fractional position. But the
blend *between* two stops is still done in linear RGB, which is not perceptually
straight. Interpolating between a deep blue and a pale blue in linear RGB dips
through a desaturated grey rather than holding the hue, which is what makes
cyanotype and ember read muddy at their midpoints.

Oklab is the right space: it was designed so that a straight line between two
colours holds hue and steps evenly in lightness. The conversion is a 3×3 matrix,
a cube root, and another 3×3 — cheap enough to do per fragment, and cheaper
still if the palette is converted once on the CPU and uploaded already in Oklab.

**Where:** `rampDither` and `gradient` in `gpu/shaders.js`; `ramp_ordered`,
`ramp_error_diffuse` and `gradient_map` in `ditherwall/effects.py`.

**Watch for:** the ramps are currently authored as sRGB tuples and their
*luminance* ordering is what the bracketing depends on. Oklab's `L` is not the
same ordering as Rec.709 luminance for saturated colours, so decide explicitly
which one brackets and which one interpolates rather than letting them drift
apart. The differential harness covers `gradient · ironbow` and three ramp
dithers, so a mistake here should show up as a structure drop.

**Risk:** low. **Payoff:** visible on every multi-stop palette.

## 5. Display-P3 for the fluoro riso inks

Real Risograph fluorescent pink (`#FF48B0`-ish) and orange are outside the sRGB
gamut — they are fluorescent, so they re-emit as well as reflect. Clamped into
sRGB they read flat and slightly dirty next to a real print, which is the single
biggest gap between the riso genre and the thing it is imitating.

A `color(display-p3 ...)` canvas plus P3 primaries for `RISO_INKS` would get most
of it back on any modern phone or laptop display, all of which are P3.

**Where:** canvas context creation in `gpu/main.js`
(`getContext('webgl2', { colorSpace: 'display-p3' })`), the ink constants in
`gpu/presets.js`, and the linear-light ink multiply in the `riso` shader.

**Watch for:** this is the item most likely to be *not worth it*, for two
reasons. The compositing already happens in linear light, so widening the gamut
is a change of primaries, not of the maths — but every existing sRGB value on
the wall then needs to be interpreted in the new space or everything shifts.
And the harness compares 8-bit PNG readbacks, which are sRGB; verifying a P3
render through it needs thought, possibly a separate path. Scope the
verification before writing the shader, not after.

**Risk:** medium — it touches the output space for every effect, not just riso.
**Payoff:** large but only on the riso genre, and invisible on an sRGB display.

## 6. Temporally stable blue noise

The animated dithers shimmer. The blue-noise threshold texture is sampled at a
position that moves with time, so each frame draws an independent sample and the
pattern boils. Blue noise is only blue *within* a frame; across frames it is
white, which is the worst possible case for the eye.

The fix is spatiotemporal blue noise — noise that is blue in space *and* along
the time axis, so successive frames are decorrelated but the sequence has no
low-frequency energy. Either generate an offline 64×64×16 volume and index the
third axis by frame, or apply the golden-ratio offset trick
(`threshold = fract(blue(px) + frame * 0.618...)`), which is one line and gets
most of the benefit.

**Where:** `threshold()` in `gpu/shaders.js`, blue-noise generation in
`tools/build_corpus.py` if the volume route is taken.

**Watch for:** neither harness will catch this, because both render a single
pinned frame. If it is worth doing it is worth measuring — capture N frames of a
static tile and report the per-pixel temporal variance, which should drop sharply
without the spatial spectrum changing.

**Risk:** low. **Payoff:** the whole wall is animated, so this affects
everything, but it is a subtle-motion improvement rather than a still-frame one.

## 7. 2048px corpus

The photographs are currently optimised to a size that limits how tightly a tile
can crop before the mip chain runs out and the source goes soft. More resolution
means tighter crops stay sharp, which matters more now that the screens resolve
sub-pixel geometry.

**Where:** `tools/build_corpus.py`, then re-run and commit `gpu/photos/`.

**Watch for:** this is pure asset work with no rendering risk, but it is the one
item with a real deployment cost — Cloudflare Pages has a per-file and per-site
budget, and the corpus is already the bulk of the deploy. Check the total before
committing; the deploy step prints it. Consider AVIF over JPEG, which would buy
back most of the size.

**Risk:** none to rendering, some to deploy size. **Payoff:** moderate, and it
compounds with anything that sharpens the process.

## 8. Anisotropic filtering

Steeply cropped tiles sample the texture at very different rates along the two
axes. Trilinear filtering picks one mip level for both, so it blurs the sharp
axis to avoid aliasing the stretched one. `EXT_texture_filter_anisotropic` is
available essentially everywhere and is a two-line change at texture setup.

**Where:** `texFromImage` in `gpu/main.js` and `gpu/diff.html`.

**Watch for:** the extension is not guaranteed, so query it and degrade
silently. Setting the max anisotropy blindly can cost fill rate on mobile; clamp
to 4 or 8 rather than the reported maximum.

**Risk:** very low. **Payoff:** small but free, and it stacks with item 7.

## 9. Error diffusion in focus mode, via a Web Worker

Floyd–Steinberg and its relatives are sequential by construction — each pixel's
error depends on the one before it — so they cannot run in a fragment shader.
They are the reason `ditherwall/effects.py` still exists as a separate engine,
and they produce a visibly different, more organic texture than any ordered
dither.

Focus mode is where this becomes affordable: one photograph, one tile, no
60fps requirement. Run the kernel in a Web Worker over an `ImageData` of the
focused crop, hand back a bitmap, and draw it in place of the shader output.

**Where:** new worker module, wired into the focus path in `gpu/main.js`.

**Watch for:** the port must match `KERNELS` in `ditherwall/effects.py` exactly,
including the serpentine traversal — this is the one item where a pixel-exact
comparison against the reference is both possible and worth writing, since both
sides would be doing identical integer-order arithmetic. Do that rather than
reusing the structural harness.

**Risk:** medium — new execution path, new failure mode (worker never returns).
**Payoff:** the only genuinely *new* capability on this list; everything else
improves something that already renders.

## 10. Partial tone normalisation

`normalize_tone` stretches every crop to the full range and then gamma-shifts
its mean toward 0.48. That is what makes tiles comparable, and it was the right
call while the wall was proving out kernels. But it also means a genuinely
low-contrast photograph — fog, snow, an overcast horizon — gets forced to the
same contrast as everything else, and loses the quality that made it worth
including.

The fix is to blend rather than replace: normalise toward the target by a
fraction that depends on how far the source already is, so a normal photograph
is untouched and only a badly exposed one gets pulled.

**Where:** `normalize_tone` in `ditherwall/effects.py`, `toneMap` in
`gpu/shaders.js`, and `cropTone()` in `gpu/main.js`, which computes the
percentiles and gamma on the CPU. All three have to agree or the harness will
say so.

**Watch for:** this changes the input to every effect, so both harnesses will
move on every case. Expect to re-baseline `MAX_EXPOSURE_DELTA` and
`MAX_RESIDUAL` — and re-baseline them from measured values once the two
implementations agree, not from guesses. The last time these tolerances were set
from guesses they were loose enough to pass a 38× regression.

**Risk:** high, in the sense that it touches everything at once. **Payoff:**
aesthetic rather than technical, and the hardest to judge without looking at a
lot of photographs.

---

## Not planned

- **Mobile captions.** There is no hover on touch, and the corner HUD is
  deliberately minimal. Needs a different interaction, not a port of this one.
- **Fullscreen kuwahara and crystallize.** Both are many-tap filters and are the
  frame-rate floor at full size. Would need a separable or downsampled
  approximation.
- **Error diffusion on the whole wall.** See item 9 — sequential by
  construction. Focus mode only.
