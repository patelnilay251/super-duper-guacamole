// Single WebGL2 context driving the whole wall.
//
// One context, not one per tile: browsers cap concurrent WebGL contexts at
// roughly 8-16, so a canvas per tile fails outright past a dozen. Instead the
// canvas covers the page and each tile is drawn by setting gl.viewport to its
// rectangle and running a full-screen quad through that tile's program.

import { VERT, FRAG } from './shaders.js';
import { PRESETS, pickEffect, pickPlacement } from './presets.js';
import { splitCanvas } from './layout.js';

const canvas = document.getElementById('stage');
const labels = document.getElementById('labels');
const statusEl = document.getElementById('status');

const gl = canvas.getContext('webgl2', {
  antialias: false,
  powerPreference: 'high-performance',
  preserveDrawingBuffer: false,
});

if (!gl) {
  document.body.innerHTML =
    '<p style="padding:2rem;font:14px monospace;color:#eee">WebGL2 is required and unavailable in this browser.</p>';
  throw new Error('no webgl2');
}

// A tile holds its process for a while, then cross-fades into another one.
// Both are drawn; the incoming one is composited over the outgoing at rising
// alpha, so the change reads as a dissolve rather than a cut.
const MORPH_MIN = 6, MORPH_SPREAD = 14, MORPH_SECONDS = 1.8;

// A style change ripples outward from wherever it was triggered rather than
// cutting. Each tile reuses the cross-fade it already has; only the start time
// differs, so the wall dissolves into the new style in a wave.
const RIPPLE_SECONDS = 0.9, RIPPLE_JITTER = 0.25;

// Focus: one tile grows to fill the canvas while the rest are pushed back.
const FOCUS_INSET = 26;          // px of canvas left visible around the tile
const FOCUS_COARSEN = 3.2;       // how much the unfocused processes enlarge
const FOCUS_DIM = 0.62;

// Critically-damped-ish spring. Chosen over an eased tween because it is
// interruptible: clicking another tile mid-flight redirects instead of snapping.
//
// Integrated in fixed sub-steps rather than one step of dt. Explicit Euler
// diverges once damping * dt exceeds 2 -- about 74ms here, so any frame below
// ~14fps sends it running away instead of settling. It reached 2.35 on a
// software rasteriser, which drove the dim past 1 and rendered the wall black.
const SPRING_H = 1 / 240;

function springStep(s, target, dt) {
  const k = 190, c = 27;
  const steps = Math.min(32, Math.max(1, Math.ceil(dt / SPRING_H)));
  const h = dt / steps;
  for (let i = 0; i < steps; i++) {
    s.v += (-k * (s.x - target) - c * s.v) * h;
    s.x += s.v * h;
  }
  if (Math.abs(s.x - target) < 0.0005 && Math.abs(s.v) < 0.0025) { s.x = target; s.v = 0; }
  // Clamped regardless: everything downstream lerps on the assumption of 0..1.
  s.x = Math.min(1, Math.max(0, s.x));
  return s.x;
}

const state = {
  tiles: [],
  photos: [],
  textures: [],
  programs: {},
  noise: null,
  pointer: { x: 0, y: 0, active: 0 },
  frames: 0,
  fps: 0,
  paused: false,
  genre: 'everything',
  dpr: Math.min(devicePixelRatio || 1, 2),
  focus: { tile: null, s: { x: 0, v: 0 } },
};

// -------------------------------------------------------------- gl helpers

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(`shader: ${gl.getShaderInfoLog(sh)}`);
  }
  return sh;
}

function program(fragSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragSrc));
  gl.bindAttribLocation(p, 0, 'aPos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`link: ${gl.getProgramInfoLog(p)}`);
  }
  // Cache uniform locations once; getUniformLocation per frame is a real cost.
  const uniforms = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
    uniforms[name] = gl.getUniformLocation(p, name);
  }
  return { p, uniforms };
}

function loadTexture(img) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.generateMipmap(gl.TEXTURE_2D);   // cropMean() samples a coarse level
  return tex;
}

function noiseTexture(img) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 64, 64, 0, gl.RED, gl.UNSIGNED_BYTE, img);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  return tex;
}

// ------------------------------------------------------------------ assets

async function loadImage(src) {
  const img = new Image();
  img.decoding = 'async';
  img.src = src;
  await img.decode();
  return img;
}

// The corpus ships far more photographs than one wall uses, and a random subset
// is drawn per visit. That is what makes the wall different every time it is
// opened, and it costs nothing at runtime: the photographs that are not drawn
// are simply never requested. Over repeated visits the browser accumulates the
// whole corpus in cache, so the site gets *faster* the more it is used -- which
// is the opposite of what fetching fresh photographs from an API would do.
const WALL_PHOTOS = 36;
const WALL_PHOTOS_SMALL = 20;   // a phone pays the same bytes for far less wall

// Seeded PRNG, so a capture can pin the selection. Math.random cannot be
// seeded, and the differential harness is not the only thing that needs a
// reproducible wall -- screenshots do too.
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function choosePhotos(all, params) {
  const fallback = window.innerWidth < 900 ? WALL_PHOTOS_SMALL : WALL_PHOTOS;
  const asked = Number(params.get('photos')) || fallback;
  const want = Math.max(1, Math.min(asked, all.length));

  const seed = params.get('seed');
  const rand = seed === null ? Math.random : mulberry32(Number(seed) >>> 0);

  const idx = all.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {          // Fisher-Yates
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, want).map(i => all[i]);
}

async function boot() {
  const manifest = await (await fetch('manifest.json')).json();
  const wanted = choosePhotos(manifest.photos, new URLSearchParams(location.search));

  // A photograph that will not decode drops out instead of taking the wall with
  // it. Boot fires dozens of requests at once, so a single stalled or truncated
  // response is a transport problem, not a reason to render nothing.
  const loaded = await Promise.all(wanted.map(async p => {
    try {
      return { meta: p, img: await loadImage(p.src),
               depth: p.depth ? await loadImage(p.depth).catch(() => null) : null };
    } catch {
      console.warn(`skipped ${p.src}`);
      return null;
    }
  }));

  const ok = loaded.filter(Boolean);
  if (!ok.length) throw new Error('no photographs could be loaded');

  state.photos = ok.map(r => r.meta);
  const imgs = ok.map(r => r.img);
  const depthImgs = ok.map(r => r.depth);
  const noiseImg = await loadImage('bluenoise.png');

  state.images = imgs;
  state.textures = imgs.map(loadTexture);
  // Depth is optional: a photo without a map falls back to its own texture,
  // which reads as uniformly near and leaves depth effects harmless.
  state.depth = depthImgs.map((d, i) => d ? loadTexture(d) : state.textures[i]);
  state.noise = noiseTexture(noiseImg);

  for (const [name, src] of Object.entries(FRAG)) state.programs[name] = program(src);

  // Full-screen quad, reused by every program.
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  resize();
  requestAnimationFrame(frame);
}

// ------------------------------------------------- per-tile exposure (CPU)

// Drawing the crop into a small canvas makes the browser downsample it for us,
// so one drawImage plus a readback yields everything the tone map needs.
//
// This is measured per crop, not per photograph. A percentile range taken over
// the whole frame is wrong for a tight crop -- a crop of sky inside a
// high-contrast photo gets almost no stretch and dithers to a flat field.
const TONE_N = 32;
const toneCanvas = document.createElement('canvas');
toneCanvas.width = toneCanvas.height = TONE_N;
const toneCtx = toneCanvas.getContext('2d', { willReadFrequently: true });

// One channel through toneMap and then to linear light, mirroring the shader.
function toneLinear(byte, lo, span, gamma) {
  const s = Math.pow(Math.min(1, Math.max(0, (byte / 255 - lo) / span)), gamma);
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

function cropTone(img, cx, cy, cw, ch) {
  toneCtx.drawImage(img, cx * img.width, cy * img.height, cw * img.width, ch * img.height,
                    0, 0, TONE_N, TONE_N);
  const d = toneCtx.getImageData(0, 0, TONE_N, TONE_N).data;

  const lum = new Float64Array(d.length / 4);
  let acc = 0;
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    lum[j] = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
    acc += lum[j];
  }

  const sorted = Float64Array.from(lum).sort();
  const at = q => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const lo = at(0.015), hi = at(0.985);

  // Gamma is measured after the stretch, matching normalize_tone's order.
  const span = Math.max(1e-3, hi - lo);
  let post = 0;
  for (let j = 0; j < lum.length; j++) post += Math.min(1, Math.max(0, (lum[j] - lo) / span));
  const mean = Math.min(0.98, Math.max(0.02, post / lum.length));
  const gamma = Math.min(2.0, Math.max(0.5, Math.log(0.48) / Math.log(mean)));

  // Warm/cool spread, for riso's second separation. Measured through the same
  // tone map the shader applies and in linear light, because that is where the
  // shader takes the difference. Free here: the pixels are already read back.
  const chroma = new Float64Array(lum.length);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    chroma[j] = toneLinear(d[i], lo, span, gamma) - toneLinear(d[i + 2], lo, span, gamma);
  }
  const cs = Float64Array.from(chroma).sort();
  const cAt = q => cs[Math.min(cs.length - 1, Math.floor(q * cs.length))];

  return { lo, hi, gamma, chromaLo: cAt(0.03), chromaHi: cAt(0.97) };
}

// ------------------------------------------------------------------ layout

function buildTiles() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const rects = splitCanvas(w, h, tileTarget(w, h));

  state.focus.tile = null;
  state.focus.s.x = 0;
  state.focus.s.v = 0;
  document.body.classList.remove('focused');

  state.tiles = rects.map((r) => {
    const photo = Math.floor(Math.random() * state.photos.length);
    const place = pickPlacement();
    const t = { ...r, photo, seed: Math.random(), fx: pickEffect(state.genre), next: null, morph: 0 };

    const meta = state.photos[photo];
    const tileAspect = t.w / t.h, photoAspect = meta.w / meta.h;
    let cw = 1, ch = 1;
    if (photoAspect > tileAspect) cw = tileAspect / photoAspect; else ch = photoAspect / tileAspect;
    cw /= place.zoom; ch /= place.zoom;
    t.crop = [(1 - cw) * place.cropX, (1 - ch) * place.cropY, cw, ch];
    t.tone = cropTone(state.images[photo], ...t.crop);
    // Stagger first transitions so the wall never changes all at once.
    t.swapAt = performance.now() / 1000 + MORPH_MIN + Math.random() * MORPH_SPREAD;
    return t;
  });
  renderLabels();
}

function tileTarget(w, h) {
  const area = w * h;
  return Math.max(8, Math.min(40, Math.round(area / 46000)));
}

function positionLabel(t, view) {
  if (!t.label) return;
  t.label.style.left = `${view.x}px`;
  t.label.style.top = `${view.y + view.h}px`;
  t.label.style.width = `${view.w}px`;
}

function renderLabels() {
  labels.innerHTML = '';
  for (const t of state.tiles) {
    const el = document.createElement('div');
    el.className = 'label';
    el.style.left = `${t.x}px`;
    el.style.top = `${t.y + t.h}px`;
    el.style.width = `${t.w}px`;
    el.innerHTML =
      `<span class="effect">${t.fx.label}</span>` +
      `<span class="credit">photo · ${state.photos[t.photo].author}</span>`;
    t.label = el;
    labels.appendChild(el);
  }
}

function resize() {
  const w = innerWidth, h = innerHeight;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * state.dpr);
  canvas.height = Math.round(h * state.dpr);
  buildTiles();
}

// --------------------------------------------------------------- rendering

// `view` carries the geometry and treatment for this draw, which may be the
// tile's resting values or an interpolation toward its focused ones.
function drawTile(t, fx, time, alpha, view) {
  const { p, uniforms: u } = state.programs[fx.program];
  gl.useProgram(p);

  const d = state.dpr;
  // gl.viewport origin is bottom-left; the layout is top-left.
  gl.viewport(
    Math.round(view.x * d),
    Math.round((canvas.clientHeight - view.y - view.h) * d),
    Math.round(view.w * d),
    Math.round(view.h * d),
  );

  const photo = state.photos[t.photo];

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.textures[t.photo]);
  gl.uniform1i(u.uTex, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, state.noise);
  gl.uniform1i(u.uNoise, 1);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, state.depth[t.photo]);
  gl.uniform1i(u.uDepthTex, 2);

  gl.uniform4f(u.uCrop, view.crop[0], view.crop[1], view.crop[2], view.crop[3]);
  gl.uniform1f(u.uCropGamma, view.gamma);
  gl.uniform2f(u.uRes, view.w * d, view.h * d);
  gl.uniform1f(u.uTime, time);
  gl.uniform1f(u.uSeed, t.seed);
  gl.uniform2f(u.uPointer, state.pointer.x, state.pointer.y);
  gl.uniform1f(u.uPointerAmt, state.pointer.active);
  gl.uniform2f(u.uTone, view.lo, view.hi);
  if (u.uChroma) gl.uniform2f(u.uChroma, view.chromaLo, view.chromaHi);
  gl.uniform1f(u.uCoarsen, view.coarsen);
  gl.uniform1f(u.uDim, view.dim);

  if (u.uAlpha) gl.uniform1f(u.uAlpha, alpha);

  if (u.uPalette && fx.palette) {
    gl.uniform3fv(u.uPalette, fx.palette.flat());
    gl.uniform1i(u.uPaletteLen, fx.palette.length);
  }
  for (const [k, v] of Object.entries(fx.uniforms || {})) {
    const loc = u[k];
    if (!loc) continue;
    if (typeof v === 'number') gl.uniform1f(loc, v);
    else if (Array.isArray(v) && typeof v[0] === 'number') gl.uniform3fv(loc, v);
    else if (Array.isArray(v)) { gl.uniform3fv(loc, v.flat()); }
  }
  if (fx.kind !== undefined && u.uKind) gl.uniform1i(u.uKind, fx.kind);
  if (fx.rampLen && u.uRampLen) gl.uniform1i(u.uRampLen, fx.rampLen);

  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

// Cover-fit a photo into an arbitrary aspect at zoom 1, centred: the framing a
// tile relaxes into when it is focused.
function fullCrop(t, w, h) {
  const meta = state.photos[t.photo];
  const target = w / h, photo = meta.w / meta.h;
  let cw = 1, ch = 1;
  if (photo > target) cw = target / photo; else ch = photo / target;
  return [(1 - cw) * 0.5, (1 - ch) * 0.5, cw, ch];
}

const lerp = (a, b, k) => a + (b - a) * k;

function viewFor(t, k) {
  // k is this tile's focus weight: 1 fully focused, 0 fully at rest.
  if (k <= 0) {
    const away = state.focus.s.x;   // how focused *something else* is
    return {
      x: t.x, y: t.y, w: t.w, h: t.h, crop: t.crop,
      lo: t.tone.lo, hi: t.tone.hi, gamma: t.tone.gamma,
      chromaLo: t.tone.chromaLo, chromaHi: t.tone.chromaHi,
      coarsen: lerp(1, FOCUS_COARSEN, away),
      dim: lerp(0, FOCUS_DIM, away),
    };
  }

  const cw = canvas.clientWidth, chh = canvas.clientHeight;
  const fx = FOCUS_INSET, fy = FOCUS_INSET;
  const fw = cw - FOCUS_INSET * 2, fh = chh - FOCUS_INSET * 2;

  const x = lerp(t.x, fx, k), y = lerp(t.y, fy, k);
  const w = lerp(t.w, fw, k), h = lerp(t.h, fh, k);

  // The crop relaxes out of its tight framing into the whole photograph as the
  // tile grows, so focusing reveals the frame rather than just magnifying it.
  const target = t.focusCrop || t.crop;
  const crop = t.crop.map((v, i) => lerp(v, target[i], k));
  const tone = t.focusTone || t.tone;

  return {
    x, y, w, h, crop,
    lo: lerp(t.tone.lo, tone.lo, k),
    hi: lerp(t.tone.hi, tone.hi, k),
    gamma: lerp(t.tone.gamma, tone.gamma, k),
    chromaLo: lerp(t.tone.chromaLo, tone.chromaLo, k),
    chromaHi: lerp(t.tone.chromaHi, tone.chromaHi, k),
    coarsen: 1,
    dim: 0,
  };
}

function focusTile(t) {
  const f = state.focus;
  if (f.tile === t) { f.tile = null; return; }
  if (t) {
    // One readback, when focus begins -- not thirty-five, and not per frame.
    const cw = canvas.clientWidth - FOCUS_INSET * 2;
    const chh = canvas.clientHeight - FOCUS_INSET * 2;
    t.focusCrop = fullCrop(t, cw, chh);
    t.focusTone = cropTone(state.images[t.photo], ...t.focusCrop);
  }
  f.tile = t;
}

// A style change does not rebuild the wall. Layout, photographs and crops stay
// put -- only the effect changes, staggered by distance from the origin, so the
// tone measurements are not redone and nothing jumps.
function restyle(originX, originY) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const far = Math.hypot(w, h);
  for (const t of state.tiles) {
    const dx = t.x + t.w / 2 - originX;
    const dy = t.y + t.h / 2 - originY;
    const d = Math.hypot(dx, dy) / far;
    t.restyleAt = performance.now() / 1000
      + d * RIPPLE_SECONDS + Math.random() * RIPPLE_JITTER;
  }
}

let last = performance.now(), acc = 0, accFrames = 0;

function frame(now) {
  requestAnimationFrame(frame);
  if (state.paused) { last = now; return; }

  const time = now / 1000;
  const dt = Math.min(0.1, (now - last) / 1000);   // clamped: tab switches jump
  gl.disable(gl.SCISSOR_TEST);
  gl.clearColor(0.04, 0.04, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  springStep(state.focus.s, state.focus.tile ? 1 : 0, dt);
  const focused = state.focus.tile;
  // Ease the spring's output so the growth reads as motion rather than a scale.
  const fk = state.focus.s.x;
  const focusK = fk * fk * (3 - 2 * fk);

  // The focused tile is drawn last so it lands on top of everything else.
  const order = focused
    ? [...state.tiles.filter(t => t !== focused), focused]
    : state.tiles;

  for (const t of order) {
    // A restyle fires on its own staggered clock, independent of the ambient
    // swap cycle, so a style change ripples rather than waiting its turn.
    if (t.restyleAt && time >= t.restyleAt && !t.next) {
      t.next = pickEffect(state.genre);
      t.morph = 0;
      t.restyleAt = 0;
      if (t.label) t.label.querySelector('.effect').textContent = t.next.label;
    } else if (!t.next && !t.restyleAt && time >= t.swapAt) {
      t.next = pickEffect(state.genre);
      t.morph = 0;
      if (t.label) t.label.querySelector('.effect').textContent = t.next.label;
    }

    const view = viewFor(t, t === focused ? focusK : 0);
    if (t === focused) positionLabel(t, view);

    drawTile(t, t.fx, time, 1, view);

    if (t.next) {
      t.morph = Math.min(1, t.morph + dt / MORPH_SECONDS);
      // Smoothstep: a linear fade spends too long looking like a double exposure.
      const a = t.morph * t.morph * (3 - 2 * t.morph);
      drawTile(t, t.next, time, a, view);
      if (t.morph >= 1) {
        t.fx = t.next;
        t.next = null;
        t.swapAt = time + MORPH_MIN + Math.random() * MORPH_SPREAD;
      }
    }
  }

  // Once the collapse has fully settled, drop the focus geometry.
  if (!focused && state.focus.s.x === 0) document.body.classList.remove('focused');

  state.frames++;
  accFrames++;
  acc += now - last;
  last = now;
  if (acc > 500) {
    state.fps = Math.round((accFrames * 1000) / acc);
    acc = 0; accFrames = 0;
    statusEl.textContent = `${state.tiles.length} · ${state.fps}fps`;
  }
}

// ------------------------------------------------------------------ events

const hud = document.getElementById('hud');
const genreList = document.getElementById('genres');
const toggle = document.getElementById('toggle');
const currentGenre = document.getElementById('currentGenre');
const help = document.getElementById('help');

const GENRES = ['everything', ...Object.keys(PRESETS)];
const KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'];

// Chrome rests nearly transparent and wakes on any pointer movement, so the
// wall is never permanently covered but the controls are never more than a
// twitch away.
const WAKE_MS = 2600;
let sleepTimer;

function wake() {
  hud.classList.add('awake');
  clearTimeout(sleepTimer);
  sleepTimer = setTimeout(() => {
    // Stay awake while the list is open or something inside has focus.
    if (genreList.classList.contains('open') || hud.contains(document.activeElement)) return;
    hud.classList.remove('awake');
    // The caption goes with it. Leaving one tile labelled after everything else
    // has faded looks like a stuck element rather than a deliberate state.
    if (nearTile && nearTile.label) nearTile.label.classList.remove('near');
    nearTile = null;
  }, WAKE_MS);
}

// --------------------------------------------------------------- captions

// Only the tile under the pointer is captioned. Thirty-five permanent labels
// cover more of the wall than the header ever did.
let nearTile = null;

// Tiles are inset by the gutter, so their rectangles do not quite tile the
// canvas. Hit-testing against them literally means clicks landing in a gap
// match nothing. Expanding by half the gutter makes the wall contiguous.
const HIT_PAD = 4;

function tileAt(x, y) {
  return state.tiles.find(t =>
    x >= t.x - HIT_PAD && x <= t.x + t.w + HIT_PAD &&
    y >= t.y - HIT_PAD && y <= t.y + t.h + HIT_PAD);
}

function captionUnder(x, y) {
  if (labels.classList.contains('pinned') || state.focus.tile) return;
  const hit = tileAt(x, y);
  if (hit === nearTile) return;
  if (nearTile && nearTile.label) nearTile.label.classList.remove('near');
  if (hit && hit.label) hit.label.classList.add('near');
  nearTile = hit || null;
}

// ----------------------------------------------------------------- genres

function setGenre(name, originX, originY) {
  if (!GENRES.includes(name)) return;
  state.genre = name;
  currentGenre.textContent = name;
  for (const li of genreList.children) {
    li.setAttribute('aria-selected', String(li.dataset.genre === name));
  }
  // Ripple out of the corner the control lives in, unless told otherwise.
  restyle(originX ?? 0, originY ?? canvas.clientHeight);
}

for (const [i, name] of GENRES.entries()) {
  const li = document.createElement('li');
  li.dataset.genre = name;
  li.setAttribute('role', 'option');
  li.setAttribute('aria-selected', String(name === state.genre));
  li.innerHTML = `<span class="key">${KEYS[i] ?? ''}</span><span>${name}</span>`;
  li.onclick = (e) => { setGenre(name, e.clientX, e.clientY); openList(false); };
  genreList.appendChild(li);
}

function openList(open) {
  genreList.classList.toggle('open', open);
  toggle.setAttribute('aria-expanded', String(open));
  if (open) wake();
}

toggle.addEventListener('click', () => openList(!genreList.classList.contains('open')));

// ---------------------------------------------------------------- pointer

addEventListener('pointermove', (e) => {
  state.pointer.x = (e.clientX / innerWidth) * 2 - 1;
  state.pointer.y = (e.clientY / innerHeight) * 2 - 1;
  state.pointer.active = 1;
  wake();
  captionUnder(e.clientX, e.clientY);
});

addEventListener('pointerleave', () => { state.pointer.active = 0; });

addEventListener('pointerdown', (e) => {
  if (hud.contains(e.target) || help.contains(e.target)) return;
  openList(false);

  const hit = tileAt(e.clientX, e.clientY);

  if (state.focus.tile && hit !== state.focus.tile) {
    focusTile(null);              // clicking elsewhere while focused collapses
  } else if (hit) {
    focusTile(hit);
    document.body.classList.add('focused');
  }

  if (state.focus.tile) {
    // Only the focused tile keeps a caption.
    if (nearTile && nearTile.label) nearTile.label.classList.remove('near');
    nearTile = state.focus.tile;
    if (nearTile.label) nearTile.label.classList.add('near');
  }
});

// --------------------------------------------------------------- keyboard

const ACTIONS = {
  ' ': () => { state.paused = !state.paused; },
  r: () => buildTiles(),
  l: () => {
    const pinned = labels.classList.toggle('pinned');
    if (pinned && nearTile && nearTile.label) nearTile.label.classList.remove('near');
    nearTile = null;
  },
  f: () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  },
  h: () => document.body.classList.toggle('bare'),
  '?': () => { help.hidden = !help.hidden; },
  escape: () => {
    if (state.focus.tile) { focusTile(null); return; }
    openList(false);
    help.hidden = true;
  },
};

addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  const idx = KEYS.indexOf(e.key);
  if (idx >= 0 && idx < GENRES.length) {
    setGenre(GENRES[idx]);
    wake();
    e.preventDefault();
    return;
  }

  const action = ACTIONS[e.key.toLowerCase()] ?? ACTIONS[e.key];
  if (action) {
    action();
    wake();
    e.preventDefault();
  }
});

// ------------------------------------------------------------------ misc

let resizeTimer;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { resize(); nearTile = null; }, 200);
});

document.addEventListener('visibilitychange', () => { state.paused = document.hidden; });

// Lets the capture and test scripts drive the wall without depending on the
// chrome being visible, which it usually is not.
window.setGenre = setGenre;
window.__wall = state;

boot()
  .then(() => wake())
  .catch(err => {
    statusEl.textContent = `failed: ${err.message}`;
    console.error(err);
  });
