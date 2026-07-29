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

async function boot() {
  const manifest = await (await fetch('manifest.json')).json();
  state.photos = manifest.photos;

  const [imgs, depthImgs, noiseImg] = await Promise.all([
    Promise.all(state.photos.map(p => loadImage(p.src))),
    Promise.all(state.photos.map(p => p.depth ? loadImage(p.depth) : null)),
    loadImage('bluenoise.png'),
  ]);

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

  return { lo, hi, gamma: Math.min(2.0, Math.max(0.5, Math.log(0.48) / Math.log(mean))) };
}

// ------------------------------------------------------------------ layout

function buildTiles() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const rects = splitCanvas(w, h, tileTarget(w, h));

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

function drawTile(t, fx, time, alpha) {
  const { p, uniforms: u } = state.programs[fx.program];
  gl.useProgram(p);

  const d = state.dpr;
  // gl.viewport origin is bottom-left; the layout is top-left.
  gl.viewport(
    Math.round(t.x * d),
    Math.round((canvas.clientHeight - t.y - t.h) * d),
    Math.round(t.w * d),
    Math.round(t.h * d),
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

  gl.uniform4f(u.uCrop, t.crop[0], t.crop[1], t.crop[2], t.crop[3]);
  gl.uniform1f(u.uCropGamma, t.tone.gamma);
  gl.uniform2f(u.uRes, t.w * d, t.h * d);
  gl.uniform1f(u.uTime, time);
  gl.uniform1f(u.uSeed, t.seed);
  gl.uniform2f(u.uPointer, state.pointer.x, state.pointer.y);
  gl.uniform1f(u.uPointerAmt, state.pointer.active);
  gl.uniform2f(u.uTone, t.tone.lo, t.tone.hi);

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

  for (const t of state.tiles) {
    if (!t.next && time >= t.swapAt) {
      t.next = pickEffect(state.genre);
      t.morph = 0;
      if (t.label) t.label.querySelector('.effect').textContent = t.next.label;
    }

    drawTile(t, t.fx, time, 1);

    if (t.next) {
      t.morph = Math.min(1, t.morph + dt / MORPH_SECONDS);
      // Smoothstep: a linear fade spends too long looking like a double exposure.
      const a = t.morph * t.morph * (3 - 2 * t.morph);
      drawTile(t, t.next, time, a);
      if (t.morph >= 1) {
        t.fx = t.next;
        t.next = null;
        t.swapAt = time + MORPH_MIN + Math.random() * MORPH_SPREAD;
      }
    }
  }

  state.frames++;
  accFrames++;
  acc += now - last;
  last = now;
  if (acc > 500) {
    state.fps = Math.round((accFrames * 1000) / acc);
    acc = 0; accFrames = 0;
    statusEl.textContent =
      `${state.tiles.length} tiles · ${state.fps} fps · ${state.photos.length} photos · gpu`;
  }
}

// ------------------------------------------------------------------ events

addEventListener('pointermove', (e) => {
  state.pointer.x = (e.clientX / innerWidth) * 2 - 1;
  state.pointer.y = (e.clientY / innerHeight) * 2 - 1;
  state.pointer.active = 1;
});
addEventListener('pointerleave', () => { state.pointer.active = 0; });

let resizeTimer;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(resize, 200);
});

document.addEventListener('visibilitychange', () => { state.paused = document.hidden; });

document.getElementById('shuffle').addEventListener('click', buildTiles);
document.getElementById('freeze').addEventListener('click', (e) => {
  state.paused = !state.paused;
  e.currentTarget.textContent = state.paused ? '▶ resume' : '⏸ freeze';
});

const chips = document.getElementById('genres');
for (const name of ['everything', ...Object.keys(PRESETS)]) {
  const b = document.createElement('button');
  b.className = 'chip';
  b.textContent = name;
  b.setAttribute('aria-pressed', String(name === state.genre));
  b.onclick = () => {
    state.genre = name;
    [...chips.children].forEach(c => c.setAttribute('aria-pressed', String(c === b)));
    buildTiles();
  };
  chips.appendChild(b);
}

boot().catch(err => {
  statusEl.textContent = `failed: ${err.message}`;
  console.error(err);
});
