// GLSL ports of the Python effects engine.
//
// Every effect is per-pixel over a texture, which is exactly what a fragment
// shader is for. The one family that does not port is error diffusion: it is
// sequential by construction, and stays on the CPU.

export const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

// Shared prelude: cropping, tone handling, palettes, threshold matrices.
const COMMON = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uTex;      // source photograph
uniform sampler2D uNoise;    // 64x64 blue noise
uniform sampler2D uDepthTex; // precomputed depth: 1 near, 0 far
uniform vec4  uCrop;         // xy = offset, zw = scale, in texture space
uniform vec2  uRes;          // tile size in device pixels
uniform float uTime;         // seconds
uniform float uSeed;         // per-tile constant
uniform vec2  uPointer;      // -1..1 across the wall; (0,0) when absent
uniform float uPointerAmt;   // 0 disables interaction
uniform vec2  uTone;         // per-photo luminance percentiles (lo, hi)
uniform float uCropGamma;    // per-tile exposure, computed on the CPU
uniform vec3  uPalette[8];
uniform int   uPaletteLen;
uniform float uAlpha;       // morph blend weight; 1 when not transitioning
uniform float uCoarsen;     // >1 enlarges the process; how a tile is "defocused"
uniform float uDim;         // 0..1 toward the ground colour

const float PI = 3.14159265;

// Screen-space processes read pixels through here rather than gl_FragCoord
// directly. Dividing enlarges every cell, dot and matrix at once, which is what
// pushing a tile back looks like in this medium -- blurring a one-bit dither
// just turns it to grey mush and throws away the pattern entirely.
vec2 screenPx() { return gl_FragCoord.xy / max(uCoarsen, 0.001); }

// Applied to every fragment on the way out.
vec3 finish(vec3 c) { return mix(c, vec3(0.045, 0.045, 0.055), clamp(uDim, 0.0, 1.0)); }

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// WebGL's texture origin is bottom-left, but an image uploads top row first, so
// t=0 is the top of the photograph while vUv.y=0 is the bottom of the viewport.
// Without flipping y here the whole frame renders upside down -- which it did,
// unnoticed, until the differential harness compared it against the reference.
vec2 srcUv(vec2 uv) {
  return vec2(uCrop.x + uv.x * uCrop.z,
              uCrop.y + (1.0 - uv.y) * uCrop.w);
}

vec3 sample0(vec2 uv) { return texture(uTex, srcUv(uv)).rgb; }

// Percentile stretch (from the manifest) then a gamma shift pulling the crop
// mean toward mid grey -- the same two steps as normalize_tone.
//
// uCropGamma is computed once per tile on the CPU. It was originally derived
// here from nine textureLod samples, which cost nine extra texture fetches on
// every single pixel to recompute a value that is constant across the tile.
vec3 toneMap(vec3 c) {
  vec3 s = clamp((c - uTone.x) / max(1e-3, uTone.y - uTone.x), 0.0, 1.0);
  return pow(s, vec3(uCropGamma));
}

vec3 srcToned(vec2 uv) { return toneMap(sample0(uv)); }

// Monocular depth, estimated offline and shipped as a texture. 1 is near.
float depthAt(vec2 uv) { return texture(uDepthTex, srcUv(uv)).r; }

// Recursive Bayer, compact form. The +0.5/n^2 offset centres the matrix on 0.5;
// without it the mean sits low and the whole image darkens.
float bayer2(vec2 a) { a = floor(a); return fract(a.x / 2.0 + a.y * a.y * 0.75); }
float bayer4(vec2 a) { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
float bayer8(vec2 a) { return bayer4(0.5 * a) * 0.25 + bayer2(a); }

float threshold(vec2 px, int kind) {
  if (kind == 0) return bayer2(px) + 0.125;
  if (kind == 1) return bayer4(px) + 0.03125;
  if (kind == 2) return bayer8(px) + 0.0078125;
  if (kind == 3) return texture(uNoise, px / 64.0).r;   // blue noise
  return fract(sin(dot(px, vec2(12.9898, 78.233))) * 43758.5453);  // white
}

// Ordered dithering against a palette treated as a tonal ramp: quantise in
// luminance, not by nearest colour in RGB. Matching in RGB collapses a green
// photograph onto the light end of a four-green palette.
vec3 rampDither(float g, float t) {
  int n = uPaletteLen - 1;
  float lvl = clamp(floor(g * float(n) + t), 0.0, float(n));
  int i = int(lvl);
  for (int k = 0; k < 8; k++) { if (k == i) return uPalette[k]; }
  return uPalette[0];
}

// Screened dot: radius scales as sqrt(coverage) so printed area tracks tone.
float dotScreen(vec2 px, float coverage, float cell, float angle) {
  float ca = cos(angle), sa = sin(angle);
  vec2 r = vec2(px.x * ca - px.y * sa, px.x * sa + px.y * ca);
  vec2 d = mod(r, cell) - cell * 0.5;
  float dist = length(d) / (cell * 0.5);
  return step(dist, sqrt(clamp(coverage, 0.0, 1.0)) * 1.16);
}

// ---- noise -------------------------------------------------------------

float hash11(float p) { return fract(sin(p * 127.1) * 43758.5453); }
float hash21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float valueNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
             mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y);
}

// ---- filtering ---------------------------------------------------------

// The texture already carries a mipmap chain, so a blur is one biased fetch
// rather than a multi-tap kernel. Far cheaper than a real Gaussian and
// indistinguishable once it is used as a glow.
vec3 blurred(vec2 uv, float lod) { return textureLod(uTex, srcUv(uv), lod).rgb; }

float lumaAt(vec2 uv) { return luma(srcToned(uv)); }

vec2 sobel(vec2 uv) {
  vec2 t = 1.0 / uRes;
  float tl = lumaAt(uv + vec2(-t.x,  t.y)), l = lumaAt(uv + vec2(-t.x, 0.0)), bl = lumaAt(uv - t);
  float tr = lumaAt(uv + t),                r = lumaAt(uv + vec2( t.x, 0.0)), br = lumaAt(uv + vec2( t.x, -t.y));
  float tp = lumaAt(uv + vec2(0.0,  t.y)),  b = lumaAt(uv - vec2(0.0, t.y));
  return vec2((tr + 2.0 * r + br) - (tl + 2.0 * l + bl),
              (bl + 2.0 * b + br) - (tl + 2.0 * tp + tr));
}
`;

// --------------------------------------------------------------------------
// effect programs
// --------------------------------------------------------------------------

export const FRAG = {
  // Ordered / threshold dithering. uKind picks the matrix; the matrix drifts
  // with time so the grain crawls instead of sitting still.
  ordered: COMMON + `
uniform int uKind;
uniform float uScale;
void main() {
  float pulse = 1.0 + 0.35 * sin(uTime * 0.6 + uSeed * 6.28);
  float zoom = uScale * pulse * (1.0 + uPointerAmt * 0.9 * length(uPointer));
  vec2 px = screenPx() / max(zoom, 0.35);
  px += vec2(uTime * 3.0 * uSeed, uTime * 1.7);         // crawl
  float g = luma(srcToned(vUv));
  fragColor = vec4(finish(rampDither(g, threshold(px, uKind))), uAlpha);
}`,

  // Rotated halftone screen. The screen angle rotates continuously, which is
  // the single cheapest way to make a print-like tile feel alive.
  halftone: COMMON + `
uniform float uCell;
uniform vec3 uInk;
uniform vec3 uStock;
void main() {
  float angle = 0.5 + 0.25 * sin(uTime * 0.25 + uSeed * 6.28);
  float cell = uCell * (1.0 + 0.25 * sin(uTime * 0.4 + uSeed * 3.0));
  cell *= 1.0 + uPointerAmt * 1.6 * length(uPointer);
  float g = luma(srcToned(vUv));
  float ink = dotScreen(screenPx(), 1.0 - g, max(cell, 2.0), angle);
  fragColor = vec4(finish(mix(uStock, uInk, ink)), uAlpha);
}`,

  // False-colour ramp. Animating the ramp offset makes heat appear to flow.
  gradient: COMMON + `
uniform vec3 uRamp[6];
uniform int uRampLen;
void main() {
  float g = luma(srcToned(vUv));
  float drift = 0.12 * sin(uTime * 0.35 + uSeed * 6.28);
  g = clamp(g + drift + uPointerAmt * 0.25 * uPointer.x, 0.0, 1.0);
  float pos = g * float(uRampLen - 1);
  int i = int(floor(pos));
  float f = fract(pos);
  vec3 a = uRamp[0], b = uRamp[0];
  for (int k = 0; k < 6; k++) { if (k == i) a = uRamp[k]; if (k == i + 1) b = uRamp[k]; }
  fragColor = vec4(finish(mix(a, b, f)), uAlpha);
}`,

  // Two-ink screen print. Each ink has its own screen angle and its own
  // registration error, and the errors drift -- which is what sells it.
  riso: COMMON + `
uniform vec3 uInkA;
uniform vec3 uInkB;
uniform float uCell;
uniform float uSlip;
void main() {
  float cell = max(uCell * (1.0 + uPointerAmt * 1.2 * length(uPointer)), 2.5);

  vec2 slipA = uSlip * vec2(sin(uTime * 0.7 + uSeed * 6.0), cos(uTime * 0.5 + uSeed * 4.0));
  vec2 slipB = uSlip * vec2(cos(uTime * 0.6 + uSeed * 2.0), sin(uTime * 0.8 + uSeed * 5.0));

  vec3 ca = srcToned(vUv + slipA / uRes);
  vec3 cb = srcToned(vUv + slipB / uRes);

  float density = 1.0 - luma(ca);                       // first ink: overall tone
  float chroma  = clamp((cb.r - cb.b) * 1.6 + 0.5, 0.0, 1.0) * 0.85;

  float a = dotScreen(screenPx() + slipA, density * 0.95, cell, 0.35);
  float b = dotScreen(screenPx() + slipB, chroma,         cell, 1.20);

  vec3 out0 = vec3(1.0);
  out0 *= 1.0 - a * (1.0 - uInkA);                      // multiply onto paper
  out0 *= 1.0 - b * (1.0 - uInkB);
  fragColor = vec4(finish(out0), uAlpha);
}`,

  // Shadow-mask tube: scanlines, an RGB aperture stripe, and a roll bar
  // drifting down the tile the way an out-of-sync tube does.
  crt: COMMON + `
uniform float uCurve;
void main() {
  vec2 uv = vUv;
  vec2 c = uv - 0.5;
  uv += c * dot(c, c) * uCurve;                          // barrel distortion
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { fragColor = vec4(finish(vec3(0.02, 0.02, 0.03)), uAlpha); return; }

  float disp = 0.0016 * (1.0 + uPointerAmt * 2.0 * length(uPointer));
  vec3 col = vec3(srcToned(uv + vec2(disp, 0.0)).r, srcToned(uv).g, srcToned(uv - vec2(disp, 0.0)).b);

  float scan = 0.72 + 0.28 * step(0.5, fract(gl_FragCoord.y / 3.0));
  float roll = 1.0 + 0.11 * sin((uv.y + uTime * 0.12 + uSeed) * 40.0);
  vec3 mask = vec3(1.0);
  int stripe = int(mod(gl_FragCoord.x, 3.0));
  if (stripe == 0) mask = vec3(1.32, 0.86, 0.86);
  else if (stripe == 1) mask = vec3(0.86, 1.32, 0.86);
  else mask = vec3(0.86, 0.86, 1.32);

  col = clamp((col - 0.5) * 1.18 + 0.5, 0.0, 1.0);
  fragColor = vec4(finish(clamp(col * scan * roll * mask + 0.02, 0.0, 1.0)), uAlpha);
}`,

  // Radial lens dispersion. The amount breathes, and the pointer pulls it wider.
  chromatic: COMMON + `
uniform float uAmount;
void main() {
  float amt = uAmount * (1.0 + 0.4 * sin(uTime * 0.5 + uSeed * 6.28))
                      * (1.0 + uPointerAmt * 2.2 * length(uPointer));
  vec2 c = vUv - 0.5;
  vec3 col;
  col.r = srcToned(0.5 + c * (1.0 - amt)).r;
  col.g = srcToned(vUv).g;
  col.b = srcToned(0.5 + c * (1.0 + amt)).b;
  fragColor = vec4(finish(col), uAlpha);
}`,

  // Highlights bleeding into a glow, taken from a mip level rather than a
  // multi-tap kernel.
  bloom: COMMON + `
uniform float uThreshold;
uniform float uStrength;
void main() {
  vec3 base = srcToned(vUv);
  float th = uThreshold - 0.12 * sin(uTime * 0.4 + uSeed * 6.28) - uPointerAmt * 0.15 * length(uPointer);
  vec3 wide = toneMap(blurred(vUv, 4.5));
  vec3 glow = max(wide - th, vec3(0.0)) / max(1e-3, 1.0 - th);
  fragColor = vec4(finish(clamp(base + glow * uStrength, 0.0, 1.0)), uAlpha);
}`,

  // Jittered-grid Voronoi: each cell takes the colour at its own seed point.
  // The seeds orbit, so the facets shift like slow-moving glass.
  crystallize: COMMON + `
uniform float uCell;
void main() {
  float cell = uCell * (1.0 + uPointerAmt * 1.5 * length(uPointer));
  // Tile-local pixels, derived from vUv rather than gl_FragCoord. Every tile is
  // drawn through its own gl.viewport, but gl_FragCoord stays in framebuffer
  // coordinates -- feeding that back into a UV lands outside the texture and
  // clamps to an edge pixel, which is why distant tiles came out flat.
  vec2 px = vUv * uRes / (cell * max(uCoarsen, 0.001));
  vec2 base = floor(px);

  float bestD = 1e9;
  vec2 bestSeed = base;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 g = base + vec2(float(x), float(y));
      float h = hash21(g);
      vec2 jitter = 0.5 + 0.45 * vec2(sin(uTime * 0.5 + h * 6.28), cos(uTime * 0.4 + h * 5.1));
      vec2 seed = g + jitter;
      float d = dot(seed - px, seed - px);
      if (d < bestD) { bestD = d; bestSeed = seed; }
    }
  }
  fragColor = vec4(finish(srcToned(clamp(bestSeed * cell / uRes, 0.0, 1.0))), uAlpha);
}`,

  // Painterly edge-preserving filter: take the mean of the flattest quadrant.
  kuwahara: COMMON + `
uniform float uRadius;
void main() {
  vec2 t = 1.0 / uRes;
  float r = uRadius;
  vec3 bestMean = vec3(0.0);
  float bestVar = 1e9;

  for (int q = 0; q < 4; q++) {
    vec2 dir = vec2(q == 0 || q == 3 ? -1.0 : 1.0, q < 2 ? -1.0 : 1.0);
    vec3 sum = vec3(0.0); float sum2 = 0.0; float n = 0.0;
    for (int j = 0; j <= 3; j++) {
      for (int i = 0; i <= 3; i++) {
        vec2 o = dir * vec2(float(i), float(j)) * (r / 3.0) * t;
        vec3 c = srcToned(vUv + o);
        sum += c; sum2 += luma(c) * luma(c); n += 1.0;
      }
    }
    vec3 mean = sum / n;
    float variance = sum2 / n - luma(mean) * luma(mean);
    if (variance < bestVar) { bestVar = variance; bestMean = mean; }
  }
  fragColor = vec4(finish(clamp(bestMean, 0.0, 1.0)), uAlpha);
}`,

  // Monochrome terminal tube with glow bleed and a slow phosphor decay ripple.
  phosphor: COMMON + `
uniform vec3 uTint;
void main() {
  float g = luma(srcToned(vUv));
  float glow = luma(toneMap(blurred(vUv, 3.5)));
  g = clamp(g * 0.92 + glow * 0.28, 0.0, 1.0);
  g *= 0.86 + 0.14 * sin((vUv.y + uTime * 0.2 + uSeed) * 26.0);

  vec3 col = mix(vec3(0.02, 0.05, 0.03), uTint, pow(g, 1.15));
  float scan = 0.62 + 0.38 * step(0.5, fract(gl_FragCoord.y / 3.0));
  fragColor = vec4(finish(clamp(col * scan, 0.0, 1.0)), uAlpha);
}`,

  // Photocopy: crushed tone curve, toner speckle, and dropout that reshuffles
  // every second or so, as though the sheet were being fed again.
  xerox: COMMON + `
uniform float uBias;
void main() {
  float g = luma(srcToned(vUv));
  float pass = floor(uTime * 0.8 + uSeed * 10.0);          // a new copy each pass
  float hard = 1.0 / (1.0 + exp(-(g - uBias) * 13.0));
  float speckle = (hash21(screenPx() + pass * 37.0) - 0.5) * 0.32;
  hard = step(0.5, clamp(hard + speckle, 0.0, 1.0));
  hard = clamp(hard + step(0.994, hash21(screenPx() * 1.7 + pass)), 0.0, 1.0);
  fragColor = vec4(finish(mix(vec3(0.09, 0.09, 0.11), vec3(0.94, 0.93, 0.90), hard)), uAlpha);
}`,

  // Engraving: successive line sets cut in as the tone darkens.
  crosshatch: COMMON + `
uniform float uSpacing;
uniform vec3 uInk;
uniform vec3 uStock;
void main() {
  float g = luma(toneMap(blurred(vUv, 1.5)));
  float spacing = uSpacing * (1.0 + uPointerAmt * 1.4 * length(uPointer));
  float drift = uTime * 0.15 + uSeed * 6.28;
  float ink = 0.0;
  for (int k = 0; k < 4; k++) {
    float angle = float(k) * 0.7854 + 0.1 * sin(drift + float(k));
    float cut = 0.82 - float(k) * 0.20;
    float proj = gl_FragCoord.x * cos(angle) - gl_FragCoord.y * sin(angle);
    if (g < cut && mod(proj, spacing) < 1.35) ink = 1.0;
  }
  fragColor = vec4(finish(mix(uStock, uInk, ink)), uAlpha);
}`,

  // Sobel magnitude, drawn as light on dark.
  edges: COMMON + `
uniform float uGain;
void main() {
  vec2 g = sobel(vUv);
  float m = clamp(length(g) * uGain * (1.0 + uPointerAmt * 1.5 * length(uPointer)), 0.0, 1.0);
  m *= 0.85 + 0.15 * sin(uTime * 0.6 + uSeed * 6.28);
  fragColor = vec4(finish(vec3(m)), uAlpha);
}`,

  // Two-colour map with a posterised option, so the ramp shows as flat bands.
  duotone: COMMON + `
uniform vec3 uDark;
uniform vec3 uLight;
uniform float uLevels;
void main() {
  float g = luma(srcToned(vUv));
  g = clamp(g + 0.08 * sin(uTime * 0.3 + uSeed * 6.28) + uPointerAmt * 0.2 * uPointer.x, 0.0, 1.0);
  if (uLevels > 1.5) g = floor(g * uLevels) / (uLevels - 1.0);
  fragColor = vec4(finish(mix(uDark, uLight, clamp(g, 0.0, 1.0))), uAlpha);
}`,

  // Warp along a smooth noise field: heat haze, or liquid, depending on speed.
  displace: COMMON + `
uniform float uAmount;
void main() {
  float amt = uAmount * (1.0 + uPointerAmt * 1.8 * length(uPointer));
  vec2 p = vUv * 4.0 + uTime * 0.12;
  vec2 flow = vec2(valueNoise(p), valueNoise(p + 31.4)) - 0.5;
  fragColor = vec4(finish(srcToned(clamp(vUv + flow * amt, 0.001, 0.999))), uAlpha);
}`,

  // Horizontal datamosh: rows shift in blocks, re-cut a few times a second.
  datamosh: COMMON + `
uniform float uAmount;
void main() {
  float pass = floor(uTime * 1.6 + uSeed * 20.0);
  float band = floor(gl_FragCoord.y / (6.0 + 26.0 * hash11(floor(gl_FragCoord.y / 24.0) + pass)));
  float shift = (hash21(vec2(band, pass)) - 0.5) * uAmount
              * (1.0 + uPointerAmt * 2.0 * abs(uPointer.x));
  vec2 uv = vec2(fract(vUv.x + shift / uRes.x * 40.0), vUv.y);
  vec3 col = srcToned(uv);
  if (hash21(vec2(band, pass + 5.0)) > 0.88) col = col.gbr;   // channel swap
  fragColor = vec4(finish(col), uAlpha);
}`,

  // ---- depth-aware ------------------------------------------------------
  //
  // A photographic process is normally applied uniformly across a frame. With a
  // depth map it can vary through the scene instead: the screen gets finer as
  // the subject comes forward, and coarsens away into the distance the way an
  // atmosphere does.

  // Dither resolution keyed to distance: fine near, coarse far.
  depthDither: COMMON + `
uniform int uKind;
uniform float uNear;
uniform float uFar;
void main() {
  float d = depthAt(vUv);
  // Pointer sweeps the plane of best resolution back and forth through the scene.
  float focus = uPointerAmt > 0.5 ? clamp(0.5 - uPointer.y * 0.5, 0.0, 1.0) : 1.0;
  float sharp = 1.0 - abs(d - focus);
  float scale = mix(uFar, uNear, sharp);
  scale *= 1.0 + 0.15 * sin(uTime * 0.5 + uSeed * 6.28);

  vec2 px = screenPx() / max(scale, 0.35);
  px += vec2(uTime * 2.0 * uSeed, uTime * 1.1);
  fragColor = vec4(finish(rampDither(luma(srcToned(vUv)), threshold(px, uKind))), uAlpha);
}`,

  // Halftone whose screen ruling opens up with distance, like a print fading
  // into haze. The classic "atmospheric perspective" cue, done with dots.
  depthHalftone: COMMON + `
uniform float uNear;
uniform float uFar;
uniform vec3 uInk;
uniform vec3 uStock;
void main() {
  float d = depthAt(vUv);
  float cell = mix(uFar, uNear, d) * (1.0 + uPointerAmt * 0.8 * length(uPointer));
  float angle = 0.5 + 0.2 * sin(uTime * 0.2 + uSeed * 6.28);

  // Distance also washes the tone out, so far dots are small as well as sparse.
  float g = luma(srcToned(vUv));
  g = mix(min(g + 0.32, 1.0), g, d);

  float ink = dotScreen(screenPx(), 1.0 - g, max(cell, 2.0), angle);
  fragColor = vec4(finish(mix(uStock, uInk, ink)), uAlpha);
}`,

  // Depth cut into flat planes, each printed in its own ink -- a separation by
  // distance rather than by colour.
  depthPlanes: COMMON + `
uniform float uPlanes;
void main() {
  float d = depthAt(vUv);
  float drift = 0.06 * sin(uTime * 0.3 + uSeed * 6.28) + uPointerAmt * 0.12 * uPointer.y;
  float plane = floor(clamp(d + drift, 0.0, 0.999) * uPlanes);

  // Tone still dithers inside each plane, so the image survives the banding.
  float g = luma(srcToned(vUv));
  vec2 px = screenPx() / (1.0 + plane * 0.9);
  float t = threshold(px, 2);
  float lit = step(t, g);

  int idx = int(min(plane, float(uPaletteLen - 1)));
  vec3 ink = uPalette[0];
  for (int k = 0; k < 8; k++) { if (k == idx) ink = uPalette[k]; }
  fragColor = vec4(finish(mix(ink * 0.35, ink, lit)), uAlpha);
}`,

  // Parallax: the pointer shifts near pixels more than far ones, so a flat
  // photograph gains volume. Dithered on top, because the grain staying put
  // while the image slides underneath is what makes the illusion land.
  depthParallax: COMMON + `
uniform int uKind;
uniform float uAmount;
void main() {
  vec2 drive = uPointerAmt > 0.5 ? uPointer : vec2(sin(uTime * 0.4 + uSeed * 6.28), cos(uTime * 0.31));
  float d = depthAt(vUv);

  // Two refinement steps: sample depth, shift, resample. Cheap, and enough to
  // stop near objects smearing at the edges.
  vec2 uv = vUv - drive * uAmount * d;
  d = depthAt(clamp(uv, 0.0, 1.0));
  uv = clamp(vUv - drive * uAmount * d, 0.0, 1.0);

  float g = luma(srcToned(uv));
  fragColor = vec4(finish(rampDither(g, threshold(screenPx(), uKind))), uAlpha);
}`,

  // Atmosphere: the process holds near, then dissolves into fog with distance.
  depthFog: COMMON + `
uniform vec3 uFog;
uniform float uDensity;
void main() {
  float d = depthAt(vUv);
  float t = clamp(pow(1.0 - d, uDensity) + 0.1 * sin(uTime * 0.25 + uSeed * 6.28)
                  + uPointerAmt * 0.2 * uPointer.y, 0.0, 1.0);

  float g = luma(srcToned(vUv));
  float cell = mix(4.0, 13.0, t);
  float ink = dotScreen(screenPx(), (1.0 - g) * (1.0 - t * 0.75), cell, 0.6);

  vec3 near = mix(vec3(1.0), vec3(0.06, 0.06, 0.08), ink);
  fragColor = vec4(finish(mix(near, uFog, t * 0.85)), uAlpha);
}`,
};
