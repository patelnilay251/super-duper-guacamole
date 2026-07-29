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

const float PI = 3.14159265;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec2 srcUv(vec2 uv) { return uCrop.xy + uv * uCrop.zw; }

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
  vec2 px = gl_FragCoord.xy / max(zoom, 0.35);
  px += vec2(uTime * 3.0 * uSeed, uTime * 1.7);         // crawl
  float g = luma(srcToned(vUv));
  fragColor = vec4(rampDither(g, threshold(px, uKind)), 1.0);
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
  float ink = dotScreen(gl_FragCoord.xy, 1.0 - g, max(cell, 2.0), angle);
  fragColor = vec4(mix(uStock, uInk, ink), 1.0);
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
  fragColor = vec4(mix(a, b, f), 1.0);
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

  float a = dotScreen(gl_FragCoord.xy + slipA, density * 0.95, cell, 0.35);
  float b = dotScreen(gl_FragCoord.xy + slipB, chroma,         cell, 1.20);

  vec3 out0 = vec3(1.0);
  out0 *= 1.0 - a * (1.0 - uInkA);                      // multiply onto paper
  out0 *= 1.0 - b * (1.0 - uInkB);
  fragColor = vec4(out0, 1.0);
}`,
};
