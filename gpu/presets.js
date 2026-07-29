// Effect presets: the palettes, ramps and inks from the Python engine, bound to
// the shader programs that implement them.

const rgb = (...c) => c.map(v => v / 255);

const PALETTES = {
  mono: [rgb(0, 0, 0), rgb(255, 255, 255)],
  gameboy: [rgb(15, 56, 15), rgb(48, 98, 48), rgb(139, 172, 15), rgb(155, 188, 15)],
  cyanotype: [rgb(8, 22, 48), rgb(24, 62, 110), rgb(68, 124, 176), rgb(150, 196, 224), rgb(235, 245, 252)],
  ember: [rgb(20, 12, 28), rgb(94, 30, 40), rgb(196, 84, 46), rgb(240, 176, 96), rgb(255, 240, 205)],
  sepia: [rgb(32, 22, 16), rgb(94, 68, 46), rgb(168, 130, 92), rgb(226, 202, 168), rgb(252, 244, 230)],
};

const RAMPS = {
  ironbow: [rgb(0, 0, 12), rgb(46, 8, 92), rgb(148, 20, 108), rgb(232, 92, 44), rgb(255, 190, 40), rgb(255, 252, 220)],
  magma: [rgb(0, 0, 8), rgb(70, 16, 90), rgb(168, 42, 90), rgb(244, 106, 62), rgb(252, 226, 176), rgb(252, 226, 176)],
  arctic: [rgb(4, 8, 28), rgb(18, 70, 130), rgb(60, 160, 200), rgb(170, 226, 240), rgb(250, 252, 255), rgb(250, 252, 255)],
  sodium: [rgb(6, 4, 2), rgb(72, 26, 4), rgb(176, 76, 8), rgb(244, 160, 28), rgb(255, 240, 190), rgb(255, 240, 190)],
};

const RISO_INKS = [
  ['fluoro pink · blue', rgb(255, 72, 176), rgb(0, 120, 191)],
  ['yellow · blue', rgb(255, 232, 0), rgb(0, 120, 191)],
  ['red · teal', rgb(255, 102, 94), rgb(0, 169, 157)],
  ['purple · green', rgb(118, 82, 205), rgb(0, 169, 92)],
  ['orange · navy', rgb(255, 108, 47), rgb(26, 44, 124)],
];

const KIND = { bayer2: 0, bayer4: 1, bayer8: 2, blue: 3, white: 4 };

// Each entry returns the per-tile draw description.
function ordered(label, kind, palette, scale) {
  return { label, program: 'ordered', kind: KIND[kind], palette: PALETTES[palette], uniforms: { uScale: scale } };
}

function halftone(label, cell, ink, stock) {
  return { label, program: 'halftone', uniforms: { uCell: cell, uInk: ink, uStock: stock } };
}

function gradient(label, ramp) {
  return { label, program: 'gradient', rampLen: RAMPS[ramp].length, uniforms: { uRamp: RAMPS[ramp] } };
}

function riso(label, a, b, cell, slip) {
  return { label, program: 'riso', uniforms: { uInkA: a, uInkB: b, uCell: cell, uSlip: slip } };
}

const INK = rgb(18, 16, 14), STOCK = rgb(247, 241, 226);

const shader = (label, program, uniforms) => ({ label, program, uniforms });

function duotone(label, dark, light, levels = 0) {
  return shader(label, 'duotone', { uDark: dark, uLight: light, uLevels: levels });
}

export const PRESETS = {
  dither: [
    ordered('bayer 2×2', 'bayer2', 'mono', 1),
    ordered('bayer 4×4', 'bayer4', 'mono', 1),
    ordered('bayer 8×8', 'bayer8', 'mono', 1),
    ordered('blue noise', 'blue', 'mono', 1),
    ordered('white noise', 'white', 'mono', 1),
    ordered('bayer 4×4 · game boy', 'bayer4', 'gameboy', 1),
    ordered('bayer 8×8 · ember', 'bayer8', 'ember', 1),
    ordered('blue noise · cyanotype', 'blue', 'cyanotype', 1),
    ordered('bayer 8×8 · sepia', 'bayer8', 'sepia', 1),
    ordered('chunky bayer', 'bayer4', 'mono', 2.5),
    ordered('chunky game boy', 'bayer2', 'gameboy', 3.0),
  ],
  newsprint: [
    halftone('screen 45°', 7, INK, STOCK),
    halftone('screen fine', 4, INK, STOCK),
    halftone('screen coarse', 12, INK, STOCK),
    halftone('screen 60°', 6, INK, STOCK),
    ordered('atkinson-ish', 'blue', 'mono', 1),
    ordered('bayer 8×8', 'bayer8', 'mono', 1),
  ],
  thermal: [
    gradient('ironbow', 'ironbow'),
    gradient('magma', 'magma'),
    gradient('arctic', 'arctic'),
    gradient('sodium', 'sodium'),
  ],
  risograph: RISO_INKS.flatMap(([label, a, b]) => [
    riso(label, a, b, 5, 2),
    riso(`${label} · coarse`, a, b, 9, 3.5),
  ]),

  signal: [
    shader('crt', 'crt', { uCurve: 0.12 }),
    shader('crt · flat', 'crt', { uCurve: 0.0 }),
    shader('chromatic', 'chromatic', { uAmount: 0.02 }),
    shader('chromatic · wide', 'chromatic', { uAmount: 0.05 }),
    shader('bloom', 'bloom', { uThreshold: 0.62, uStrength: 0.9 }),
    shader('bloom · heavy', 'bloom', { uThreshold: 0.48, uStrength: 1.4 }),
    shader('datamosh', 'datamosh', { uAmount: 26 }),
    shader('datamosh · violent', 'datamosh', { uAmount: 70 }),
    shader('displace', 'displace', { uAmount: 0.05 }),
    shader('displace · liquid', 'displace', { uAmount: 0.14 }),
  ],

  terminal: [
    shader('p1 green', 'phosphor', { uTint: rgb(110, 255, 140) }),
    shader('p3 amber', 'phosphor', { uTint: rgb(255, 182, 66) }),
    shader('ice', 'phosphor', { uTint: rgb(150, 220, 255) }),
    shader('vector', 'edges', { uGain: 0.85 }),
    shader('vector · hot', 'edges', { uGain: 1.7 }),
  ],

  press: [
    shader('engraving', 'crosshatch', { uSpacing: 7, uInk: INK, uStock: STOCK }),
    shader('engraving fine', 'crosshatch', { uSpacing: 4, uInk: INK, uStock: STOCK }),
    shader('engraving coarse', 'crosshatch', { uSpacing: 11, uInk: INK, uStock: STOCK }),
    shader('xerox', 'xerox', { uBias: 0.5 }),
    shader('xerox · overexposed', 'xerox', { uBias: 0.58 }),
    shader('xerox · underexposed', 'xerox', { uBias: 0.42 }),
    duotone('duotone · ink', rgb(16, 24, 52), rgb(236, 232, 220)),
    duotone('duotone · rust', rgb(28, 14, 18), rgb(244, 186, 122)),
    duotone('posterised · ink', rgb(16, 24, 52), rgb(236, 232, 220), 5),
    duotone('posterised · rust', rgb(28, 14, 18), rgb(244, 186, 122), 4),
  ],

  painterly: [
    shader('kuwahara', 'kuwahara', { uRadius: 5 }),
    shader('kuwahara · heavy', 'kuwahara', { uRadius: 7 }),
    shader('crystallize', 'crystallize', { uCell: 20 }),
    shader('crystallize · fine', 'crystallize', { uCell: 9 }),
    shader('crystallize · coarse', 'crystallize', { uCell: 26 }),
  ],
};

export function pick(genre) {
  const pool = genre === 'everything'
    ? Object.values(PRESETS).flat()
    : PRESETS[genre] || Object.values(PRESETS).flat();
  const base = pool[Math.floor(Math.random() * pool.length)];
  return {
    ...base,
    zoom: 1 + Math.random() * 1.2,
    cropX: Math.random(),
    cropY: Math.random() * 0.85,
  };
}
