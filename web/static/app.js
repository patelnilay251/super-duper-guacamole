// Live gallery. Every tile owns its own refresh cycle, so the wall changes
// continuously and unevenly rather than blinking over all at once.

const grid = document.getElementById('grid');
const statusEl = document.getElementById('status');
const genresEl = document.getElementById('genres');

const SIZES = [
  { label: 'small',  px: 170 },
  { label: 'medium', px: 260 },
  { label: 'large',  px: 380 },
];

const state = {
  genre: 'everything',
  size: 1,
  paused: false,
  tiles: [],
  rendered: 0,
  latency: 400,        // EWMA of observed render+transfer time, ms
};

// Refresh rate is derived from measured latency rather than fixed, so the wall
// self-throttles instead of queueing: on a fast machine tiles turn over every
// few seconds, on a fraction-of-a-core host they slow down until the server
// keeps up. N tiles refreshing every T seconds demand N/T renders per second,
// against a capacity of roughly WORKERS/latency.
const WORKERS = 2, HEADROOM = 1.6;
function refreshInterval() {
  const n = Math.max(1, state.tiles.length);
  const base = (n * state.latency / WORKERS) * HEADROOM;
  return Math.min(90000, Math.max(4500, base));
}

const rand = (lo, hi) => lo + Math.random() * (hi - lo);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// How many grid rows a tile spans. Mixed heights are what stop the wall
// reading as a plain contact sheet.
const ROW = 10, GAP = 8;
function spanFor(index) {
  const shapes = [16, 20, 24, 26, 30, 34];
  const h = shapes[Math.floor(Math.random() * shapes.length)] * (state.size === 0 ? 0.7 : 1);
  return Math.max(12, Math.round(h));
}

class Tile {
  constructor(index) {
    this.index = index;
    this.timer = null;

    this.el = document.createElement('figure');
    this.el.className = 'tile loading';
    this.el.style.gridRowEnd = `span ${spanFor(index)}`;
    this.el.tabIndex = 0;

    this.img = document.createElement('img');
    this.img.alt = '';
    this.img.decoding = 'async';

    this.cap = document.createElement('figcaption');
    this.effect = document.createElement('span');
    this.effect.className = 'effect';
    this.credit = document.createElement('span');
    this.credit.className = 'credit';
    this.cap.append(this.effect, this.credit);

    this.el.append(this.img, this.cap);
    this.el.addEventListener('click', () => this.load());
  }

  // Ask for exactly the device pixels we will display. Dithering is a
  // per-pixel pattern; resampling it in the browser turns it to mush.
  targetSize() {
    const r = this.el.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    return {
      w: Math.max(80, Math.round((r.width || 240) * dpr)),
      h: Math.max(80, Math.round((r.height || 200) * dpr)),
    };
  }

  async load() {
    const { w, h } = this.targetSize();
    const url = `/api/tile.png?genre=${encodeURIComponent(state.genre)}&w=${w}&h=${h}&_=${Math.random()}`;
    this.el.classList.add('loading');
    const started = performance.now();
    try {
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) throw new Error(res.status);

      // Metadata rides in percent-encoded headers, since HTTP headers are
      // latin-1 and effect names contain en-dashes.
      const decode = (v) => { try { return decodeURIComponent(v || ''); } catch { return v || ''; } };
      const effect = decode(res.headers.get('X-Effect')) || 'unknown';
      const author = decode(res.headers.get('X-Author'));
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);

      await new Promise((ok, fail) => {
        this.img.onload = ok;
        this.img.onerror = fail;
        this.img.src = objectUrl;
      });

      if (this.lastUrl) URL.revokeObjectURL(this.lastUrl);
      this.lastUrl = objectUrl;

      this.effect.textContent = effect;
      this.credit.textContent = author ? `photo · ${author}` : '';
      this.el.setAttribute('aria-label', `${effect}${author ? `, photograph by ${author}` : ''}`);
      this.img.classList.add('ready');
      this.el.classList.remove('loading');

      // Blend into the latency estimate that drives the refresh rate.
      state.latency = state.latency * 0.8 + (performance.now() - started) * 0.2;
      state.rendered++;
      updateStatus();
    } catch (err) {
      this.el.classList.remove('loading');
    }
    this.schedule();
  }

  schedule() {
    clearTimeout(this.timer);
    if (state.paused) return;
    // Wide jitter around the adaptive interval, so tiles never fall into step.
    const t = refreshInterval();
    this.timer = setTimeout(() => this.load(), rand(t * 0.7, t * 1.5));
  }

  stop() { clearTimeout(this.timer); }

  destroy() {
    this.stop();
    if (this.lastUrl) URL.revokeObjectURL(this.lastUrl);
    this.el.remove();
  }
}

function updateStatus() {
  const label = state.paused ? 'paused' : 'live';
  const every = Math.round(refreshInterval() / 1000);
  statusEl.textContent =
    `${state.tiles.length} tiles · ${label} · ${state.rendered} renders · `
    + `${Math.round(state.latency)}ms · ~${every}s`;
}

// Fill the viewport plus a little, so there is something below the fold.
function tileCount() {
  const px = SIZES[state.size].px;
  const cols = Math.max(1, Math.floor(innerWidth / Math.min(px, innerWidth * 0.46)));
  const rows = Math.ceil(innerHeight / (px * 0.8)) + 1;
  return Math.min(60, Math.max(6, cols * rows));
}

function build() {
  state.tiles.forEach(t => t.destroy());
  state.tiles = [];
  grid.innerHTML = '';

  const n = tileCount();
  for (let i = 0; i < n; i++) {
    const tile = new Tile(i);
    state.tiles.push(tile);
    grid.appendChild(tile.el);
  }
  updateStatus();

  // Stagger the first paint so the server's render queue is not hit at once.
  state.tiles.forEach((tile, i) => setTimeout(() => tile.load(), i * (reduceMotion ? 40 : 110)));
}

async function loadGenres() {
  try {
    const { genres } = await (await fetch('/api/genres')).json();
    genresEl.innerHTML = '';
    for (const g of genres) {
      const b = document.createElement('button');
      b.className = 'chip';
      b.type = 'button';
      b.textContent = g.label;
      b.title = g.blurb;
      b.setAttribute('aria-pressed', String(g.id === state.genre));
      b.addEventListener('click', () => {
        state.genre = g.id;
        [...genresEl.children].forEach(c => c.setAttribute('aria-pressed', String(c === b)));
        build();
      });
      genresEl.appendChild(b);
    }
  } catch {
    statusEl.textContent = 'could not reach the server';
  }
}

document.getElementById('pause').addEventListener('click', (e) => {
  state.paused = !state.paused;
  e.currentTarget.setAttribute('aria-pressed', String(state.paused));
  e.currentTarget.textContent = state.paused ? '▶ resume' : '⏸ pause';
  state.tiles.forEach(t => (state.paused ? t.stop() : t.schedule()));
  updateStatus();
});

document.getElementById('shuffle').addEventListener('click', () => {
  state.tiles.forEach((t, i) => setTimeout(() => t.load(), i * 60));
});

document.getElementById('size').addEventListener('click', () => {
  state.size = (state.size + 1) % SIZES.length;
  document.getElementById('sizeLabel').textContent = SIZES[state.size].label;
  document.documentElement.style.setProperty('--tile', `${SIZES[state.size].px}px`);
  build();
});

// Stop rendering work while the tab is hidden -- pointless on a phone.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) state.tiles.forEach(t => t.stop());
  else if (!state.paused) state.tiles.forEach(t => t.schedule());
});

let resizeTimer;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const wanted = tileCount();
    if (Math.abs(wanted - state.tiles.length) > 2) build();
  }, 400);
});

document.documentElement.style.setProperty('--tile', `${SIZES[state.size].px}px`);
loadGenres().then(build);
