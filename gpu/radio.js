// The radio: music, and a clock the wall can move to.
//
// Two ideas shape this.
//
// The clock is derived from each track's stated BPM, not from listening to the
// audio. Beat detection jitters, and its mistakes are visible -- a tile changing
// a beat late reads as a bug. A clock is exact, costs nothing, and every track
// in the corpus carries a tempo its own artist wrote down.
//
// Nothing here is required. The wall has its own drift and must look right in
// silence, which is how most people will meet it. The radio adds a layer on
// top; it never becomes the reason anything moves.

const BEATS_PER_BAR = 4;
const PHRASE_BARS = 8;
const CROSSFADE = 4.0;      // seconds of overlap between tracks
const ENERGY_ATTACK = 0.30; // how fast the smoothed level rises
const ENERGY_DECAY = 0.06;  // and falls -- slow, so it reads as a mood not a meter

export class Radio {
  constructor() {
    this.tracks = [];
    this.order = [];
    this.at = 0;
    this.playing = false;
    this.energy = 0;
    this.ready = false;

    this.bar = -1;
    this.phrase = -1;
    this.handlers = { bar: [], phrase: [], track: [] };

    this.ctx = null;
    this.decks = [];      // two, so a track can fade into the next
    this.live = 0;
  }

  on(event, fn) { this.handlers[event].push(fn); return this; }
  emit(event, arg) { for (const fn of this.handlers[event]) fn(arg); }

  get current() { return this.tracks[this.order[this.at]] ?? null; }

  async load(url = 'audio.json') {
    const res = await fetch(url);
    if (!res.ok) return false;
    const data = await res.json();
    this.tracks = data.tracks ?? [];
    if (!this.tracks.length) return false;

    this.order = this.tracks.map((_, i) => i);
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
    this.ready = true;
    return true;
  }

  // Built on the first play, because an AudioContext created before a gesture
  // starts suspended and browsers increasingly complain about it.
  init() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.bins = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.connect(this.ctx.destination);

    for (let i = 0; i < 2; i++) {
      const el = new Audio();
      el.crossOrigin = 'anonymous';
      el.preload = 'none';
      const gain = this.ctx.createGain();
      gain.gain.value = 0;
      this.ctx.createMediaElementSource(el).connect(gain);
      gain.connect(this.analyser);
      this.decks.push({ el, gain, track: null });
    }
  }

  async toggle() {
    if (this.playing) return this.pause();
    return this.play();
  }

  async play() {
    if (!this.ready) return false;
    this.init();
    if (this.ctx.state === 'suspended') await this.ctx.resume();

    // Re-cue whenever the deck is holding something other than the current
    // selection -- which is the case after picking a different track while
    // paused, not just on the very first play.
    const deck = this.decks[this.live];
    if (deck.track !== this.current) await this.cue(deck, this.current);
    try {
      await deck.el.play();
    } catch {
      return false;                     // autoplay refused; caller shows the button
    }
    this.fade(deck, 1, 0.8);
    this.playing = true;
    this.emit('track', this.current);
    return true;
  }

  pause() {
    this.playing = false;
    for (const d of this.decks) d.el.pause();
    return false;
  }

  async cue(deck, track) {
    deck.track = track;
    deck.el.src = track.src;
    deck.el.load();
  }

  fade(deck, to, seconds) {
    const now = this.ctx.currentTime;
    deck.gain.gain.cancelScheduledValues(now);
    deck.gain.gain.setValueAtTime(deck.gain.gain.value, now);
    deck.gain.gain.linearRampToValueAtTime(to, now + seconds);
  }

  advance() { return this.goTo((this.at + 1) % this.order.length); }

  // Jump to a position in the shuffled order. `seconds` controls the handover:
  // a track running out gets the long overlap, but someone picking from a list
  // wants the change to feel like a response, so that fade is much shorter.
  async goTo(at, seconds = CROSSFADE) {
    if (at === this.at && this.playing) return;
    this.at = ((at % this.order.length) + this.order.length) % this.order.length;

    // Picked while paused. Hand straight to play() rather than touching a deck:
    // before the first play there are no decks at all, because the audio graph
    // is not built until a gesture arrives.
    if (!this.playing) return this.play();

    const next = this.decks[1 - this.live];
    await this.cue(next, this.current);
    try { await next.el.play(); } catch { return; }

    const outgoing = this.decks[this.live];
    this.fade(outgoing, 0, seconds);
    this.fade(next, 1, seconds);
    setTimeout(() => { outgoing.el.pause(); outgoing.track = null; }, seconds * 1000 + 200);

    this.live = 1 - this.live;
    this.bar = -1;                      // the new track restarts the count
    this.phrase = -1;
    this.emit('track', this.current);
  }

  /** Pick from the list: a shorter fade, because this one is a response. */
  select(index) { return this.goTo(index, 1.2); }

  // Called once a frame. Advances the clock, fires bar and phrase events, and
  // keeps the smoothed level up to date.
  tick() {
    if (!this.playing || !this.ctx) return;
    const deck = this.decks[this.live];
    const track = deck.track;
    if (!track) return;

    // Level, smoothed asymmetrically: quick to notice a swell, slow to forget
    // it. A fast release would make this a volume meter, which is the thing we
    // are trying not to build.
    this.analyser.getByteFrequencyData(this.bins);
    let sum = 0;
    for (let i = 0; i < this.bins.length; i++) sum += this.bins[i];
    const level = sum / (this.bins.length * 255);
    const k = level > this.energy ? ENERGY_ATTACK : ENERGY_DECAY;
    this.energy += (level - this.energy) * k;

    const time = deck.el.currentTime;
    if (track.bpm) {
      const beats = time * track.bpm / 60;
      const bar = Math.floor(beats / BEATS_PER_BAR);
      if (bar !== this.bar) {
        this.bar = bar;
        this.emit('bar', bar);
        const phrase = Math.floor(bar / PHRASE_BARS);
        if (phrase !== this.phrase) {
          this.phrase = phrase;
          this.emit('phrase', phrase);
        }
      }
    }

    // Hand over before the end rather than at it, so the two overlap.
    const left = (deck.el.duration || Infinity) - time;
    if (left < CROSSFADE && !this.handing) {
      this.handing = true;
      this.advance().finally(() => { this.handing = false; });
    }
  }
}
