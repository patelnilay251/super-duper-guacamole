// Recursive canvas splitting, ported from the Python renderer.
//
// A uniform grid reads as a contact sheet. Splitting recursively gives tiles of
// genuinely different sizes, and a repair pass removes slivers, which otherwise
// read as seams rather than as tiles.

const MIN_SIDE = 150;
const MAX_ASPECT = 2.6;

function tooThin(r) {
  const aspect = r.w / r.h;
  const extreme = aspect > MAX_ASPECT || aspect < 1 / MAX_ASPECT;
  const splittable = aspect < 1 ? r.h >= MIN_SIDE * 2 : r.w >= MIN_SIDE * 2;
  return extreme && splittable;
}

export function splitCanvas(width, height, count, gap = 6) {
  let rects = [{ x: 0, y: 0, w: width, h: height }];

  while (rects.length < count) {
    const candidates = rects.filter(r => r.w >= MIN_SIDE * 2 || r.h >= MIN_SIDE * 2);
    if (!candidates.length) break;

    // Weight by area so large rectangles split first and sizes stay spread.
    const weights = candidates.map(r => Math.pow(r.w * r.h, 0.62));
    const total = weights.reduce((a, b) => a + b, 0);
    let roll = Math.random() * total, idx = 0;
    while (roll > weights[idx] && idx < weights.length - 1) roll -= weights[idx++];
    const target = candidates[idx];

    rects = rects.filter(r => r !== target);
    let vertical = target.w / target.h > 1;
    if (Math.random() < 0.18) vertical = !vertical;
    if (vertical && target.w < MIN_SIDE * 2) vertical = false;
    if (!vertical && target.h < MIN_SIDE * 2) vertical = true;

    const ratio = 0.36 + Math.random() * 0.28;
    if (vertical) {
      const cut = Math.max(MIN_SIDE, Math.min(target.w - MIN_SIDE, Math.round(target.w * ratio)));
      rects.push({ x: target.x, y: target.y, w: cut, h: target.h });
      rects.push({ x: target.x + cut, y: target.y, w: target.w - cut, h: target.h });
    } else {
      const cut = Math.max(MIN_SIDE, Math.min(target.h - MIN_SIDE, Math.round(target.h * ratio)));
      rects.push({ x: target.x, y: target.y, w: target.w, h: cut });
      rects.push({ x: target.x, y: target.y + cut, w: target.w, h: target.h - cut });
    }
  }

  for (let guard = 0; guard < count; guard++) {
    const slivers = rects.filter(tooThin);
    if (!slivers.length) break;
    const target = slivers.reduce((a, b) => (a.w * a.h > b.w * b.h ? a : b));
    rects = rects.filter(r => r !== target);
    if (target.w / target.h < 1) {
      const cut = Math.round(target.h * (0.42 + Math.random() * 0.16));
      rects.push({ x: target.x, y: target.y, w: target.w, h: cut });
      rects.push({ x: target.x, y: target.y + cut, w: target.w, h: target.h - cut });
    } else {
      const cut = Math.round(target.w * (0.42 + Math.random() * 0.16));
      rects.push({ x: target.x, y: target.y, w: cut, h: target.h });
      rects.push({ x: target.x + cut, y: target.y, w: target.w - cut, h: target.h });
    }
  }

  // Inset for the gutter, in CSS pixels.
  return rects.map(r => ({
    x: r.x + gap / 2,
    y: r.y + gap / 2,
    w: Math.max(1, r.w - gap),
    h: Math.max(1, r.h - gap),
  }));
}
