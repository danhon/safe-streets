// Shared by the zoo and the day dial. DATA is public_view.PublicDay.as_json():
// the day's listed passes, each only {q: the time bucket of the day, speed
// (whole, in the unit), rough, dir, kind: car|two-wheeler}. No seconds, no
// colours, no makes, no pictures. `mph` below is the speed in DATA.unit.
const UNIT = DATA.unit, TO_MPH = UNIT === "km/h" ? 1 / 1.609344 : 1;
const PER_HOUR = 60 / DATA.bucket_min, BUCKETS = 24 * PER_HOUR;
const LIMIT = DATA.limit;
const P = DATA.passes.map(p => ({q: p.q, mph: p.speed, rough: p.rough, dir: p.dir, kind: p.kind === "two-wheeler" ? "bike" : "car"}));
const over = p => p.mph > LIMIT;

function hourLabel(h) {
  const ap = h < 12 ? "am" : "pm", hh = h % 12 || 12;
  return `${hh} ${ap}`;
}
function quarterLabel(q) {
  const h = Math.floor(q / PER_HOUR), m = (q % PER_HOUR) * DATA.bucket_min, ap = h < 12 ? "am" : "pm", hh = h % 12 || 12;
  return `${hh}:${String(m).padStart(2, "0")} ${ap}`;
}
function quarterRange(q) {
  return `${quarterLabel(q)}–${quarterLabel(q + 1)}`;
}
const STATS = (() => {
  const byHour = Array.from({length: 24}, () => []);
  P.forEach(p => byHour[Math.floor(p.q / PER_HOUR)].push(p));
  const n = DATA.total, nOver = DATA.over;
  const fastest = P.reduce((a, b) => (b.mph > a.mph ? b : a), P[0]);
  const hours = byHour.map((ps, h) => ({h, ps, n: ps.length,
    over: ps.filter(over).length,
    max: ps.length ? Math.max(...ps.map(p => p.mph)) : 0,
    median: ps.length ? ps.map(p => p.mph).sort((a, b) => a - b)[Math.floor(ps.length / 2)] : 0}));
  const busiest = hours.reduce((a, b) => (b.n > a.n ? b : a));
  const worst = hours.filter(h => h.n >= 5).reduce((a, b) => (b.over / b.n > a.over / a.n ? b : a), {over: 0, n: 1});
  return {n, nOver, pctOver: P.length ? Math.round(100 * nOver / P.length) : 0, fastest, hours, busiest, worst,
          firstHour: (hours.find(h => h.n) || {h: 0}).h, lastHour: ([...hours].reverse().find(h => h.n) || {h: 23}).h};
})();
// A stable pseudo-random per pass (index-seeded), so drawings don't jump on reload.
function rnd(seed) {
  let x = Math.sin(seed * 9301 + 49297) * 233280;
  return x - Math.floor(x);
}
const DAY_NAME = new Date(DATA.date + "T12:00:00").toLocaleDateString("en-US", {weekday: "long", month: "long", day: "numeric"});

// Day dials: "full" maps 360° to the whole day; "sofar" to midnight up to the
// end of the last quarter hour with a vehicle in it, so nothing sits empty.
const END_Q = P.length ? Math.max(...P.map(p => p.q)) + 1 : BUCKETS;
const DIAL = {
  mode: (() => { try { return localStorage.getItem("dialMode") || "full"; } catch (e) { return "full"; } })(),
  span() { return this.mode === "full" ? BUCKETS : END_Q; },
  set(m) { this.mode = m; try { localStorage.setItem("dialMode", m); } catch (e) {} },
  // hour marks for the current span: ticks every hour, labels at about four of them
  marks() {
    const hours = Math.ceil(this.span() / PER_HOUR), every = this.mode === "full" ? 6 : Math.max(1, Math.round(hours / 4));
    return Array.from({length: hours}, (_, h) => ({h, f: h * PER_HOUR / this.span(), major: h % every === 0}));
  },
  caption() { return this.mode === "full" ? "360° = the whole day, midnight to midnight" : `360° = midnight to ${quarterLabel(END_Q)}, the day so far`; },
};
// A two-way switch for any dial page; onChange runs after the mode is set.
function dialToggle(el, onChange) {
  el.innerHTML = `<span class="dt">${["full", "sofar"].map(m => `<button data-m="${m}" aria-pressed="${DIAL.mode === m}">${m === "full" ? "Full day" : "So far"}</button>`).join("")}</span><span class="dtc"></span>`;
  const style = document.createElement("style");
  style.textContent = `.dt{display:inline-flex;border:1px solid var(--ring);border-radius:999px;overflow:hidden}.dt button{border:0;border-radius:0;margin:0;padding:7px 14px;background:none;color:var(--soft);font:inherit;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;cursor:pointer}.dt button[aria-pressed=true]{background:var(--ink);color:var(--bg)}.dtc{color:var(--soft);font-size:.8rem;margin-left:10px}`;
  document.head.appendChild(style);
  const cap = () => { el.querySelector(".dtc").textContent = DIAL.caption(); };
  cap();
  el.querySelectorAll("button").forEach(b => b.onclick = () => {
    DIAL.set(b.dataset.m);
    el.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x.dataset.m === DIAL.mode));
    cap(); onChange();
  });
}
