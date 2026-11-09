// Form Coach core — the same features, classifier, rep counter and form rules as the Python package
// (coach/features.py, coach/reps.py, coach/form.py). tests/test_parity.py checks both give the same numbers.

export const ANGLES = [
  ["l_elbow", 11, 13, 15], ["r_elbow", 12, 14, 16],
  ["l_shoulder", 23, 11, 13], ["r_shoulder", 24, 12, 14],
  ["l_hip", 11, 23, 25], ["r_hip", 12, 24, 26],
  ["l_knee", 23, 25, 27], ["r_knee", 24, 26, 28],
];
export const KEYPOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export const N_FEATURES = ANGLES.length + 2 + KEYPOINTS.length * 3;
const IDX = Object.fromEntries(ANGLES.map((a, i) => [a[0], i]));
const TORSO_LEAN = 8, BODY_LINE = 9;
const P = (lm, c) => 10 + KEYPOINTS.indexOf(lm) * 3 + "xyz".indexOf(c); // feature index of a keypoint coordinate

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.sqrt(dot(a, a));
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function angleDeg(a, j, b) {
  const u = sub(a, j), v = sub(b, j);
  return (Math.acos(clip(dot(u, v) / (norm(u) * norm(v) + 1e-9), -1, 1)) * 180) / Math.PI;
}

/** world: 33 × [x, y, z] (MediaPipe world landmarks, metres, y down) → feature array */
export function frameFeatures(world) {
  const f = ANGLES.map(([, a, j, b]) => angleDeg(world[a], world[j], world[b]) / 180);
  const sh = mid(world[11], world[12]), hip = mid(world[23], world[24]), ank = mid(world[27], world[28]);
  const torso = sub(sh, hip), len = norm(torso) + 1e-9;
  f.push((Math.acos(clip(-torso[1] / len, -1, 1)) * 180) / Math.PI / 90);
  f.push(angleDeg(sh, hip, ank) / 180);
  for (const k of KEYPOINTS) for (let c = 0; c < 3; c++) f.push((world[k][c] - hip[c]) / len);
  return f;
}

export function windowStats(frames) {
  const n = frames.length, d = frames[0].length;
  const mean = new Array(d).fill(0), sq = new Array(d).fill(0), mn = [...frames[0]], mx = [...frames[0]];
  for (const fr of frames) for (let i = 0; i < d; i++) {
    mean[i] += fr[i] / n; sq[i] += (fr[i] * fr[i]) / n;
    if (fr[i] < mn[i]) mn[i] = fr[i];
    if (fr[i] > mx[i]) mx[i] = fr[i];
  }
  const std = mean.map((m, i) => Math.sqrt(Math.max(0, sq[i] - m * m)));
  return [...mean, ...std, ...mn, ...mx];
}

/** The small MLP exported by coach/train_classifier.py: stats → 64 ReLU → classes (softmax). */
export function classify(model, frames) {
  const x = windowStats(frames).map((v, i) => (v - model.mean[i]) / model.std[i]);
  const h = model.w1.map((row, j) => Math.max(0, row.reduce((s, w, i) => s + w * x[i], model.b1[j])));
  const z = model.w2.map((row, j) => row.reduce((s, w, i) => s + w * h[i], model.b2[j]));
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / t);
}

export const PRIMARY = { squat: ["l_knee", "r_knee"], push_up: ["l_elbow", "r_elbow"], bicep_curl: ["l_elbow", "r_elbow"], shoulder_press: ["l_elbow", "r_elbow"] };
export const primarySignal = (exercise, f) => PRIMARY[exercise].reduce((s, n) => s + f[IDX[n]], 0) / PRIMARY[exercise].length;

/** Live rep counter: moving-average smoothing, detrending (minus the last few seconds' average), then two
 *  adaptive thresholds over the recent range (hysteresis). Settings = the ones chosen in eval/reps_eval.py. */
export class RepCounter {
  // Windows are in SECONDS (timestamps), so a slow device that delivers fewer frames still behaves the same.
  constructor({ smoothS = 0.4, low = 0.35, high = 0.65, detrendS = 3, windowS = 10, minRange = 25 / 180 } = {}) {
    Object.assign(this, { smoothS, detrendS, windowS, low, high, minRange });
    this.raw = []; this.trend = []; this.hist = []; this.state = null; this.count = 0; this.n = 0;
  }
  push(v, t = this.n++ / 10) {
    const keep = (arr, span) => { while (arr.length && t - arr[0][0] >= span - 1e-6) arr.shift(); };
    const mean = (arr) => arr.reduce((a, [, x]) => a + x, 0) / arr.length;
    this.raw.push([t, v]); keep(this.raw, this.smoothS);
    const sm = mean(this.raw);
    this.trend.push([t, sm]); keep(this.trend, this.detrendS);
    const s = sm - mean(this.trend);
    this.hist.push([t, s]); keep(this.hist, this.windowS);
    const sorted = this.hist.map(([, x]) => x).sort((a, b) => a - b);
    const lo = sorted[Math.floor(0.05 * (sorted.length - 1))], hi = sorted[Math.ceil(0.95 * (sorted.length - 1))];
    if (hi - lo < this.minRange) return false; // not moving enough to be an exercise
    let done = false;
    if (s <= lo + this.low * (hi - lo)) this.state = "low";
    else if (s >= lo + this.high * (hi - lo)) {
      if (this.state === "low") { this.count++; done = true; }
      this.state = "high";
    }
    return done;
  }
}

const deg = (rep, ...names) => rep.map((f) => names.reduce((s, n) => s + f[IDX[n]], 0) / names.length * 180);
const argmin = (a) => a.indexOf(Math.min(...a));

// Same rules as coach/form.py; thresholds come from docs/form_thresholds.json (calibrated on trainer videos).
export const RULES = {
  squat: [["knee_min", "max", "Go deeper — aim for thighs parallel to the floor"], ["lean_at_bottom", "max", "Keep your chest up"],
          ["knee_width_ratio", "min", "Push your knees out over your toes"]],
  push_up: [["elbow_min", "max", "Go lower — chest towards the floor"], ["elbow_max", "min", "Straighten your arms at the top"],
            ["body_line_min", "min", "Keep your body in a straight line (hips sagging or piking)"]],
  bicep_curl: [["elbow_min", "max", "Curl all the way up"], ["elbow_max", "min", "Lower the weight fully"],
               ["shoulder_range", "max", "Keep your elbows by your sides"]],
  shoulder_press: [["elbow_max", "min", "Press all the way up to lockout"], ["lean_max", "max", "Don't lean back — brace your core"]],
};

export function repMetrics(exercise, rep) {
  if (exercise === "squat") {
    const knee = deg(rep, "l_knee", "r_knee"), b = argmin(knee);
    const kneeW = Math.abs(rep[b][P(25, "x")] - rep[b][P(26, "x")]), ankleW = Math.abs(rep[b][P(27, "x")] - rep[b][P(28, "x")]) + 1e-6;
    return { knee_min: Math.min(...knee), lean_at_bottom: rep[b][TORSO_LEAN] * 90, knee_width_ratio: kneeW / ankleW };
  }
  if (!PRIMARY[exercise]) return {};
  const elbow = deg(rep, "l_elbow", "r_elbow");
  const m = { elbow_min: Math.min(...elbow), elbow_max: Math.max(...elbow) };
  if (exercise === "push_up") m.body_line_min = Math.min(...rep.map((f) => f[BODY_LINE] * 180));
  if (exercise === "bicep_curl") { const sh = deg(rep, "l_shoulder", "r_shoulder"); m.shoulder_range = Math.max(...sh) - Math.min(...sh); }
  if (exercise === "shoulder_press") m.lean_max = Math.max(...rep.map((f) => f[TORSO_LEAN] * 90));
  return m;
}

/** rep: frames (feature arrays) of one repetition → list of cues (empty = good rep). */
export function checkRep(exercise, rep, thresholds = {}) {
  const t = thresholds[exercise] || {}, m = repMetrics(exercise, rep);
  return (RULES[exercise] || []).filter(([k, kind]) => k in t && k in m && (kind === "max" ? m[k] > t[k] : m[k] < t[k])).map((r) => r[2]);
}
