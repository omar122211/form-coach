// node tests/replay_check.mjs → runs the demo pipeline on each docs/samples/*.json skeleton (no browser needed)
import { readFileSync, readdirSync } from "node:fs";
import { RepCounter, checkRep, classify, frameFeatures, primarySignal } from "../docs/coach.js";
const model = JSON.parse(readFileSync(new URL("../docs/model.json", import.meta.url)));
const thresholds = JSON.parse(readFileSync(new URL("../docs/form_thresholds.json", import.meta.url))).thresholds;
for (const file of readdirSync(new URL("../docs/samples/", import.meta.url)).filter((f) => f.endsWith(".json") && f !== "CREDITS.json")) {
  const s = JSON.parse(readFileSync(new URL(`../docs/samples/${file}`, import.meta.url)));
  const frames = s.frames.map(frameFeatures);
  let probs = null, exercise = null; const votes = {};
  for (let i = model.window; i <= frames.length; i += 5) {
    const p = classify(model, frames.slice(i - model.window, i));
    probs = probs ? probs.map((v, j) => 0.7 * v + 0.3 * p[j]) : p;
    const b = probs.indexOf(Math.max(...probs));
    if (!exercise || probs[b] > 0.75) exercise = model.classes[b];
    votes[model.classes[b]] = (votes[model.classes[b]] || 0) + 1;
  }
  const c = new RepCounter(); let start = 0, cues = [];
  frames.forEach((f, i) => { if (c.push(primarySignal(exercise, f), i / s.fps)) { cues.push(checkRep(exercise, frames.slice(start, i + 1), thresholds)); start = i; } });
  console.log(file, "| true:", s.exercise, s.true_reps, "| detected:", exercise, votes, "| reps:", c.count, "| cues:", JSON.stringify(cues.slice(0, 3)));
}
