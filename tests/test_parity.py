"""The browser demo (docs/coach.js) must compute exactly what the Python code computes.

Runs coach.js in Node on the same random poses and compares features, classifier output and form cues.
Skipped if Node isn't installed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from coach.features import frame_features, window_stats
from coach.form import check_rep, load_thresholds

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def run_js(script: str) -> object:
    # written to a temp module next to the tests (Windows limits command-line length)
    path = ROOT / "tests" / "_parity_tmp.mjs"
    path.write_text(script.replace('"./docs/coach.js"', '"../docs/coach.js"'), encoding="utf-8")
    try:
        out = subprocess.run([NODE, str(path)], capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=60)
    finally:
        path.unlink(missing_ok=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def random_poses(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.4, (n, 33, 3)).astype(np.float32)


def test_features_match():
    poses = random_poses(5)
    js = run_js(f"""
import {{ frameFeatures }} from "./docs/coach.js";
const poses = {json.dumps(poses.tolist())};
console.log(JSON.stringify(poses.map(frameFeatures)));
""")
    py = np.stack([frame_features(p) for p in poses])
    assert np.allclose(py, np.array(js), atol=1e-4)


def test_classifier_and_stats_match():
    poses = random_poses(20, seed=2)
    feats = np.stack([frame_features(p) for p in poses])
    d = len(window_stats(feats))
    rng = np.random.default_rng(3)
    model = {"mean": [0.0] * d, "std": [1.0] * d, "w1": rng.normal(0, 0.1, (8, d)).tolist(), "b1": [0.1] * 8,
             "w2": rng.normal(0, 0.1, (4, 8)).tolist(), "b2": [0.0] * 4}
    js = run_js(f"""
import {{ classify, windowStats }} from "./docs/coach.js";
const frames = {json.dumps(feats.tolist())}, model = {json.dumps(model)};
console.log(JSON.stringify({{ stats: windowStats(frames), probs: classify(model, frames) }}));
""")
    stats = window_stats(feats)
    h = np.maximum(0, np.array(model["w1"]) @ stats + np.array(model["b1"]))
    z = np.array(model["w2"]) @ h
    probs = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    assert np.allclose(stats, js["stats"], atol=1e-4)
    assert np.allclose(probs, js["probs"], atol=1e-5)


@pytest.mark.parametrize("exercise", ["squat", "push_up", "bicep_curl", "shoulder_press"])
def test_form_cues_match(exercise):
    feats = np.stack([frame_features(p) for p in random_poses(15, seed=4)])
    t = load_thresholds()
    js = run_js(f"""
import {{ checkRep, repMetrics }} from "./docs/coach.js";
const rep = {json.dumps(feats.tolist())};
console.log(JSON.stringify({{ cues: checkRep("{exercise}", rep, {json.dumps(t)}), metrics: repMetrics("{exercise}", rep) }}));
""")
    from coach.form import rep_metrics
    assert check_rep(exercise, feats, t) == js["cues"]
    py_m = rep_metrics(exercise, feats)
    assert py_m.keys() == js["metrics"].keys() and all(abs(py_m[k] - js["metrics"][k]) < 1e-3 for k in py_m)
