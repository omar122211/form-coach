"""Evaluate rep counting.

1. PushUpBench (227 real clips, many exercise types, human counts). Settings are tuned on a fixed 30 % "dev"
   slice and reported on the other 70 % — never tuned on the reported clips.
2. InfiniteRep synthetic videos (squat / push-up / curl / press with exact per-frame rep progress), using the
   exercise-specific signal the demo uses.

Metrics (as in the PushUpBench paper): exact match, off-by-one (OBO) accuracy, mean absolute error (MAE).

    python -m eval.reps_eval
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from coach.features import sequence_features
from coach.reps import count_reps, rep_signal, smooth

ROOT = Path(__file__).resolve().parent.parent
LM = ROOT / "data" / "landmarks"


def is_dev(clip_id: str) -> bool:
    return int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 10 < 3


def periodic_signal(feats: np.ndarray, fps: float, k: int = 4) -> np.ndarray:
    """Among the top principal components, pick the one that repeats most regularly (autocorrelation peak)."""
    x = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    if len(x) < 3:
        return np.zeros(len(x))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    best, best_score = x @ vt[0], -1.0
    lo, hi = int(0.7 * fps), int(min(8 * fps, len(x) // 2))
    for comp in vt[:k]:
        s = smooth(x @ comp, fps, 0.3)
        s = s - s.mean()
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        ac = ac / (ac[0] + 1e-9)
        if hi > lo:
            score = float(ac[lo:hi].max())
            if score > best_score:
                best, best_score = x @ comp, score
    return best


def metrics(pred: list[int], truth: list[list[int]]) -> dict:
    n = len(pred)
    exact = sum(p in t for p, t in zip(pred, truth))
    obo = sum(min(abs(p - c) for c in t) <= 1 for p, t in zip(pred, truth))
    mae = float(np.mean([abs(p - t[0]) for p, t in zip(pred, truth)]))
    return {"n": n, "exact": round(exact / n, 4), "obo": round(obo / n, 4), "mae": round(mae, 2)}


def load(dataset: str) -> list[dict]:
    clips = []
    for f in sorted((LM / dataset).glob("*.npz")):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        clips.append({"id": f.stem, "meta": meta, "fps": float(z["fps"]), "feats": sequence_features(z["world"])})
    return clips


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pub = load("pushupbench")
    dev = [c for c in pub if is_dev(c["id"])]
    test = [c for c in pub if not is_dev(c["id"])]
    print(f"PushUpBench: {len(dev)} dev / {len(test)} test clips", flush=True)

    grid = list(itertools.product(["pca", "periodic"], [0.2, 0.4], [(0.3, 0.7), (0.35, 0.65), (0.25, 0.75)], [None, 1.5, 3.0, 5.0]))
    signals = {}

    def predict(clips, method, sm, th, dt):
        out = []
        for c in clips:
            key = (c["id"], method)
            if key not in signals:
                signals[key] = rep_signal(c["feats"]) if method == "pca" else periodic_signal(c["feats"], c["fps"])
            out.append(count_reps(signals[key], c["fps"], smooth_s=sm, low=th[0], high=th[1], detrend_s=dt)[0])
        return out

    truth_dev = [c["meta"]["count"] for c in dev]
    scored = []
    for method, sm, th, dt in grid:
        m = metrics(predict(dev, method, sm, th, dt), truth_dev)
        scored.append(((m["obo"], m["exact"], -m["mae"]), (method, sm, th, dt), m))
    scored.sort(key=lambda s: s[0], reverse=True)
    _, (method, sm, th, dt), dev_m = scored[0]
    print("best on dev:", method, sm, th, dt, dev_m, flush=True)
    no_dt = max((s for s in scored if s[1][3] is None), key=lambda s: s[0])
    print("best on dev without detrending:", no_dt[1], no_dt[2], flush=True)

    truth_test = [c["meta"]["count"] for c in test]
    pred_test = predict(test, method, sm, th, dt)
    median = int(np.median([t[0] for t in truth_dev]))
    results = {
        "settings": {"signal": method, "smooth_s": sm, "thresholds": th, "detrend_s": dt},
        "pushupbench_test_without_detrend": metrics(predict(test, *no_dt[1][:3], None), truth_test),
        "pushupbench_dev": dev_m,
        "pushupbench_test": metrics(pred_test, truth_test),
        "pushupbench_test_baseline_median": metrics([median] * len(test), truth_test) | {"guess": median},
        "pushupbench_test_simple": metrics([p for p, c in zip(pred_test, test) if not c["meta"]["fuzzy_action"] and not c["meta"]["complex_action"]],
                                           [t for t, c in zip(truth_test, test) if not c["meta"]["fuzzy_action"] and not c["meta"]["complex_action"]]),
        "pushupbench_test_rows": [{"id": c["id"], "name": c["meta"]["name"], "truth": t, "pred": p} for c, t, p in zip(test, truth_test, pred_test)],
    }
    print("test:", results["pushupbench_test"], "| median-guess baseline:", results["pushupbench_test_baseline_median"], flush=True)

    # synthetic videos of the 4 demo exercises, with the exercise-specific signal
    syn = [c for c in load("exercises") if c["meta"].get("reps") is not None]
    for label, d in (("synthetic_known_exercise", dt), ("synthetic_known_exercise_no_detrend", None)):
        per, allp, allt = {}, [], []
        for ex in sorted({c["meta"]["label"] for c in syn}):
            clips = [c for c in syn if c["meta"]["label"] == ex]
            pred = [count_reps(rep_signal(c["feats"], ex), c["fps"], smooth_s=sm, low=th[0], high=th[1], detrend_s=d)[0] for c in clips]
            truth = [[c["meta"]["reps"]] for c in clips]
            per[ex] = metrics(pred, truth)
            allp += pred; allt += truth
        if allp:
            per["all"] = metrics(allp, allt)
        results[label] = per
        print(label, per, flush=True)
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "reps.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
