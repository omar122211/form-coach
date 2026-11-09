"""Calibrate the form-rule thresholds on real videos of trainers (assumed good form).

For every rule, the threshold is the 90th (or 10th) percentile of that measurement over the trainers' reps,
rounded to a clean number — so a correct rep triggers each cue only ~10 % of the time.
Calibration uses only the real videos in the classifier's TRAINING split; the expert reps of the
TEST split are used to report the false-alarm rate.

    python -m coach.calibrate_form      → docs/form_thresholds.json + eval/form.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from coach.features import sequence_features
from coach.form import RULES, THRESHOLDS_FILE, check_rep, rep_metrics, split_reps
from coach.reps import count_reps, rep_signal
from coach.train_classifier import CLASSES, load_videos, split

ROOT = Path(__file__).resolve().parent.parent
REP_SETTINGS = {"smooth_s": 0.4, "low": 0.35, "high": 0.65, "detrend_s": 3.0}


def expert_reps(vids: list[dict]) -> dict[str, list[np.ndarray]]:
    out = {c: [] for c in CLASSES}
    for v in vids:
        ex = CLASSES[v["label"]]
        _, ends = count_reps(rep_signal(v["feats"], ex), 10.0, **REP_SETTINGS)
        out[ex] += [r for r in split_reps(v["feats"], ends) if len(r) >= 5]
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = split(load_videos())
    calib = expert_reps([v for v in data["train"] + data["val"] if v["source"] == "real"])
    held_out = expert_reps([v for v in data["test"] if v["source"] == "real"])
    thresholds, report = {}, {}
    for ex, rules in RULES.items():
        values = {m: np.array([rep_metrics(ex, r)[m] for r in calib[ex]]) for m, _, _ in rules}
        thresholds[ex] = {}
        for metric, kind, _ in rules:
            v = values[metric]
            q = np.percentile(v, 90 if kind == "max" else 10)
            step = 0.05 if metric.endswith("ratio") else 5.0
            thresholds[ex][metric] = round(float(np.ceil(q / step) * step if kind == "max" else np.floor(q / step) * step), 2)
        flagged = [bool(check_rep(ex, r, thresholds)) for r in held_out[ex]]
        report[ex] = {"calibration_reps": len(calib[ex]), "held_out_reps": len(held_out[ex]),
                      "held_out_reps_with_a_cue": round(float(np.mean(flagged)), 3) if flagged else None,
                      "thresholds": thresholds[ex]}
        print(ex, report[ex], flush=True)
    THRESHOLDS_FILE.parent.mkdir(exist_ok=True)
    THRESHOLDS_FILE.write_text(json.dumps({"note": "calibrated by coach/calibrate_form.py on real trainer videos",
                                           "thresholds": thresholds}, indent=1), encoding="utf-8")
    (ROOT / "eval" / "form.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
