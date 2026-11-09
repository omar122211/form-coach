"""Form feedback per rep, from joint angles at the bottom / top of the movement.

Each rule measures one thing per rep (e.g. the smallest knee angle of a squat) and compares it with a
threshold. Thresholds are CALIBRATED on the real videos of trainers in the training data (assumed good form):
each is set so that only ~10 % of those reps would trigger the cue (see coach/calibrate_form.py).
That matters because single-camera 3D pose under-reads deep joint flexion — a full curl reads ~85°, not 40°.

These are coaching heuristics, not a medical tool. docs/coach.js applies the same rules and thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from coach.features import FEATURE_NAMES

A = {n: i for i, n in enumerate(FEATURE_NAMES)}
THRESHOLDS_FILE = Path(__file__).resolve().parent.parent / "docs" / "form_thresholds.json"

# exercise → [(metric, "max" = too high is bad | "min" = too low is bad, cue)]
RULES = {
    "squat": [("knee_min", "max", "Go deeper — aim for thighs parallel to the floor"),
              ("lean_at_bottom", "max", "Keep your chest up"),
              ("knee_width_ratio", "min", "Push your knees out over your toes")],
    "push_up": [("elbow_min", "max", "Go lower — chest towards the floor"),
                ("elbow_max", "min", "Straighten your arms at the top"),
                ("body_line_min", "min", "Keep your body in a straight line (hips sagging or piking)")],
    "bicep_curl": [("elbow_min", "max", "Curl all the way up"),
                   ("elbow_max", "min", "Lower the weight fully"),
                   ("shoulder_range", "max", "Keep your elbows by your sides")],
    "shoulder_press": [("elbow_max", "min", "Press all the way up to lockout"),
                       ("lean_max", "max", "Don't lean back — brace your core")],
}


def _deg(rep: np.ndarray, *names: str) -> np.ndarray:
    return rep[:, [A[n] for n in names]].mean(1) * 180.0


def rep_metrics(exercise: str, rep: np.ndarray) -> dict[str, float]:
    """The measurements the rules look at, in degrees (or a ratio)."""
    if exercise == "squat":
        knee = _deg(rep, "l_knee", "r_knee")
        b = int(knee.argmin())
        knee_w = abs(rep[b, A["p25_x"]] - rep[b, A["p26_x"]])
        ankle_w = abs(rep[b, A["p27_x"]] - rep[b, A["p28_x"]]) + 1e-6
        return {"knee_min": float(knee.min()), "lean_at_bottom": float(rep[b, A["torso_lean"]] * 90),
                "knee_width_ratio": float(knee_w / ankle_w)}
    if exercise in ("push_up", "bicep_curl", "shoulder_press"):
        elbow = _deg(rep, "l_elbow", "r_elbow")
        m = {"elbow_min": float(elbow.min()), "elbow_max": float(elbow.max())}
        if exercise == "push_up":
            m["body_line_min"] = float((rep[:, A["body_line"]] * 180).min())
        if exercise == "bicep_curl":
            sh = _deg(rep, "l_shoulder", "r_shoulder")
            m["shoulder_range"] = float(sh.max() - sh.min())
        if exercise == "shoulder_press":
            m["lean_max"] = float((rep[:, A["torso_lean"]] * 90).max())
        return m
    return {}


def load_thresholds() -> dict:
    return json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))["thresholds"] if THRESHOLDS_FILE.exists() else {}


_THRESHOLDS = load_thresholds()


def check_rep(exercise: str, rep: np.ndarray, thresholds: dict | None = None) -> list[str]:
    """rep: (T, N_FEATURES) frames of one repetition → list of cues (empty = good rep)."""
    t = (thresholds if thresholds is not None else _THRESHOLDS).get(exercise, {})
    m = rep_metrics(exercise, rep)
    cues = []
    for metric, kind, cue in RULES.get(exercise, []):
        if metric in t and metric in m and ((kind == "max" and m[metric] > t[metric]) or (kind == "min" and m[metric] < t[metric])):
            cues.append(cue)
    return cues


def split_reps(features: np.ndarray, ends: list[int]) -> list[np.ndarray]:
    starts = [0] + ends[:-1]
    return [features[s:e + 1] for s, e in zip(starts, ends) if e > s]
