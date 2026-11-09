"""Rep counting from a pose sequence.

1. Turn the pose sequence into ONE signal that goes up and down once per rep:
   - known exercise → its main joint angle (knees for squats, elbows for push-ups, curls and presses);
   - unknown exercise → the first principal component of all pose features (the direction of most movement).
2. Smooth it, then count full swings with two adaptive thresholds (hysteresis), so jitter around a
   single threshold can't create extra reps.

The browser demo (web/coach.js) runs the same algorithm frame by frame.
"""
from __future__ import annotations

import numpy as np

from coach.features import ANGLE_NAMES

PRIMARY = {  # exercise → angles whose mean is the rep signal
    "squat": ["l_knee", "r_knee"],
    "push_up": ["l_elbow", "r_elbow"],
    "bicep_curl": ["l_elbow", "r_elbow"],
    "shoulder_press": ["l_elbow", "r_elbow"],
}


def smooth(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    k = max(1, int(round(fps * seconds)))
    if k <= 1 or len(x) < k:
        return x.astype(float)
    pad = np.pad(x.astype(float), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def rep_signal(features: np.ndarray, exercise: str | None = None) -> np.ndarray:
    if exercise in PRIMARY:
        idx = [ANGLE_NAMES.index(n) for n in PRIMARY[exercise]]
        return features[:, idx].mean(1)
    x = features - features.mean(0)
    x = x / (x.std(0) + 1e-6)
    if len(x) < 3:
        return x[:, 0] if x.size else np.zeros(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)  # PCA: first principal direction
    return x @ vt[0]


def count_reps(signal: np.ndarray, fps: float, smooth_s: float = 0.4, low: float = 0.3, high: float = 0.7,
               min_range: float = 1e-3, detrend_s: float | None = None) -> tuple[int, list[int]]:
    """Returns (number of reps, frame index where each rep finished).

    detrend_s: subtract a slow moving average first, so drift (the person shifting position) can't hide fast reps.
    """
    if len(signal) < 3:
        return 0, []
    s = smooth(signal, fps, smooth_s)
    if detrend_s:
        s = s - smooth(s, fps, detrend_s)
    lo, hi = np.percentile(s, 5), np.percentile(s, 95)  # robust to a few outlier frames
    if hi - lo < min_range:
        return 0, []
    t_low, t_high = lo + low * (hi - lo), lo + high * (hi - lo)
    state, ends = None, []
    for i, v in enumerate(s):
        if v <= t_low:
            state = "low"
        elif v >= t_high:
            if state == "low":
                ends.append(i)  # a full low → high swing = one rep
            state = "high"
    return len(ends), ends
