"""From MediaPipe's 33 body landmarks to features: joint angles and body-relative positions.

Uses MediaPipe *world* landmarks (metres, centred on the hips) so angles don't depend on how far the
camera is or how the frame is cropped. The browser demo (web/coach.js) computes exactly the same
features — keep the two in sync (tests/test_parity.py checks this).
"""
from __future__ import annotations

import numpy as np

# MediaPipe Pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
KEYPOINTS = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

# (name, a, joint, b): the angle at `joint` between segments joint→a and joint→b
ANGLES = [
    ("l_elbow", L_SHOULDER, L_ELBOW, L_WRIST), ("r_elbow", R_SHOULDER, R_ELBOW, R_WRIST),
    ("l_shoulder", L_HIP, L_SHOULDER, L_ELBOW), ("r_shoulder", R_HIP, R_SHOULDER, R_ELBOW),
    ("l_hip", L_SHOULDER, L_HIP, L_KNEE), ("r_hip", R_SHOULDER, R_HIP, R_KNEE),
    ("l_knee", L_HIP, L_KNEE, L_ANKLE), ("r_knee", R_HIP, R_KNEE, R_ANKLE),
]
ANGLE_NAMES = [a[0] for a in ANGLES]
FEATURE_NAMES = (ANGLE_NAMES + ["torso_lean", "body_line"]
                 + [f"p{i}_{c}" for i in KEYPOINTS for c in ("x", "y", "z")])
N_FEATURES = len(FEATURE_NAMES)  # 8 angles + 2 posture + 13 points × 3 = 49


def angle(a: np.ndarray, joint: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees at `joint` (0-180)."""
    u, v = a - joint, b - joint
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def frame_features(world: np.ndarray) -> np.ndarray:
    """world: (33, 3) world landmarks in metres (MediaPipe: x right, y DOWN, z towards camera).
    Returns a fixed-length feature vector."""
    angles = [angle(world[a], world[j], world[b]) / 180.0 for _, a, j, b in ANGLES]
    mid_sh = (world[L_SHOULDER] + world[R_SHOULDER]) / 2
    mid_hip = (world[L_HIP] + world[R_HIP]) / 2
    mid_ank = (world[L_ANKLE] + world[R_ANKLE]) / 2
    torso = mid_sh - mid_hip
    torso_len = np.linalg.norm(torso) + 1e-9
    # 0 = upright, 1 = horizontal (e.g. push-up plank)
    lean = float(np.degrees(np.arccos(np.clip(-torso[1] / torso_len, -1, 1)))) / 90.0
    body_line = angle(mid_sh, mid_hip, mid_ank) / 180.0  # 1 = straight shoulder-hip-ankle line
    rel = (world[KEYPOINTS] - mid_hip) / torso_len        # body-relative positions, scale-free
    return np.concatenate([angles, [lean, body_line], rel.reshape(-1)]).astype(np.float32)


def sequence_features(worlds: np.ndarray) -> np.ndarray:
    """(T, 33, 3) → (T, N_FEATURES)."""
    return np.stack([frame_features(w) for w in worlds]) if len(worlds) else np.zeros((0, N_FEATURES), np.float32)


def window_stats(window: np.ndarray) -> np.ndarray:
    """(T, F) window → summary vector (mean, std, min, max per feature): input of the small classifier."""
    return np.concatenate([window.mean(0), window.std(0), window.min(0), window.max(0)]).astype(np.float32)
