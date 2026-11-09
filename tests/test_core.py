"""Offline tests for features, rep counting and form rules (no videos or models needed)."""
import numpy as np

from coach.features import ANGLE_NAMES, FEATURE_NAMES, N_FEATURES, angle, frame_features, window_stats
from coach.form import check_rep, split_reps
from coach.reps import count_reps, rep_signal


def standing_pose() -> np.ndarray:
    """A simple upright skeleton in MediaPipe world coordinates (metres, y down)."""
    w = np.zeros((33, 3), np.float32)
    pts = {0: (0, -0.65, 0), 11: (-0.18, -0.45, 0), 12: (0.18, -0.45, 0), 13: (-0.2, -0.2, 0), 14: (0.2, -0.2, 0),
           15: (-0.2, 0.05, 0), 16: (0.2, 0.05, 0), 23: (-0.1, 0, 0), 24: (0.1, 0, 0), 25: (-0.1, 0.45, 0),
           26: (0.1, 0.45, 0), 27: (-0.1, 0.9, 0), 28: (0.1, 0.9, 0)}
    for i, p in pts.items():
        w[i] = p
    return w


def test_angle_basics():
    assert round(angle(np.array([1, 0, 0]), np.zeros(3), np.array([0, 1, 0]))) == 90
    assert round(angle(np.array([1, 0, 0]), np.zeros(3), np.array([-1, 0, 0]))) == 180


def test_features_of_a_standing_person():
    f = frame_features(standing_pose())
    assert f.shape == (N_FEATURES,) == (len(FEATURE_NAMES),)
    knee = f[ANGLE_NAMES.index("l_knee")] * 180
    assert knee > 175, "straight legs ≈ 180°"
    assert f[FEATURE_NAMES.index("torso_lean")] * 90 < 5, "upright torso"


def test_features_ignore_scale_and_position():
    w = standing_pose()
    a, b = frame_features(w), frame_features(w * 1.7 + np.array([0.3, -0.2, 0.5], np.float32))
    assert np.allclose(a, b, atol=1e-4)


def test_count_reps_on_a_clean_signal():
    fps = 10
    t = np.arange(0, 30, 1 / fps)
    signal = np.cos(2 * np.pi * t / 2.5)  # one rep every 2.5 s → 12 reps
    n, ends = count_reps(signal, fps)
    assert n in (11, 12) and len(ends) == n


def test_noise_does_not_create_reps():
    rng = np.random.default_rng(0)
    fps = 10
    t = np.arange(0, 20, 1 / fps)
    noisy = np.cos(2 * np.pi * t / 2.0) + rng.normal(0, 0.15, len(t))
    assert count_reps(noisy, fps)[0] in (9, 10)
    assert count_reps(np.zeros(100), fps)[0] == 0


def test_known_exercise_uses_the_main_joint():
    feats = np.zeros((50, N_FEATURES), np.float32)
    feats[:, ANGLE_NAMES.index("l_knee")] = np.linspace(0, 1, 50)
    feats[:, ANGLE_NAMES.index("r_knee")] = np.linspace(0, 1, 50)
    assert np.allclose(rep_signal(feats, "squat"), np.linspace(0, 1, 50))


def squat_rep(bottom_knee_deg: float, lean_deg: float) -> np.ndarray:
    frames = np.tile(frame_features(standing_pose()), (21, 1))
    knees = np.concatenate([np.linspace(175, bottom_knee_deg, 11), np.linspace(bottom_knee_deg, 175, 11)[1:]])
    frames[:, ANGLE_NAMES.index("l_knee")] = knees / 180
    frames[:, ANGLE_NAMES.index("r_knee")] = knees / 180
    frames[10, FEATURE_NAMES.index("torso_lean")] = lean_deg / 90
    return frames


def test_form_rules():
    t = {"squat": {"knee_min": 100.0, "lean_at_bottom": 45.0, "knee_width_ratio": 0.2}}
    assert check_rep("squat", squat_rep(85, 30), t) == []
    cues = check_rep("squat", squat_rep(130, 65), t)
    assert any("deeper" in c for c in cues) and any("chest" in c for c in cues)


def test_split_reps_and_window_stats():
    feats = np.arange(40, dtype=np.float32).reshape(20, 2)
    reps = split_reps(feats, [5, 12, 19])
    assert [len(r) for r in reps] == [6, 8, 8]
    assert window_stats(feats).shape == (8,)
