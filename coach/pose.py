"""Run MediaPipe Pose Landmarker over a video and keep the 3D world landmarks.

    python -m coach.pose <video.mp4>     # prints frames found
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"  # Apache-2.0, ~5.5 MB — the same file the web demo uses
TARGET_FPS = 10.0  # enough for exercise movements, 3× less work than 30 fps


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class PoseExtractor:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        self.mp, self.PoseLandmarker = mp, PoseLandmarker
        self.options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ensure_model())),
                                             running_mode=RunningMode.VIDEO, num_poses=1,
                                             min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)

    def video(self, path: Path | str, max_seconds: float = 120.0) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (world landmarks (T, 33, 3), visibility (T, 33), fps). Frames with no person are skipped."""
        # a fresh landmarker per video: VIDEO mode needs timestamps that only ever increase
        landmarker = self.PoseLandmarker.create_from_options(self.options)
        cap = cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / TARGET_FPS))
        worlds, vis, i, last_ts = [], [], 0, -1
        while True:
            ok, frame = cap.read()
            if not ok or i / src_fps > max_seconds:
                break
            if i % step == 0:
                h, w = frame.shape[:2]
                if max(h, w) > 640:  # the model works at 256 px anyway
                    scale = 640 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts = int(i / src_fps * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                res = landmarker.detect_for_video(image, ts)
                if res.pose_world_landmarks:
                    worlds.append([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
                    vis.append([p.visibility for p in res.pose_landmarks[0]])
            i += 1
        cap.release()
        landmarker.close()
        fps = src_fps / step
        return np.asarray(worlds, np.float32).reshape(-1, 33, 3), np.asarray(vis, np.float32).reshape(-1, 33), fps


if __name__ == "__main__":
    ex = PoseExtractor()
    w, v, fps = ex.video(sys.argv[1])
    print(f"{len(w)} frames with a person at {fps:.1f} fps")
