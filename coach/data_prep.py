"""Download the two datasets and unpack only what's needed.

- PushUpBench (CC BY 4.0): 227 exercise clips with human rep counts — the rep-counting benchmark.
- Real-Time Exercise Recognition Dataset: real videos (25 per exercise) and synthetic InfiniteRep videos
  (100 per exercise, with per-frame rep progress) for squat, push-up, biceps curl and shoulder press.

    python -m coach.data_prep          # ~4 GB download, then ~5 GB of videos in data/
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = {"barbell biceps curl": "bicep_curl", "push-up": "push_up", "shoulder press": "shoulder_press", "squat": "squat"}


def main() -> None:
    snapshot_download("anonymousatom/pushupbench", repo_type="dataset", local_dir=DATA / "pushupbench")
    root = DATA / "exercise-recognition"
    snapshot_download("RickyRiccio/Real_Time_Exercise_Recognition_Dataset", repo_type="dataset", local_dir=root)
    out = root / "videos"
    with zipfile.ZipFile(root / "Real-Time Exercise Recognition Dataset.zip") as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            ext = info.filename.lower().rsplit(".", 1)[-1]
            if info.filename.startswith("final_kaggle_with_additional_video/") and len(parts) == 3 and parts[1] in LABELS and ext in ("mp4", "mov"):
                dest = out / LABELS[parts[1]] / ("real_" + parts[-1].replace(" ", "_"))
            elif info.filename.startswith("synthetic_dataset/") and len(parts) == 4 and parts[2] in LABELS and ext in ("mp4", "json"):
                dest = out / LABELS[parts[2]] / ("syn_" + parts[-1])  # skip the 600k per-frame label images
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print("videos ready in", out)


if __name__ == "__main__":
    main()
