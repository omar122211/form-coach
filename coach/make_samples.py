"""Save a few recorded sets as skeleton-only JSON for the demo's "replay a recorded set" option.

Only body-point coordinates are kept (no video, no faces). Source clips: PushUpBench (CC BY 4.0).

    python -m coach.make_samples squat=pushupbench/130 bicep_curl=pushupbench/85 push_up=pushupbench/98
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    out_dir = ROOT / "docs" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    credits = {}
    for arg in sys.argv[1:]:
        name, src = arg.split("=")
        z = np.load(ROOT / "data" / "landmarks" / f"{src}.npz")
        meta = json.loads(str(z["meta"]))
        fps = float(z["fps"])
        frames = z["world"][: int(45 * fps)]  # at most 45 s
        data = {"fps": round(fps, 3), "source": "PushUpBench (CC BY 4.0) — skeleton coordinates only",
                "exercise": meta.get("name"), "true_reps": meta.get("count"), "frames": np.round(frames, 3).tolist()}
        (out_dir / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        credits[name] = {"clip": meta.get("video_path"), "reps": meta.get("count")}
        print(name, len(frames), "frames", meta.get("name"), meta.get("count"))
    (out_dir / "CREDITS.json").write_text(json.dumps(credits, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
