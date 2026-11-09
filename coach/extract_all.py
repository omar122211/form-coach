"""Extract pose landmarks from every video once (slow part), so training and evaluation are fast.

    python -m coach.extract_all pushupbench
    python -m coach.extract_all exercises

Writes data/landmarks/<dataset>/<id>.npz with world landmarks, visibility, fps and labels.
Runs one video per CPU core.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_extractor = None


def _init():
    global _extractor
    from coach.pose import PoseExtractor
    _extractor = PoseExtractor()


def _work(job: dict) -> tuple[str, int | str]:
    out = Path(job["out"])
    if out.exists():
        return job["id"], "cached"
    try:
        world, vis, fps = _extractor.video(job["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, world=world, vis=vis, fps=fps, meta=json.dumps(job["meta"]))
        return job["id"], len(world)
    except Exception as e:  # a broken video shouldn't stop the batch
        return job["id"], f"error: {e}"


def pushupbench_jobs() -> list[dict]:
    root = DATA / "pushupbench"
    rows = [json.loads(l) for l in (root / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for r in rows:
        path = root / r["video_path"]
        if path.exists():
            jobs.append({"id": str(r["id"]), "path": str(path), "out": str(DATA / "landmarks" / "pushupbench" / f"{r['id']}.npz"),
                         "meta": {k: r[k] for k in ("id", "name", "count", "fuzzy_action", "complex_action", "video_path")}})
    return jobs


def exercise_jobs() -> list[dict]:
    root = DATA / "exercise-recognition" / "videos"
    jobs = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT):
        rel = path.relative_to(root)
        label = rel.parts[0]
        vid = "__".join(rel.with_suffix("").parts)
        meta = {"label": label, "file": str(rel).replace("\\", "/"), "source": "synthetic" if path.name.startswith("syn_") else "real"}
        ann = path.with_suffix(".json")
        if ann.exists():  # synthetic videos record rep progress per frame → exact rep count
            frames = json.loads(ann.read_text(encoding="utf-8"))["images"]
            meta["reps"] = int(round(max(f.get("rep_count", 0) for f in frames)))
        jobs.append({"id": vid, "path": str(path), "out": str(DATA / "landmarks" / "exercises" / f"{vid}.npz"), "meta": meta})
    return jobs


def main() -> None:
    which = sys.argv[1]
    jobs = {"pushupbench": pushupbench_jobs, "exercises": exercise_jobs}[which]()
    workers = max(1, (os.cpu_count() or 4) // 2)
    print(f"{len(jobs)} videos, {workers} workers", flush=True)
    started, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for fut in as_completed([pool.submit(_work, j) for j in jobs]):
            vid, result = fut.result()
            done += 1
            if isinstance(result, str) and result.startswith("error") or done % 25 == 0:
                print(f"{done}/{len(jobs)} {vid}: {result} ({(time.time() - started) / 60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
