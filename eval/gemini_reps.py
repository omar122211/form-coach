"""Ask Gemini to count reps in the same PushUpBench test clips (video in, JSON count out), for comparison.

    python -m eval.gemini_reps --n 60     # free tier: one clip every few seconds

Writes eval/gemini_reps.json. Uses only clips from the reported (non-dev) split.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from eval.reps_eval import is_dev, metrics

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    gemini_api_key: str
    model: str = "gemini-3.5-flash-lite"


class Count(BaseModel):
    count: int = Field(ge=0, description="Number of complete repetitions")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--pause", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=None, help="video frames per second sent to Gemini (API default: 1)")
    args = ap.parse_args()
    from google import genai
    from google.genai import errors, types

    s = Settings()
    client = genai.Client(api_key=s.gemini_api_key, http_options=types.HttpOptions(timeout=180_000))
    rows = [json.loads(l) for l in (ROOT / "data" / "pushupbench" / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not is_dev(str(r["id"]))]
    random.Random(5).shuffle(rows)
    out = []
    for r in rows[: args.n]:
        video = (ROOT / "data" / "pushupbench" / r["video_path"]).read_bytes()
        prompt = (f"Count how many complete repetitions of the exercise \"{r['name']}\" are performed in this video. "
                  "Count every repetition from start to end.")
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=s.model, contents=[types.Part(inline_data=types.Blob(data=video, mime_type="video/mp4"),
                                                    video_metadata=types.VideoMetadata(fps=args.fps) if args.fps else None), prompt],
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=Count))
            pred = resp.parsed.count if isinstance(resp.parsed, Count) else Count.model_validate_json(resp.text).count
        except (errors.APIError, ValueError) as e:
            print("error", r["id"], e, flush=True)
            pred = None
        out.append({"id": r["id"], "name": r["name"], "truth": r["count"], "pred": pred, "s": round(time.perf_counter() - t0, 1)})
        print(f"{len(out)}/{args.n} {r['name']}: gemini {pred} vs {r['count']}", flush=True)
        time.sleep(args.pause)
    ok = [o for o in out if o["pred"] is not None]
    result = {"model": s.model, "video_fps": args.fps or 1, "metrics": metrics([o["pred"] for o in ok], [o["truth"] for o in ok]),
              "errors": len(out) - len(ok), "median_seconds": sorted(o["s"] for o in ok)[len(ok) // 2] if ok else None, "rows": out}
    name = "gemini_reps.json" if not args.fps else f"gemini_reps_{args.fps:g}fps.json"
    (ROOT / "eval" / name).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(result["metrics"], flush=True)


if __name__ == "__main__":
    main()
