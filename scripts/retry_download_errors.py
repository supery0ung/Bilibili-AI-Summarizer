"""One-off: single-threaded retry of all status==error videos' downloads.

Diagnostic for distinguishing rate-limit (succeeds on calm retry) from real
auth/cookie failure (still 412). Only touches the downloader (no GPU models).
On success the item is advanced to 'downloaded' so the next run continues it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "BILIBILI_MODEL_CACHE",
        Path.home() / ".cache" / "bilibili_summarizer" / "ai_models",
    )
)
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_CACHE_ROOT))
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_ROOT / "huggingface"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import Pipeline
from core.state import now_iso
from utils import safe_filename


def main() -> None:
    pipeline = Pipeline(PROJECT_ROOT / "config.yaml")
    state = pipeline.state
    dl = pipeline.downloader  # lazy; no GPU

    errors = [
        bvid for bvid, s in state._state["videos"].items()
        if isinstance(s, dict) and s.get("status") == "error"
    ]
    print(f"Found {len(errors)} error videos. Retrying single-threaded...\n", flush=True)

    results = []
    for i, bvid in enumerate(errors, 1):
        vs = state.get_video_state(bvid)
        title = (vs.title or bvid)
        url = f"https://www.bilibili.com/video/{bvid}"
        safe_name = safe_filename(f"{bvid}_{title[:50]}")
        print(f"[{i}/{len(errors)}] {bvid} {title[:40]} ...", flush=True)

        path, err = dl.download_with_retry(url, safe_name)
        if path:
            state.update(bvid, status="downloaded", audio_path=str(path),
                         title=title, up_name=vs.up_name, last_attempt=now_iso())
            print(f"    -> OK downloaded\n", flush=True)
            results.append((bvid, "OK", title))
        else:
            state.update(bvid, status="error", error=f"Download failed ({err})",
                         last_attempt=now_iso())
            print(f"    -> FAIL [{err}]\n", flush=True)
            results.append((bvid, err, title))

    print("\n===== SUMMARY =====", flush=True)
    for bvid, res, title in results:
        print(f"{res:10} {bvid}  {title[:40]}", flush=True)
    ok = sum(1 for _, r, _ in results if r == "OK")
    print(f"\n{ok}/{len(results)} succeeded", flush=True)


if __name__ == "__main__":
    main()
