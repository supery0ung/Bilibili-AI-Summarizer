"""One-off probe: measure transcribe-then-label feasibility & speed.

Run:  python scripts/probe_aligner.py <audio.m4a>
Loads ASR + aligner once, then times:
  1. transcribe WITHOUT timestamps  (warmup)
  2. transcribe WITHOUT timestamps  (steady state)
  3. transcribe WITH timestamps     (alignment overhead)
Writes a UTF-8 report to scripts/probe_aligner_out.txt so Chinese text and
timestamps survive the Windows console encoding.
"""
import os
import sys
import time
from pathlib import Path

MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "BILIBILI_MODEL_CACHE",
        Path.home() / ".cache" / "bilibili_summarizer" / "ai_models",
    )
)
TEMP_ROOT = Path(
    os.environ.get(
        "BILIBILI_TEMP",
        Path.home() / ".cache" / "bilibili_summarizer" / "temp",
    )
)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_ROOT / "huggingface" / "hub"))
os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

import numpy as np
import librosa
import torch
from qwen_asr import Qwen3ASRModel

PROBE_SECONDS = 180
OUT = os.path.join(os.path.dirname(__file__), "probe_aligner_out.txt")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/probe_aligner.py <audio file>")
        return 1
    audio_path = sys.argv[1]
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s.encode("ascii", "replace").decode("ascii"))
        lines.append(s)

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y = y[: PROBE_SECONDS * 16000].astype(np.float32)
    dur = len(y) / 16000
    log(f"audio: {os.path.basename(audio_path)}  clip={dur:.1f}s")

    free0 = torch.cuda.mem_get_info()[0] / 1024**2
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=6,
        max_new_tokens=2048,
    )
    # match production runaway guard
    try:
        model.model.thinker.generation_config.max_time = 240
    except Exception as e:
        log(f"(could not set max_time: {e})")
    free1 = torch.cuda.mem_get_info()[0] / 1024**2
    log(f"VRAM used by ASR+aligner load: {free0 - free1:.0f} MB (free now {free1:.0f} MB)")

    def run(label: str, ts: bool):
        t0 = time.time()
        res = model.transcribe(audio=(y, sr), language=None, return_time_stamps=ts)
        el = time.time() - t0
        r = res[0]
        n_items = len(list(r.time_stamps)) if (ts and r.time_stamps is not None) else 0
        log(f"[{label}] {el:.1f}s  chars={len(r.text)}  lang={r.language!r}  ts_items={n_items}  "
            f"realtime_factor={el/dur:.2f}x")
        return r

    log("--- 1. transcribe NO timestamps (warmup) ---")
    run("warmup", False)
    log("--- 2. transcribe NO timestamps (steady) ---")
    run("steady_no_ts", False)
    log("--- 3. transcribe WITH timestamps ---")
    r = run("with_ts", True)

    items = list(r.time_stamps) if r.time_stamps is not None else []
    log(f"text head: {r.text[:80]}")
    log("first 10 aligned items (text | start | end):")
    for it in items[:10]:
        log(f"   {it.text} | {it.start_time} | {it.end_time}")
    if items:
        log(f"span: {items[0].start_time} .. {items[-1].end_time}  (clip {dur:.0f}s)")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
