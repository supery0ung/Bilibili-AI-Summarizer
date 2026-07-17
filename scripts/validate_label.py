"""End-to-end validation of the transcribe-then-label multi-speaker path.

Run:  python scripts/validate_label.py <audio.m4a> [seconds]
Builds the Qwen3ASRClient from config.yaml, slices the first N seconds (default
300) to a temp wav, runs _diarize_and_label, times it, and writes the resulting
speaker-labelled markdown to scripts/validate_label_out.md (UTF-8).
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
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clients.qwen_asr_client import Qwen3ASRClient

OUT = Path(__file__).resolve().parent / "validate_label_out.md"


def main() -> int:
    audio = sys.argv[1]
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    qc = cfg.get("qwen3", {})
    hf_token = qc.get("hf_token") or cfg.get("whisperx", {}).get("hf_token")

    # Slice first N seconds to a temp wav
    y, sr = librosa.load(audio, sr=16000, mono=True)
    y = y[: secs * 16000].astype(np.float32)
    clip = TEMP_ROOT / "validate_label_clip.wav"
    sf.write(str(clip), y, sr)
    dur = len(y) / sr
    print(f"clip: {dur:.0f}s -> {clip}")

    client = Qwen3ASRClient(
        model_name=qc.get("model", "Qwen/Qwen3-ASR-1.7B"),
        device=qc.get("device", "cuda:0"),
        language=qc.get("language"),
        diarize=True,
        hf_token=hf_token,
        min_speakers=qc.get("min_speakers", 1),
        max_speakers=qc.get("max_speakers", 5),
    )

    t0 = time.time()
    turns = client._diarize_and_label(clip)
    elapsed = time.time() - t0
    md = client._format_turns_to_markdown(turns, title="VALIDATION", author="test")
    OUT.write_text(md, encoding="utf-8")

    n_labels = md.count("说话人")
    speakers = sorted({t.speaker for t in turns})
    print(f"elapsed: {elapsed:.1f}s ({elapsed/dur:.2f}x realtime)")
    print(f"segments: {len(turns)}, raw speakers: {speakers}")
    print(f"说话人 labels in markdown: {n_labels}")
    print(f"markdown chars: {len(md)}; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
