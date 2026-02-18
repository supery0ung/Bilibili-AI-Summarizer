"""ASR Benchmark Tool - Compare different transcription configurations.

Downloads a test video and transcribes it with multiple parameter combos,
then generates a comparison report.

Usage:
    python benchmark_asr.py [--audio PATH]   # use existing audio file
    python benchmark_asr.py                  # auto-download BV1L9cPzdEfc
"""

from __future__ import annotations

import gc
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TEST_VIDEO_URL = "https://www.bilibili.com/video/BV1L9cPzdEfc"
TEST_VIDEO_BVID = "BV1L9cPzdEfc"
BENCHMARK_DIR = Path("E:/bilibili_summarizer_v3/output/benchmark")

# Load project config for credentials
with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    config_name: str
    engine: str
    char_count: int
    line_count: int
    paragraph_count: int
    duration_seconds: float
    output_file: str
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Test configurations
# ──────────────────────────────────────────────

# WhisperX initial prompt for Chinese
INITIAL_PROMPT = (
    "以下是普通话的句子，使用简体中文输出，包含正确的标点符号。"
    "这是一段视频的语音转录，请注意使用逗号、句号、问号和感叹号。"
)

WHISPERX_CONFIGS = {
    "1_default": {
        "desc": "WhisperX 默认参数 (beam=5, VAD default)",
        "asr_options": {
            "initial_prompt": INITIAL_PROMPT,
            "beam_size": 5,
        },
        "vad_options": {},  # use whisperx defaults
    },
    "2_sensitive_vad": {
        "desc": "WhisperX 灵敏 VAD (onset=0.3, offset=0.3)",
        "asr_options": {
            "initial_prompt": INITIAL_PROMPT,
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
        },
        "vad_options": {
            "vad_onset": 0.3,
            "vad_offset": 0.3,
        },
    },
    "3_ultra_sensitive_vad": {
        "desc": "WhisperX 超灵敏 VAD (onset=0.2, offset=0.2)",
        "asr_options": {
            "initial_prompt": INITIAL_PROMPT,
            "beam_size": 5,
            "best_of": 5,
        },
        "vad_options": {
            "vad_onset": 0.2,
            "vad_offset": 0.2,
        },
    },
    "4_no_suppress_blank": {
        "desc": "WhisperX 不抑制空白 + 灵敏 VAD",
        "asr_options": {
            "initial_prompt": INITIAL_PROMPT,
            "beam_size": 5,
            "suppress_blank": False,
        },
        "vad_options": {
            "vad_onset": 0.3,
            "vad_offset": 0.3,
        },
    },
    "5_halluc_threshold": {
        "desc": "WhisperX hallucination_silence_threshold=1",
        "asr_options": {
            "initial_prompt": INITIAL_PROMPT,
            "beam_size": 5,
            "hallucination_silence_threshold": 1,
        },
        "vad_options": {
            "vad_onset": 0.3,
            "vad_offset": 0.3,
        },
    },
    "6_condition_prev_text": {
        "desc": "WhisperX condition_on_previous_text=True",
        "asr_options": {
            "initial_prompt": INITIAL_PROMPT,
            "beam_size": 5,
            "condition_on_previous_text": True,
        },
        "vad_options": {
            "vad_onset": 0.3,
            "vad_offset": 0.3,
        },
    },
}


# ──────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────
def download_test_audio() -> Path:
    """Download test video audio using existing downloader."""
    from clients.downloader import VideoDownloader

    dl = VideoDownloader(
        output_dir=BENCHMARK_DIR / "media",
        audio_only=True,
        ffmpeg_location=CONFIG.get("download", {}).get("ffmpeg_location"),
    )

    expected = BENCHMARK_DIR / "media" / f"{TEST_VIDEO_BVID}_test.m4a"
    if expected.exists():
        logger.info(f"Audio already downloaded: {expected}")
        return expected

    # Try to find any existing audio
    for f in (BENCHMARK_DIR / "media").glob(f"{TEST_VIDEO_BVID}*"):
        if f.suffix in (".m4a", ".mp3", ".wav"):
            logger.info(f"Found existing audio: {f}")
            return f

    logger.info(f"Downloading test audio from {TEST_VIDEO_URL}...")
    result = dl.download(TEST_VIDEO_URL, f"{TEST_VIDEO_BVID}_test")
    if result is None:
        raise RuntimeError("Failed to download test audio")
    return result


# ──────────────────────────────────────────────
# WhisperX benchmark
# ──────────────────────────────────────────────
def run_whisperx_benchmark(audio_path: Path, config_name: str, config: dict) -> BenchmarkResult:
    """Run a single WhisperX benchmark configuration."""
    import whisperx

    output_file = BENCHMARK_DIR / f"{config_name}.txt"
    logger.info(f"\n{'='*60}")
    logger.info(f"Config: {config_name} - {config['desc']}")
    logger.info(f"ASR Options: {config['asr_options']}")
    logger.info(f"VAD Options: {config['vad_options']}")
    logger.info(f"{'='*60}")

    try:
        # Load model with specific config
        start = time.time()

        load_kwargs = {
            "whisper_arch": "large-v3",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "compute_type": "float16",
            "asr_options": config["asr_options"],
        }
        if config["vad_options"]:
            load_kwargs["vad_options"] = config["vad_options"]

        model = whisperx.load_model(**load_kwargs)

        # Load audio
        audio = whisperx.load_audio(str(audio_path))

        # Transcribe
        result = model.transcribe(audio, batch_size=16, language="zh")
        
        elapsed = time.time() - start

        # Extract text
        segments = result.get("segments", [])
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)
        
        # Save output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Config: {config_name}\n")
            f.write(f"# Description: {config['desc']}\n")
            f.write(f"# Segments: {len(segments)}\n")
            f.write(f"# Duration: {elapsed:.1f}s\n")
            f.write(f"# Chars: {len(full_text)}\n\n")
            for seg in segments:
                t_start = seg.get("start", 0)
                t_end = seg.get("end", 0)
                text = seg.get("text", "").strip()
                f.write(f"[{t_start:.1f}-{t_end:.1f}] {text}\n")

        # Count paragraphs (non-empty lines of actual content)
        content_lines = [l for l in full_text.split("。") if l.strip()]

        bench_result = BenchmarkResult(
            config_name=config_name,
            engine="whisperx_large-v3",
            char_count=len(full_text),
            line_count=len(segments),
            paragraph_count=len(content_lines),
            duration_seconds=round(elapsed, 1),
            output_file=str(output_file),
        )

        logger.info(f"✓ Done: {len(full_text)} chars, {len(segments)} segments, {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        bench_result = BenchmarkResult(
            config_name=config_name,
            engine="whisperx_large-v3",
            char_count=0,
            line_count=0,
            paragraph_count=0,
            duration_seconds=0,
            output_file="",
            error=str(e),
        )
    finally:
        # Unload to free VRAM between configs
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return bench_result


# ──────────────────────────────────────────────
# Qwen3 ASR benchmark
# ──────────────────────────────────────────────
def run_qwen3_benchmark(audio_path: Path) -> BenchmarkResult:
    """Run Qwen3-ASR benchmark."""
    output_file = BENCHMARK_DIR / "7_qwen3_asr.txt"
    config_name = "7_qwen3_asr"

    logger.info(f"\n{'='*60}")
    logger.info(f"Config: {config_name} - Qwen3-ASR-1.7B")
    logger.info(f"{'='*60}")

    try:
        from clients.qwen_asr_client import Qwen3ASRClient

        start = time.time()
        client = Qwen3ASRClient(
            model_name=CONFIG.get("qwen3", {}).get("model", "Qwen/Qwen3-ASR-1.7B"),
            device=CONFIG.get("qwen3", {}).get("device", "cuda:0"),
            language=CONFIG.get("qwen3", {}).get("language", "Chinese"),
            ffmpeg_location=CONFIG.get("download", {}).get("ffmpeg_location"),
        )

        text = client.transcribe(audio_path)
        elapsed = time.time() - start

        # Save
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Config: {config_name}\n")
            f.write(f"# Description: Qwen3-ASR-1.7B default\n")
            f.write(f"# Duration: {elapsed:.1f}s\n")
            f.write(f"# Chars: {len(text)}\n\n")
            f.write(text)

        lines = [l for l in text.split("\n") if l.strip()]
        sentences = [s for s in text.split("。") if s.strip()]

        bench_result = BenchmarkResult(
            config_name=config_name,
            engine="qwen3_asr_1.7b",
            char_count=len(text),
            line_count=len(lines),
            paragraph_count=len(sentences),
            duration_seconds=round(elapsed, 1),
            output_file=str(output_file),
        )

        logger.info(f"✓ Done: {len(text)} chars, {elapsed:.1f}s")

        # Cleanup
        client.unload_model()

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        bench_result = BenchmarkResult(
            config_name=config_name,
            engine="qwen3_asr_1.7b",
            char_count=0,
            line_count=0,
            paragraph_count=0,
            duration_seconds=0,
            output_file="",
            error=str(e),
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return bench_result


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────
def generate_report(results: list[BenchmarkResult]) -> str:
    """Generate markdown comparison report."""
    report_lines = [
        "# ASR 转录基准测试报告",
        "",
        f"**测试视频**: {TEST_VIDEO_URL}",
        f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结果对比",
        "",
        "| # | 配置 | 引擎 | 字符数 | 段落数 | 耗时(秒) | 状态 |",
        "|---|------|------|--------|--------|----------|------|",
    ]

    # Find max char count for highlighting
    max_chars = max((r.char_count for r in results if not r.error), default=0)

    for r in results:
        status = "❌ " + (r.error or "Error")[:30] if r.error else "✅"
        char_marker = " ⭐" if r.char_count == max_chars and not r.error else ""
        report_lines.append(
            f"| {r.config_name.split('_')[0]} | {r.config_name} | {r.engine} | "
            f"{r.char_count}{char_marker} | {r.paragraph_count} | {r.duration_seconds} | {status} |"
        )

    report_lines.extend([
        "",
        "> ⭐ = 最多字符数 (可能意味着最完整的转录)",
        "",
        "## 详细参数",
        "",
    ])

    # Add config details
    for name, cfg in WHISPERX_CONFIGS.items():
        report_lines.append(f"### {name}")
        report_lines.append(f"- **描述**: {cfg['desc']}")
        report_lines.append(f"- **ASR**: `{cfg['asr_options']}`")
        report_lines.append(f"- **VAD**: `{cfg['vad_options']}`")
        report_lines.append("")

    report_lines.extend([
        "### 7_qwen3_asr",
        "- **描述**: Qwen3-ASR-1.7B 默认参数",
        "- **引擎**: Alibaba Qwen3-ASR",
        "",
        "---",
        "",
        "## 如何使用结果",
        "",
        "1. 比较字符数：越多通常代表转录越完整",
        "2. 查看各配置的输出文件，搜索你知道视频中有但可能被遗漏的内容",
        "3. 选择最佳配置后，更新 `whisperx_client.py` 中的参数",
    ])

    return "\n".join(report_lines)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ASR Benchmark Tool")
    parser.add_argument("--audio", type=Path, help="Path to existing audio file (skip download)")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen3-ASR test")
    parser.add_argument("--only", type=str, help="Run only a specific config (e.g. '2_sensitive_vad')")
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get audio
    if args.audio:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return 1
    else:
        audio_path = download_test_audio()

    logger.info(f"Audio file: {audio_path} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")

    results: list[BenchmarkResult] = []

    # Step 2: Run WhisperX configs
    for name, cfg in WHISPERX_CONFIGS.items():
        if args.only and name != args.only:
            continue
        result = run_whisperx_benchmark(audio_path, name, cfg)
        results.append(result)

    # Step 3: Run Qwen3 ASR
    if not args.skip_qwen and not args.only:
        result = run_qwen3_benchmark(audio_path)
        results.append(result)

    # Step 4: Generate report
    report = generate_report(results)
    report_path = BENCHMARK_DIR / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also save raw results as JSON
    json_path = BENCHMARK_DIR / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmark complete!")
    logger.info(f"Report: {report_path}")
    logger.info(f"Results: {json_path}")
    logger.info(f"{'='*60}")

    # Print summary table
    print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
