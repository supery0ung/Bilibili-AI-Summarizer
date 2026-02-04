#!/usr/bin/env python3
"""Bilibili Summarizer V3 - Whisper Transcription Edition.

Usage:
    python main.py run              # Run complete pipeline (A → F)
    python main.py fetch            # Step A: Fetch + filter + build queue
    python main.py download         # Step B: Download audio
    python main.py transcribe       # Step C: Transcribe with Whisper
    python main.py correct          # Step D: Correct with Qwen3
    python main.py summarize        # Step E: Summarize with Qwen3
    python main.py epub             # Step F: Convert to EPUB
    python main.py upload           # Step G: Upload to WeChat Reading
    python main.py status           # Show current status
"""

from __future__ import annotations

# === Model Cache Configuration ===
# Default to ~/ai_models if not specified
import os
MODEL_BASE = os.environ.get("MODEL_BASE_DIR", os.path.join(os.path.expanduser("~"), "ai_models")).replace("\\", "/")

os.environ.setdefault("XDG_CACHE_HOME", f"{MODEL_BASE}")
os.environ.setdefault("WHISPER_CACHE", f"{MODEL_BASE}/whisper")
os.environ.setdefault("HF_HOME", f"{MODEL_BASE}/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{MODEL_BASE}/huggingface/hub")

import argparse
import json
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import Pipeline


def cmd_run(args: argparse.Namespace) -> int:
    """Run complete pipeline - processes videos one by one."""
    pipeline = Pipeline(args.config, headless=args.headless)
    results = pipeline.run_all(
        max_items=args.max_items,
        upload=getattr(args, 'upload', False)
    )
    return 0 if results.get("error", 0) == 0 else 1


def cmd_fetch(args: argparse.Namespace) -> int:
    """Step A: Fetch + filter + build queue."""
    pipeline = Pipeline(args.config)
    queue = pipeline.run_step_a()

    if queue:
        print("\nQueued videos:")
        for i, item in enumerate(queue[:10], 1):
            print(f"  {i}. {item.title[:60]}")
        if len(queue) > 10:
            print(f"  ... and {len(queue) - 10} more")

    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Step B: Download audio from videos."""
    pipeline = Pipeline(args.config)
    stats = pipeline.run_step_b_download(max_items=args.max_items)

    if stats.get("error", 0) > 0 or stats.get("download_failed", 0) > 0:
        return 1
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Step C: Transcribe downloaded audio with Whisper."""
    pipeline = Pipeline(args.config, headless=args.headless)
    stats = pipeline.run_step_c_transcribe(max_items=args.max_items)

    if stats.get("error", 0) > 0:
        return 1
    return 0


def cmd_correct(args: argparse.Namespace) -> int:
    """Step D: Correct transcripts with Qwen3."""
    pipeline = Pipeline(args.config)
    stats = pipeline.run_step_d_correct(max_items=args.max_items)
    return 0 if stats.get("error", 0) == 0 else 1


def cmd_summarize(args: argparse.Namespace) -> int:
    """Step E: Summarize corrected transcripts with Qwen3."""
    pipeline = Pipeline(args.config)
    stats = pipeline.run_step_e_summarize(max_items=args.max_items)
    return 0 if stats.get("error", 0) == 0 else 1


def cmd_epub(args: argparse.Namespace) -> int:
    """Step F: Convert transcripts to EPUB."""
    pipeline = Pipeline(args.config)
    stats = pipeline.run_step_f_epub(force_all=getattr(args, 'force_all', False))
    return 0 if stats.get("error", 0) == 0 else 1


def cmd_upload(args: argparse.Namespace) -> int:
    """Step G: Upload to WeChat Reading."""
    pipeline = Pipeline(args.config, headless=args.headless)
    stats = pipeline.run_step_g_upload(max_items=args.max_items)
    return 0 if stats.get("error", 0) == 0 else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show current pipeline status."""
    pipeline = Pipeline(args.config)
    status = pipeline.get_status()

    print("=== Pipeline Status ===")
    print(f"\nState file: {status['state_file']}")
    print(f"Queue file: {status['queue_file']}")
    print(f"EPUBs generated: {status['epub_count']}")
    print(f"Media files: {status.get('media_count', 0)}")

    print("\nVideo statistics:")
    for stat, count in sorted(status['video_stats'].items()):
        print(f"  {stat}: {count}")

    # List uploaded titles
    uploaded_items = [
        v for v in pipeline.state._state["videos"].values() 
        if v.get("status") == "uploaded"
    ]
    if uploaded_items:
        print("\nRecently Uploaded Titles:")
        # Sort by last_attempt if available, or just take last 5
        sorted_uploaded = sorted(
            uploaded_items, 
            key=lambda x: x.get("last_attempt", ""), 
            reverse=True
        )
        for item in sorted_uploaded[:10]:
            title = item.get("title", "Unknown Title")
            print(f"  ✓ {title}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bilibili Summarizer V3 - Whisper Transcription Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run complete pipeline (video by video)")
    run_parser.add_argument(
        "--max-items", type=int, default=10,
        help="Maximum items to process (default: 10)"
    )
    run_parser.add_argument(
        "--no-upload", action="store_false", dest="upload", default=True,
        help="Do NOT upload generated EPUBs to WeChat Reading"
    )
    run_parser.add_argument(
        "--headful", action="store_false", dest="headless", default=True,
        help="Run browser in headful (visible) mode instead of headless"
    )
    run_parser.set_defaults(func=cmd_run)

    # fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch + filter + build queue")
    fetch_parser.set_defaults(func=cmd_fetch)

    # download command
    download_parser = subparsers.add_parser("download", help="Step B: Download audio")
    download_parser.add_argument(
        "--max-items", type=int, default=10,
        help="Maximum items to download (default: 10)"
    )
    download_parser.set_defaults(func=cmd_download)

    # transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Step C: Transcribe with Whisper")
    transcribe_parser.add_argument(
        "--max-items", type=int, default=10,
        help="Maximum items to transcribe (default: 10)"
    )
    transcribe_parser.add_argument(
        "--headful", action="store_false", dest="headless", default=True,
        help="Run browser in headful (visible) mode instead of headless"
    )
    transcribe_parser.set_defaults(func=cmd_transcribe)

    # correct command
    correct_parser = subparsers.add_parser("correct", help="Step D: Correct with Qwen3")
    correct_parser.add_argument(
        "--max-items", type=int, default=10,
        help="Maximum items to process (default: 10)"
    )
    correct_parser.set_defaults(func=cmd_correct)

    # summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Step E: Summarize with Qwen3")
    summarize_parser.add_argument(
        "--max-items", type=int, default=10,
        help="Maximum items to process (default: 10)"
    )
    summarize_parser.set_defaults(func=cmd_summarize)

    # epub command
    epub_parser = subparsers.add_parser("epub", help="Step F: Convert to EPUB")
    epub_parser.add_argument(
        "--force-all", action="store_true",
        help="Regenerate EPUBs for all items with transcripts"
    )
    epub_parser.set_defaults(func=cmd_epub)

    # upload command
    upload_parser = subparsers.add_parser("upload", help="Step G: Upload to WeChat Reading")
    upload_parser.add_argument(
        "--max-items", type=int, default=10,
        help="Maximum items to upload (default: 10)"
    )
    upload_parser.add_argument(
        "--headful", action="store_false", dest="headless", default=True,
        help="Run browser in headful (visible) mode instead of headless"
    )
    upload_parser.set_defaults(func=cmd_upload)

    # status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
