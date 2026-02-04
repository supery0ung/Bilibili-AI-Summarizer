#!/usr/bin/env python3
"""Test Whisper with transcribe_to_markdown (includes title, author, paragraphs)."""

import os
import sys
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Set model cache to E drive
os.environ.setdefault("WHISPER_CACHE", "E:/ai_models/whisper")

PROJECT_ROOT = Path(__file__).resolve().parent

# Direct import to avoid circular import
import importlib.util
spec = importlib.util.spec_from_file_location("whisper_client", PROJECT_ROOT / "clients" / "whisper_client.py")
whisper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(whisper_module)
WhisperClient = whisper_module.WhisperClient

def main():
    # Find the test audio file
    media_dir = Path("E:/bilibili_summarizer_v3/output/media")
    audio_files = list(media_dir.glob("BV1eK6iBxEr5*.m4a"))
    
    if not audio_files:
        print("Audio file not found!")
        return 1
    
    audio_path = audio_files[0]
    
    # Extract title from filename (remove BVID prefix)
    filename = audio_path.stem
    title = filename.split("_", 1)[1] if "_" in filename else filename
    
    print(f"Testing Whisper transcribe_to_markdown")
    print(f"Audio: {audio_path.name}")
    print(f"Title: {title}")
    print(f"File size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("-" * 60)
    
    # Load config for ffmpeg path
    import yaml
    config_path = PROJECT_ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ffmpeg_location = config.get("download", {}).get("ffmpeg_location")
    
    # Initialize Whisper (language=None for auto-detect)
    client = WhisperClient(
        model_name="large-v3",
        device="auto",
        language=None,  # Auto-detect
        ffmpeg_location=ffmpeg_location,
    )
    
    # Transcribe to Markdown (with title, author, paragraphs)
    print("\nStarting transcription...")
    markdown = client.transcribe_to_markdown(
        audio_path,
        title=title,
        author="小王albert",  # 正确的UP主
    )
    
    # Save result
    output_path = PROJECT_ROOT / "test_whisper_markdown_output.md"
    output_path.write_text(markdown, encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("MARKDOWN OUTPUT (first 2000 chars):")
    print("=" * 60)
    print(markdown[:2000])
    if len(markdown) > 2000:
        print(f"\n... [truncated, total {len(markdown)} chars]")
    
    print(f"\n✓ Full result saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
