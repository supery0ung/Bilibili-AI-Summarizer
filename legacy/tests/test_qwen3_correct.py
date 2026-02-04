#!/usr/bin/env python3
"""Test Qwen3-8B for correcting Whisper transcript and generating summary."""

import sys
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent

# Direct import
import importlib.util
spec = importlib.util.spec_from_file_location("ollama_client", PROJECT_ROOT / "clients" / "ollama_client.py")
ollama_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ollama_module)
OllamaClient = ollama_module.OllamaClient


def main():
    # Read Whisper output
    whisper_output = PROJECT_ROOT / "test_whisper_markdown_output.md"
    
    if not whisper_output.exists():
        print(f"Error: {whisper_output} not found!")
        return 1
    
    content = whisper_output.read_text(encoding="utf-8")
    
    # Extract parts
    lines = content.split("\n")
    title = lines[0].lstrip("# ").strip() if lines else "Unknown"
    
    # Find author line
    author = ""
    for line in lines:
        if line.startswith("**UP主**:"):
            author = line.replace("**UP主**:", "").strip()
            break
    
    # Extract transcript (everything after ---)
    transcript_start = content.find("---")
    if transcript_start != -1:
        transcript = content[transcript_start + 3:].strip()
    else:
        transcript = content
    
    print("=" * 60)
    print("Qwen3-8B Text Correction & Summarization")
    print("=" * 60)
    print(f"Title: {title}")
    print(f"Author: {author}")
    print(f"Transcript length: {len(transcript)} chars")
    print("-" * 60)
    
    # Initialize Qwen3 client
    print("\nConnecting to Ollama...")
    client = OllamaClient(model="qwen3:8b")
    
    # Run correction and summarization
    print("\nProcessing with Qwen3-8B (this may take a few minutes)...\n")
    result = client.correct_and_summarize(
        transcript=transcript,
        title=title,
        author=author,
    )
    
    # Build output markdown
    output_lines = [
        f"# {title}",
        "",
        f"**UP主**: {author}",
        "",
        "---",
        "",
        "## 内容摘要",
        "",
        result["summary"],
        "",
        "## 要点列表",
        "",
        result["outline"],
        "",
        "---",
        "",
        "## 校正后的完整文本",
        "",
        result["corrected_text"],
    ]
    
    output_md = "\n".join(output_lines)
    
    # Save result
    output_path = PROJECT_ROOT / "test_qwen3_corrected_output.md"
    output_path.write_text(output_md, encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("OUTPUT PREVIEW")
    print("=" * 60)
    
    # Print summary and outline
    print("\n## 内容摘要\n")
    print(result["summary"])
    print("\n## 要点列表\n")
    print(result["outline"])
    
    print("\n" + "=" * 60)
    print(f"✓ Full result saved to: {output_path}")
    print(f"  - Corrected text: {len(result['corrected_text'])} chars")
    print(f"  - Summary: {len(result['summary'])} chars")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
