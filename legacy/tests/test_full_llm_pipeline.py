#!/usr/bin/env python3
"""Test full LLM pipeline: correct + summarize + build final markdown."""

import sys
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent

# Direct import to avoid circular import
import importlib.util
spec = importlib.util.spec_from_file_location("ollama_client", PROJECT_ROOT / "clients" / "ollama_client.py")
ollama_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ollama_module)
OllamaClient = ollama_module.OllamaClient
build_final_markdown = ollama_module.build_final_markdown


def main():
    # Read test input
    input_file = PROJECT_ROOT / "test_whisper_markdown_output.md"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return 1
    
    content = input_file.read_text(encoding="utf-8")
    
    # Parse header
    lines = content.split("\n")
    title = lines[0].lstrip("# ").strip() if lines else "Unknown"
    
    author = ""
    for line in lines:
        if line.startswith("**UP主**:"):
            author = line.replace("**UP主**:", "").strip()
            break
    
    # Extract body (after ---)
    body_start = content.find("---")
    if body_start != -1:
        body_start = content.find("\n", body_start) + 1
        raw_text = content[body_start:].strip()
    else:
        raw_text = content
    
    print("=" * 60)
    print("Full LLM Pipeline Test")
    print("=" * 60)
    print(f"Title: {title}")
    print(f"Author: {author}")
    print(f"Input length: {len(raw_text)} chars")
    print("-" * 60)
    
    # Initialize Ollama client
    print("\nConnecting to Ollama...")
    client = OllamaClient(
        model="qwen3:8b",
        prompts_dir=PROJECT_ROOT / "prompts",
    )
    
    # Process
    def progress_cb(current, total):
        print(f"  Correcting paragraph {current}/{total}...", end="\r")
    
    print("\nStep 1/2: Correcting transcript...")
    corrected = client.correct_text(raw_text, progress_callback=progress_cb)
    print()
    print(f"  ✓ Corrected: {len(corrected)} chars")
    
    print("\nStep 2/2: Generating summary...")
    summary = client.summarize(corrected, title=title, author=author)
    print(f"  ✓ Summary: {len(summary)} chars")
    
    # Build final markdown
    final_md = build_final_markdown(
        title=title,
        author=author,
        summary=summary,
        corrected_text=corrected,
    )
    
    # Save output
    output_file = PROJECT_ROOT / "test_full_pipeline_output.md"
    output_file.write_text(final_md, encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("OUTPUT PREVIEW")
    print("=" * 60)
    print(final_md[:2000])
    if len(final_md) > 2000:
        print(f"\n... [truncated, total {len(final_md)} chars]")
    
    print("\n" + "=" * 60)
    print(f"✓ Result saved to: {output_file.name}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
