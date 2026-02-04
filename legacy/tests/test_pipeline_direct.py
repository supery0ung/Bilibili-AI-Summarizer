#!/usr/bin/env python3
"""Direct test of pipeline steps D, E, F using existing files."""
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Direct imports
import importlib.util
spec = importlib.util.spec_from_file_location("ollama_client", PROJECT_ROOT / "clients" / "ollama_client.py")
ollama_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ollama_module)
OllamaClient = ollama_module.OllamaClient
build_final_markdown = ollama_module.build_final_markdown

from utils import convert_md_to_epub


def main():
    transcripts_dir = Path("E:/bilibili_summarizer_v3/output/transcripts")
    epub_dir = Path("E:/bilibili_summarizer_v3/output/epub")
    
    # Find existing transcripts
    print("=== Existing Transcripts ===")
    transcripts = list(transcripts_dir.glob("*.md"))
    for t in transcripts:
        if not t.name.endswith(".corrected.md") and not t.name.endswith(".final.md"):
            print(f"  {t.name} ({t.stat().st_size} bytes)")
    
    # Use the first non-corrected, non-final transcript
    test_file = None
    for t in transcripts:
        if not t.name.endswith(".corrected.md") and not t.name.endswith(".final.md"):
            if "美元之锚" in t.name or "铜命脉" in t.name or "欧盟" in t.name:
                test_file = t
                break
    
    if not test_file:
        print("No suitable test file found!")
        return 1
    
    print(f"\n=== Using: {test_file.name} ===")
    
    # Read content
    content = test_file.read_text(encoding="utf-8")
    lines = content.split("\n")
    title = lines[0].lstrip("# ").strip() if lines else "Unknown"
    
    # Find author
    author = ""
    for line in lines:
        if line.startswith("**UP主**:"):
            author = line.replace("**UP主**:", "").strip()
            break
    
    # Extract body
    body_start = content.find("---")
    if body_start != -1:
        body_start = content.find("\n", body_start) + 1
        raw_text = content[body_start:].strip()
    else:
        raw_text = content
    
    print(f"Title: {title}")
    print(f"Author: {author}")
    print(f"Text length: {len(raw_text)} chars")
    
    # Initialize Ollama
    print("\n=== Step D: Correct ===")
    client = OllamaClient(
        model="qwen3:8b",
        prompts_dir=PROJECT_ROOT / "prompts",
    )
    
    def progress_cb(current, total):
        print(f"  Correcting {current}/{total}...", end="\r")
    
    corrected_text = client.correct_text(raw_text, progress_callback=progress_cb)
    print()
    print(f"  ✓ Corrected: {len(corrected_text)} chars")
    
    # Save corrected
    corrected_path = test_file.with_suffix(".corrected.md")
    corrected_md = f"# {title}\n\n**UP主**: {author}\n\n---\n\n{corrected_text}"
    corrected_path.write_text(corrected_md, encoding="utf-8")
    print(f"  ✓ Saved: {corrected_path.name}")
    
    # Step E: Summarize
    print("\n=== Step E: Summarize ===")
    summary = client.summarize(corrected_text, title=title, author=author)
    print(f"  ✓ Summary: {len(summary)} chars")
    
    # Build final markdown
    final_md = build_final_markdown(
        title=title,
        author=author,
        summary=summary,
        corrected_text=corrected_text,
    )
    
    final_path = test_file.with_suffix(".final.md")
    final_path.write_text(final_md, encoding="utf-8")
    print(f"  ✓ Saved: {final_path.name}")
    
    # Step F: EPUB
    print("\n=== Step F: EPUB ===")
    epub_path = epub_dir / f"{test_file.stem}.epub"
    convert_md_to_epub(final_path, epub_path, title)
    print(f"  ✓ Saved: {epub_path.name}")
    
    print("\n=== Pipeline Test Complete ===")
    print(f"Corrected: {corrected_path}")
    print(f"Final MD:  {final_path}")
    print(f"EPUB:      {epub_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
