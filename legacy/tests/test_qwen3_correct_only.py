#!/usr/bin/env python3
"""Test Qwen3-8B: Only correct errors, preserve all text - with thinking disabled."""

import sys
import re
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent

import requests


def correct_paragraph(text: str, model: str = "qwen3:8b") -> str:
    """Correct a single paragraph, preserving all content."""
    
    # Skip short or empty text
    if len(text.strip()) < 20:
        return text
    
    # Simple, direct prompt - no system message, disable thinking with /no_think
    prompt = f"""修正以下语音识别文本中的错别字和乱码，保留所有内容，直接输出修正后的文本：

{text}

/no_think"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": len(text) * 2,
        },
    }

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        response = resp.json().get("response", "").strip()
        
        # Remove any <think> blocks (greedy)
        response = re.sub(r'<think>[\s\S]*?</think>', '', response)
        # Also remove unclosed <think> blocks
        response = re.sub(r'<think>[\s\S]*', '', response)
        response = response.strip()
        
        # If response is much shorter than original (>50% reduction), keep original
        if len(response) < len(text) * 0.5:
            print(f"    [Warning] Output too short ({len(response)} vs {len(text)}), keeping original")
            return text
        
        # If response is much longer (>2x), probably added explanations
        if len(response) > len(text) * 2:
            print(f"    [Warning] Output too long ({len(response)} vs {len(text)}), keeping original")
            return text
            
        return response
        
    except Exception as e:
        print(f"    [Error] {e}, keeping original")
        return text


def main():
    # Read input file
    input_file = PROJECT_ROOT / "test_whisper_markdown_output.md"
    content = input_file.read_text(encoding="utf-8")
    
    # Parse header and body
    lines = content.split("\n")
    
    # Find where content starts (after ---)
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break
    
    header_lines = lines[:body_start + 1]
    body_lines = lines[body_start:]
    
    # Get paragraphs (non-empty lines)
    paragraphs = [line for line in body_lines if line.strip()]
    
    print("=" * 60)
    print("Qwen3-8B Text Correction (No Thinking Mode)")
    print("=" * 60)
    print(f"Input file: {input_file.name}")
    print(f"Paragraphs to process: {len(paragraphs)}")
    print("-" * 60)
    
    corrected_paragraphs = []
    total_original = 0
    total_corrected = 0
    
    for i, para in enumerate(paragraphs, 1):
        print(f"\n[{i}/{len(paragraphs)}] Processing ({len(para)} chars)...")
        
        corrected = correct_paragraph(para)
        corrected_paragraphs.append(corrected)
        
        total_original += len(para)
        total_corrected += len(corrected)
        
        diff = len(corrected) - len(para)
        status = "✓" if abs(diff) < len(para) * 0.2 else "⚠"
        print(f"    {status} Result: {len(corrected)} chars ({diff:+d})")
    
    print("\n" + "-" * 60)
    print(f"Total: {total_original} -> {total_corrected} chars ({total_corrected - total_original:+d})")
    
    # Rebuild document
    output_lines = header_lines.copy()
    for para in corrected_paragraphs:
        output_lines.append("")
        output_lines.append(para)
    
    output_content = "\n".join(output_lines)
    
    # Save result
    output_file = PROJECT_ROOT / "test_qwen3_correct_only_output.md"
    output_file.write_text(output_content, encoding="utf-8")
    
    print("\n" + "=" * 60)
    print(f"✓ Result saved to: {output_file.name}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
