"""Ollama LLM client for text correction and summarization."""

from __future__ import annotations

import re
import requests
from pathlib import Path
from typing import Optional


# Default prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class OllamaClient:
    """Client for Ollama API to run local LLMs like Qwen3."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        prompts_dir: Optional[Path] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompts_dir = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
        self._verify_connection()

    def _verify_connection(self):
        """Verify Ollama server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                print(f"Warning: Model '{self.model}' not found. Available: {models}")
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")

    def unload_model(self):
        """Unload the model from Ollama server to free VRAM."""
        print(f"Unloading Ollama model '{self.model}'...")
        try:
            # Setting keep_alive to 0 in a generate request unloads the model immediately
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=5
            )
            print("  ✓ Ollama model unloaded")
        except Exception as e:
            print(f"  ✗ Failed to unload model: {e}")

    def should_filter(self, title: str, author: str) -> bool:
        """Use LLM to decide if a video should be kept or skipped.
        
        Returns:
            True if video should be KEPT, False if it should be SKIPPED.
        """
        try:
            prompt_template = self._load_prompt("filter")
            prompt = prompt_template.format(title=title, author=author)
            
            result = self.generate(
                prompt,
                temperature=0.0,  # Deterministic
                max_tokens=512,
            ).upper().strip()
            
            if "SKIP" in result:
                return False
            return True  # Default to KEEP if uncertain
            
        except Exception as e:
            print(f"    [Error] AI filtering failed: {e}")
            return True  # Keep on error

    def _load_prompt(self, name: str) -> str:
        """Load prompt template from file."""
        prompt_file = self.prompts_dir / f"{name}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8")

    def _clean_response(self, response: str) -> str:
        """Remove <think> blocks from response."""
        # Remove closed <think> blocks
        response = re.sub(r'<think>[\s\S]*?</think>', '', response)
        # Remove unclosed <think> blocks
        response = re.sub(r'<think>[\s\S]*', '', response)
        return response.strip()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> str:
        """Generate text completion."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=600,  # 10 minutes for long generations
        )
        resp.raise_for_status()
        response = resp.json().get("response", "").strip()
        
        return self._clean_response(response)

    def correct_paragraph(self, text: str) -> str:
        """Correct a single paragraph of ASR text.
        
        Returns original text if correction fails or output is invalid.
        """
        if len(text.strip()) < 20:
            return text
        
        try:
            prompt_template = self._load_prompt("correct")
            prompt = prompt_template.format(text=text)
            
            result = self.generate(
                prompt,
                temperature=0.1,
                max_tokens=len(text) * 2,
            )
            
            # Validate result length
            if len(result) < len(text) * 0.5 or len(result) > len(text) * 2:
                return text
                
            return result
            
        except Exception as e:
            print(f"    [Error] Correction failed: {e}")
            return text

    def correct_text(self, text: str, progress_callback=None) -> str:
        """Correct full text by processing paragraph by paragraph.
        
        Args:
            text: Full text to correct
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Corrected text with paragraphs preserved
        """
        paragraphs = [p for p in text.split("\n") if p.strip()]
        corrected = []
        
        for i, para in enumerate(paragraphs):
            if progress_callback:
                progress_callback(i + 1, len(paragraphs))
            
            corrected_para = self.correct_paragraph(para)
            corrected.append(corrected_para)
        
        return "\n\n".join(corrected)

    def summarize(
        self,
        text: str,
        title: str = "",
        author: str = "",
    ) -> str:
        """Generate summary of the text.
        
        Args:
            text: The full transcript text
            title: Video title for context
            author: UP主 name
            
        Returns:
            Markdown formatted summary with outline
        """
        try:
            prompt_template = self._load_prompt("summarize")
            prompt = prompt_template.format(
                text=text,
                title=title or "未知标题",
                author=author or "未知UP主",
            )
            
            result = self.generate(
                prompt,
                temperature=0.3,
                max_tokens=4096,
            )
            
            if len(result) < 100:
                print(f"    [Warning] Summary too short ({len(result)} chars).")
            
            return result
        except requests.Timeout:
            print("    [Error] Ollama summarization timed out.")
            return f"总结失败：API 超时。原始内容长度: {len(text)}"
        except Exception as e:
            print(f"    [Error] Summarization failed: {e}")
            return f"总结失败：{str(e)}"

    def process_transcript(
        self,
        text: str,
        title: str = "",
        author: str = "",
        progress_callback=None,
    ) -> dict:
        """Full processing pipeline: correct + summarize.
        
        Args:
            text: Raw ASR transcript
            title: Video title
            author: UP主 name
            progress_callback: Optional callback for progress updates
            
        Returns:
            dict with keys: corrected_text, summary, title, author
        """
        print("Step 1/2: Correcting transcript...")
        corrected = self.correct_text(text, progress_callback)
        
        print("Step 2/2: Generating summary...")
        summary = self.summarize(corrected, title=title, author=author)
        
        return {
            "title": title,
            "author": author,
            "corrected_text": corrected,
            "summary": summary,
        }


def build_final_markdown(
    title: str,
    author: str,
    summary: str,
    corrected_text: str,
) -> str:
    """Build final markdown document combining summary and full text.
    
    Args:
        title: Video title
        author: UP主 name
        summary: Generated summary (markdown format)
        corrected_text: Corrected full transcript
        
    Returns:
        Complete markdown document
    """
    lines = [
        f"# {title}",
        "",
        f"**UP主**: {author}",
        "",
        "---",
        "",
        summary,
        "",
        "---",
        "",
        "## 完整文本",
        "",
        corrected_text,
    ]
    
    return "\n".join(lines)


def test_connection():
    """Quick test to verify Ollama is working."""
    try:
        client = OllamaClient()
        response = client.generate("你好，请用一句话介绍自己。/no_think", max_tokens=100)
        print(f"Ollama test successful: {response[:100]}...")
        return True
    except Exception as e:
        print(f"Ollama test failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
