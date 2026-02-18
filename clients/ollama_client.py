"""Ollama LLM client for text correction and summarization."""

from __future__ import annotations

import re
import requests
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Default prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# ASR Hallucination patterns to filter out (common Whisper artifacts)
ASR_HALLUCINATION_PATTERNS = [
    # 明镜与点点 promotion (very common Whisper hallucination for Chinese content)
    r"请不吝点赞\s*订阅\s*转发\s*打赏支持明镜与点点栏目",
    r"请不吝点赞、订阅、转发、打赏支持明镜与点点栏目",
    # Generic end-of-video phrases that Whisper often hallucinates
    r"谢谢收看",
    r"感谢收看",
    # Redundant prompt echoes from initial_prompt leaking into output
    r"请注意使用逗号、句号和感叹号。这是一段视频的语音转录，",
    r"这是一段视频的语音转录，请注意使用逗号、句号和感叹号。",
]


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
                logger.warning(f"Model '{self.model}' not found. Available: {models}")
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")

    def unload_model(self):
        """Unload the model from Ollama server to free VRAM."""
        logger.info(f"Unloading Ollama model '{self.model}'...")
        try:
            # Setting keep_alive to 0 in a generate request unloads the model immediately
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=5
            )
            logger.info("  ✓ Ollama model unloaded")
        except Exception as e:
            logger.error(f"  ✗ Failed to unload model: {e}")

    def should_filter(self, title: str, author: str) -> bool:
        """Use LLM to decide if a video should be kept or skipped.
        
        Returns:
            True if video should be KEPT, False if it should be SKIPPED.
        """
        try:
            prompt_template = self._load_prompt("filter")
            prompt = prompt_template.replace("{title}", title).replace("{author}", author)
            
            result = self.generate(
                prompt,
                temperature=0.0,  # Deterministic
                max_tokens=512,
            ).upper().strip()
            
            if "SKIP" in result:
                return False
            return True  # Default to KEEP if uncertain
            
        except Exception as e:
            logger.error(f"    [Error] AI filtering failed: {e}")
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

    def _filter_asr_hallucinations(self, text: str) -> str:
        """Remove common ASR hallucination patterns from text.
        
        These patterns are artifacts from Whisper's pre-training data that
        appear when the model encounters silence or background music.
        """
        filtered = text
        for pattern in ASR_HALLUCINATION_PATTERNS:
            filtered = re.sub(pattern, '', filtered)
        
        # Clean up any resulting double spaces or empty lines
        filtered = re.sub(r'  +', ' ', filtered)
        filtered = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered)
        
        return filtered.strip()

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
        
        # Explicitly decode as UTF-8 to avoid platform-specific encoding issues
        import json
        resp_json = json.loads(resp.content.decode('utf-8'))
        response = resp_json.get("response", "").strip()
        
        return self._clean_response(response)

    def identify_speakers(self, text: str, title: str = "", author: str = "") -> dict[str, str]:
        """Identify speakers from the beginning of the transcript.
        
        Returns:
            Dictionary mapping speaker labels to names.
        """
        # Read only the beginning of the text (e.g., first 5000 chars)
        sample_text = text[:5000]
        
        try:
            prompt_template = self._load_prompt("identify_speakers")
            prompt = prompt_template.replace("{text}", sample_text).replace("{title}", title or "未知标题").replace("{author}", author or "未知UP主")
            
            result = self.generate(
                prompt,
                temperature=0.0,  # Strict JSON output
                max_tokens=512,
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                import json
                mapping = json.loads(json_match.group(0))
                # Filter out null values and normalize keys (remove brackets)
                normalized = {}
                for k, v in mapping.items():
                    if v and str(v).lower() != "null":
                        # Strip common brackets and spaces from key
                        clean_k = re.sub(r'[\[\]\s]', '', k)
                        # Fallback: if value is literally author name, map to "主持人"
                        if author and str(v).strip() == author:
                            v = "主持人"
                        normalized[clean_k] = v
                return normalized
            return {}
            
        except Exception as e:
            logger.error(f"    [Error] Speaker identification failed: {e}")
            return {}

    def correct_paragraph(self, text: str, title: str = "", author: str = "", speaker_map: dict = None, language: str = "zh") -> str:
        """Correct a single paragraph of ASR text.
        
        Returns original text if correction fails or output is invalid.
        """
        if len(text.strip()) < 20:
            return text
        
        try:
            prompt_template = self._load_prompt("correct")
            
            # Format speaker map into a readable string
            speaker_info = "无"
            if speaker_map:
                speaker_info = ", ".join([f"{k} 是 {v}" for k, v in speaker_map.items()])
            
            extra_instructions = ""
            if language and language.lower() not in ("zh", "chinese", "cmn"):
                extra_instructions = (
                    "### 强制性双语要求 (Bilingual Requirement):\n"
                    "1. **保留原文**：保留每一段原始语言文本。\n"
                    "2. **紧跟翻译**：在每一段原文之后，换行附带简体中文翻译。\n"
                    "3. **格式要求**：直接输出对比文本，不要包含任何如 '[Original Paragraph]' 或 '[中文翻译段落]' 之类的标签。"
                )
                
            prompt = prompt_template.replace("{text}", text) \
                                   .replace("{title}", title or "未知标题") \
                                   .replace("{author}", author or "未知UP主") \
                                   .replace("{speaker_map}", speaker_info) \
                                   .replace("{language_hint}", "") \
                                   .replace("{extra_instructions}", extra_instructions)
            
            result = self.generate(
                prompt,
                temperature=0.1,
                max_tokens=len(text) * 2 + 500,
            )
            
            # Validate result length
            if len(result) < len(text) * 0.5 or len(result) > len(text) * 2:
                return text
                
            return result
            
        except Exception as e:
            logger.error(f"    [Error] Correction failed: {e}")
            return text

    def correct_text(self, text: str, title: str = "", author: str = "", speaker_map: dict = None, language: str = "zh", progress_callback=None) -> str:
        """Deprecated: Use correct_text_batched for better performance."""
        return self.correct_text_batched(text, title, author, speaker_map, language, progress_callback)

    def correct_text_batched(self, text: str, title: str = "", author: str = "", speaker_map: dict = None, language: str = "zh", progress_callback=None) -> str:
        """Correct full text by processing in larger chunks (much faster).
        
        Args:
            text: Full text to correct
            title: Video title
            author: UP主 name
            speaker_map: Pre-identified speaker mapping
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Corrected text with paragraphs preserved
        """
        import re
        # Pattern to match speaker label at start of paragraph
        speaker_prefix_pattern = re.compile(r'^(\*\*\[[^\]]+\]\*\*[：:\s]*)')
        
        paragraphs = [p for p in text.split("\n") if p.strip()]
        if not paragraphs:
            return text

        # Group paragraphs into chunks (approx 2000-3000 chars)
        chunks = []
        current_chunk = []
        current_len = 0
        target_chunk_len = 2500

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > target_chunk_len and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len + 2
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        corrected_chunks = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            
            # For each chunk, we can either strip prefixes or send as is.
            # Stripping is safer but more complex in chunks. 
            # Let's try sending chunks directly with the speaker_map context.
            try:
                prompt_template = self._load_prompt("correct")
                speaker_info = "无"
                if speaker_map:
                    speaker_info = ", ".join([f"{k} 是 {v}" for k, v in speaker_map.items()])
                
                extra_instructions = ""
                # Determine if we need bilingual output
                is_chinese = language and language.lower() in ("zh", "chinese", "cmn")
                if not is_chinese:
                    extra_instructions = (
                        "### 强制性双语要求 (Bilingual Requirement):\n"
                        "1. **保留原文**：每一段必须先输出原始语言文本。\n"
                        "2. **紧跟翻译**：在每一段原文之后，必须紧跟其对应的简体中文翻译。\n"
                        "3. **格式要求**：直接输出双语对比文本，**严禁**包含任何如 '[Original Paragraph]' 或 '[中文翻译段落]' 这种额外的辅助标签。\n"
                        "4. **严禁遗漏**：严禁只输出中文或只输出原文，必须保持双语对照。"
                    )
                    
                prompt = prompt_template.replace("{text}", chunk) \
                                       .replace("{title}", title or "未知标题") \
                                       .replace("{author}", author or "未知UP主") \
                                       .replace("{speaker_map}", speaker_info) \
                                       .replace("{language_hint}", "") \
                                       .replace("{extra_instructions}", extra_instructions)
                
                # Use higher max_tokens for chunks
                batch_corrected = self.generate(
                    prompt,
                    temperature=0.1,
                    max_tokens=len(chunk) * 2 + 1000,
                )
                
                # Fallback if LLM fails or returns garbage
                if len(batch_corrected) < len(chunk) * 0.3:
                    logger.warning(f"  Chunk {i+1} correction seems too short, keeping original.")
                    corrected_chunks.append(chunk)
                else:
                    corrected_chunks.append(batch_corrected)
            except Exception as e:
                logger.error(f"  Chunk {i+1} correction failed: {e}")
                corrected_chunks.append(chunk)

        result = "\n\n".join(corrected_chunks)
        
        # Post-process cleanup (consistent with original logic)
        result = re.sub(r'(\*\*\[[^\]]+\]\*\*[：:\s]*)\1+', r'\1', result)
        
        if speaker_map:
            for raw_tag, real_name in speaker_map.items():
                id_match = re.search(r"(?:说话人|SPEAKER)[\s_\u3000]*([A-Z0-9]+)", raw_tag, re.I)
                tag_id = id_match.group(1) if id_match else raw_tag
                tag_pattern = rf"\*\*\[?(?:说话人|SPEAKER)[\s_\u3000]*{tag_id}\]?\*\*[:：\s]*"
                new_tag = f"**[{real_name}]** "
                result = re.sub(tag_pattern, new_tag, result)
        
        result = re.sub(r'(\*\*\[[^\]]+\]\*\*\s*)\1+', r'\1', result)
        return self._filter_asr_hallucinations(result)
        
        # Second cleanup pass: remove potential redundant name tags created by LLM and us
        # e.g. "**[闫俊杰]** **[闫俊杰]** text"
        result = re.sub(r'(\*\*\[[^\]]+\]\*\*\s*)\1+', r'\1', result)
        
        # Apply ASR hallucination filtering as final post-processing step
        return self._filter_asr_hallucinations(result)

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
            prompt = prompt_template.replace("{text}", text).replace("{title}", title or "未知标题").replace("{author}", author or "未知UP主")
            
            result = self.generate(
                prompt,
                temperature=0.3,
                max_tokens=4096,
            )
            
            if len(result) < 100:
                logger.warning(f"    [Warning] Summary too short ({len(result)} chars).")
            
            return result
        except requests.Timeout:
            logger.error("    [Error] Ollama summarization timed out.")
            return f"总结失败：API 超时。原始内容长度: {len(text)}"
        except Exception as e:
            logger.error(f"    [Error] Summarization failed: {e}")
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
        logger.info("Step 1/2: Correcting transcript...")
        corrected = self.correct_text(text, progress_callback)
        
        logger.info("Step 2/2: Generating summary...")
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
        logger.info(f"Ollama test successful: {response[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Ollama test failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
