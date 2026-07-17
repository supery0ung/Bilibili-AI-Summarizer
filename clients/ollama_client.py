"""Ollama LLM client for text correction and summarization."""

from __future__ import annotations

import json
import re
import requests
import logging
import time
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


def _detect_repetition(text: str, min_block_len: int = 50, threshold: float = 0.15) -> bool:
    """Detect if text contains large repeated blocks.

    Scans for any substring of length *min_block_len* that appears more than
    once.  If the total number of repeated characters exceeds *threshold* of
    the text length, the text is considered to have problematic repetition.

    Returns True if repetition is detected.
    """
    if len(text) < min_block_len * 2:
        return False

    seen: dict[str, int] = {}
    repeated_chars = 0
    step = max(1, min_block_len // 4)

    for i in range(0, len(text) - min_block_len + 1, step):
        block = text[i:i + min_block_len]
        if block in seen:
            repeated_chars += min_block_len
        else:
            seen[block] = i

    ratio = repeated_chars / len(text)
    if ratio >= threshold:
        logger.debug(f"Repetition ratio {ratio:.2%} exceeds threshold {threshold:.0%}")
        return True
    return False


def _paragraphs_are_duplicate(a: str, b: str) -> bool:
    """Return True if paragraph *a* is a duplicate / near-duplicate of *b*.

    Three signals, any of which is enough:
      1. Exact match.
      2. High character-level similarity when aligned from the start
         (catches repeats with minor word-level edits).
      3. Containment: a representative 40-char slice of the shorter paragraph
         appears verbatim in the longer one. This catches the common LLM
         failure where a whole earlier block is re-emitted but split at a
         different offset, so prefix-aligned comparison would miss it.
    """
    if a == b:
        return True

    min_len = min(len(a), len(b))
    if min_len > 40:
        matching = sum(1 for x, y in zip(a, b) if x == y)
        if matching > min_len * 0.8:
            return True

        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        # Probe a slice from the middle of the shorter paragraph to avoid
        # boilerplate prefixes (e.g. speaker tags) producing false positives.
        start = len(shorter) // 4
        probe = shorter[start:start + 40]
        if len(probe) >= 40 and probe in longer:
            return True

    return False


def _deduplicate_paragraphs(text: str) -> str:
    """Remove duplicate and near-duplicate paragraphs anywhere in the text.

    Unlike a consecutive-only pass, this compares each paragraph against every
    previously kept paragraph, so a whole block that the LLM re-emits later
    (with an unrelated paragraph in between) is still dropped.
    """
    paragraphs = text.split("\n\n")
    deduped: list[str] = []
    seen: list[str] = []  # stripped form of every kept paragraph

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if any(_paragraphs_are_duplicate(stripped, prev) for prev in seen):
            continue
        deduped.append(para)
        seen.append(stripped)

    return "\n\n".join(deduped)


# Sentence terminators, used to snap loop-collapse cuts to clean boundaries.
_SENT_TERM_RE = re.compile(r'[。！？!?]')

# Motif lengths (chars) tried when scanning a paragraph for a repeated loop.
# Longer first so the collapse latches onto the most specific repeating unit.
_LOOP_MOTIF_LENS = (24, 16, 10)


def _collapse_one_paragraph(para: str, max_scan: int = 4000) -> str:
    """Collapse a degenerate loop within a single paragraph, if present.

    Finds a fixed-length substring (motif) that recurs three or more times
    without overlap, then drops everything from the *second* occurrence through
    the last — snapping both cut points to sentence boundaries so the surviving
    text reads cleanly. Recurses to catch multiple distinct loops.
    """
    if len(para) > max_scan:
        return para

    # Pick the motif with the most non-overlapping occurrences. A motif that
    # spans the connector between repeats only matches every other copy, so
    # going by raw count (tie-break: longer motif) latches onto the tightest
    # in-phrase loop and collapses every copy in one pass.
    best: Optional[tuple[tuple[int, int], int, list[int]]] = None
    for motif_len in _LOOP_MOTIF_LENS:
        if len(para) < motif_len * 3:
            continue

        positions: dict[str, list[int]] = {}
        for i in range(len(para) - motif_len + 1):
            positions.setdefault(para[i:i + motif_len], []).append(i)

        for occ in positions.values():
            if len(occ) < 3:
                continue
            # Keep only non-overlapping occurrences.
            nonoverlap: list[int] = []
            barrier = -1
            for p in occ:
                if p >= barrier:
                    nonoverlap.append(p)
                    barrier = p + motif_len
            if len(nonoverlap) < 3:
                continue

            key = (len(nonoverlap), motif_len)
            if best is None or key > best[0]:
                best = (key, motif_len, nonoverlap)

    if best is None:
        return para

    motif_len, nonoverlap = best[1], best[2]
    second = nonoverlap[1]
    final_end = nonoverlap[-1] + motif_len

    # Cut start: snap back to the sentence boundary before the 2nd hit.
    heads = list(_SENT_TERM_RE.finditer(para[:second]))
    cut_start = heads[-1].end() if heads else second

    # Cut end: snap forward past the sentence containing the last hit.
    tail_match = _SENT_TERM_RE.search(para[final_end:])
    cut_end = final_end + tail_match.end() if tail_match else final_end

    collapsed = para[:cut_start] + para[cut_end:]
    if collapsed != para:
        return _collapse_one_paragraph(collapsed, max_scan)

    return para


def _collapse_internal_loops(text: str) -> str:
    """Collapse degenerate loops *within* paragraphs across the whole text.

    LLM degeneration sometimes repeats the same phrase many times inside one
    paragraph, separated only by short connectors (e.g. "…救出师父。所以孙悟空说：
    …救出师父。孙悟空想的是：…救出师父。"). Cross-paragraph dedup misses this because
    it lives in one block, and the ratio-based repetition detector misses it
    when the loop is only a localized fraction of a long chunk.
    """
    return "\n\n".join(_collapse_one_paragraph(p) for p in text.split("\n\n"))


class OllamaClient:
    """Client for Ollama API to run local LLMs like Qwen3."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        prompts_dir: Optional[Path] = None,
        correction_num_ctx: int = 12288,
        summary_num_ctx: int = 32768,
        keep_alive: str = "30m",
        hierarchical_summary_chars: int = 18000,
        summary_chunk_chars: int = 12000,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompts_dir = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
        self.correction_num_ctx = correction_num_ctx
        self.summary_num_ctx = summary_num_ctx
        self.keep_alive = keep_alive
        self.hierarchical_summary_chars = hierarchical_summary_chars
        self.summary_chunk_chars = summary_chunk_chars
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
                think=False,
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
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        stop_sequences: Optional[list[str]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        think: Optional[bool] = None,
        num_ctx: Optional[int] = None,
        keep_alive: Optional[str] = None,
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
            "keep_alive": (
                keep_alive
                if keep_alive is not None
                else getattr(self, "keep_alive", "30m")
            ),
        }

        # Ollama native thinking control (for Qwen3.5 and other thinking models)
        if think is not None:
            payload["think"] = think

        if top_p is not None: payload["options"]["top_p"] = top_p
        if top_k is not None: payload["options"]["top_k"] = top_k
        if presence_penalty is not None: payload["options"]["presence_penalty"] = presence_penalty
        if repetition_penalty is not None: payload["options"]["repetition_penalty"] = repetition_penalty
        if num_ctx is not None: payload["options"]["num_ctx"] = num_ctx

        if system:
            payload["system"] = system

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        started = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=600,  # 10 minutes for long generations
        )
        resp.raise_for_status()

        resp_json = json.loads(resp.content.decode('utf-8'))
        wall_seconds = time.perf_counter() - started
        load_seconds = resp_json.get("load_duration", 0) / 1_000_000_000
        prompt_count = resp_json.get("prompt_eval_count", 0)
        prompt_seconds = resp_json.get("prompt_eval_duration", 0) / 1_000_000_000
        eval_count = resp_json.get("eval_count", 0)
        eval_seconds = resp_json.get("eval_duration", 0) / 1_000_000_000
        logger.info(
            "Ollama timing: wall=%.1fs load=%.1fs prompt=%d (%.1f tok/s) "
            "output=%d (%.1f tok/s)",
            wall_seconds,
            load_seconds,
            prompt_count,
            prompt_count / prompt_seconds if prompt_seconds else 0.0,
            eval_count,
            eval_count / eval_seconds if eval_seconds else 0.0,
        )

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
                think=False,
                num_ctx=8192,
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
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
                        # Guard against hallucinated names: if the name doesn't
                        # appear anywhere in the sample text, discard it
                        if str(v) != "主持人" and str(v) not in sample_text:
                            logger.warning(f"    [Speaker] Discarding hallucinated name '{v}' for {k} (not found in transcript)")
                            continue
                        normalized[clean_k] = v
                return normalized
            return {}
            
        except Exception as e:
            logger.error(f"    [Error] Speaker identification failed: {e}")
            return {}

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
        # Pattern to match speaker label at start of paragraph
        speaker_prefix_pattern = re.compile(r'^(\*\*\[[^\]]+\]\*\*[：:\s]*)')

        paragraphs = [p for p in text.split("\n") if p.strip()]
        if not paragraphs:
            return text

        # For non-Chinese content, use smaller chunks (1-2 paragraphs) so that
        # bilingual output stays interleaved: English paragraph → Chinese translation.
        # For Chinese content, use larger chunks for speed.
        is_chinese = language and language.lower() in ("zh", "chinese", "cmn")
        target_chunk_len = 4000 if is_chinese else 800

        chunks = []
        current_chunk = []
        current_len = 0

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

        # Pre-build prompt template parts (invariant across chunks)
        prompt_template = self._load_prompt("correct")
        speaker_info = "无"
        if speaker_map:
            speaker_info = ", ".join([f"{k} 是 {v}" for k, v in speaker_map.items()])

        language_hint = ""
        extra_instructions = ""
        if not is_chinese:
            language_hint = f"### 源文本语言：{language.upper()} (Source Language: {language.upper()})"
            extra_instructions = (
                "### 强制性双语要求 (Bilingual Requirement):\n"
                "**格式**：每一段原文后面紧跟该段的中文翻译，交替排列。严禁把所有原文放一起再集中翻译。\n"
                "示例：\n"
                "```\n"
                "English paragraph 1...\n"
                "\n"
                "第1段中文翻译...\n"
                "\n"
                "English paragraph 2...\n"
                "\n"
                "第2段中文翻译...\n"
                "```\n"
                "**规则**：\n"
                "1. 先输出校正后的原文段落，紧接着输出该段的简体中文翻译。\n"
                "2. 严禁添加 '[Original]'、'[翻译]' 等标签，直接输出文本。\n"
                "3. 严禁遗漏任何段落的翻译。每一段原文都必须有对应翻译。"
            )

        # Pre-fill everything except {text} which varies per chunk
        prompt_base = prompt_template \
            .replace("{title}", title or "未知标题") \
            .replace("{author}", author or "未知UP主") \
            .replace("{speaker_map}", speaker_info) \
            .replace("{language_hint}", language_hint) \
            .replace("{extra_instructions}", extra_instructions)

        corrected_chunks = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))

            try:
                prompt = prompt_base.replace("{text}", chunk)
                
                # Qwen3.5 non-thinking mode: use Ollama's think=False
                # + official HF recommended sampling params for non-thinking
                # For bilingual chunks, cap max_tokens to ~3x chunk size to avoid runaway repetition
                max_tok = len(chunk) * 3 + 500 if not is_chinese else 8192
                batch_corrected = self.generate(
                    prompt,
                    temperature=0.25,
                    max_tokens=min(max_tok, 8192),
                    top_p=0.8,
                    top_k=20,
                    presence_penalty=1.5,
                    repetition_penalty=1.0,
                    think=False,
                    num_ctx=getattr(self, "correction_num_ctx", 12288),
                )
                
                if not is_chinese:
                    # Clean up bilingual artifacts: remove "第N段中文翻译：" labels
                    batch_corrected = re.sub(r'第\s*\d+\s*段中文翻译[：:]\s*\n?', '', batch_corrected)

                # Detect runaway length (>4x input for non-Chinese, >2.5x for Chinese)
                max_ratio = 4.0 if not is_chinese else 2.5
                if len(batch_corrected) > len(chunk) * max_ratio:
                    logger.warning(f"  Chunk {i+1} output too long ({len(batch_corrected)} vs {len(chunk)} chars), likely repetition. Keeping original.")
                    corrected_chunks.append(chunk)
                    continue

                # Detect internal repetition within this chunk
                if _detect_repetition(batch_corrected):
                    logger.warning(f"  Chunk {i+1} has internal repetition, retrying with higher penalty...")
                    # Retry once with stronger repetition penalty
                    retry_corrected = self.generate(
                        prompt,
                        temperature=0.2,
                        max_tokens=min(max_tok, 8192) if not is_chinese else 8192,
                        top_p=0.8,
                        top_k=20,
                        presence_penalty=1.5,
                        repetition_penalty=1.1,
                        think=False,
                        num_ctx=getattr(self, "correction_num_ctx", 12288),
                    )
                    if not _detect_repetition(retry_corrected) and len(retry_corrected) >= len(chunk) * 0.3:
                        batch_corrected = retry_corrected
                        logger.info(f"  Chunk {i+1} retry succeeded.")
                    else:
                        logger.warning(f"  Chunk {i+1} retry still has repetition. Keeping original.")
                        corrected_chunks.append(chunk)
                        continue

                # Fallback if LLM fails or returns garbage
                min_ratio = 0.3
                if len(batch_corrected) < len(chunk) * min_ratio:
                    logger.warning(f"  Chunk {i+1} correction seems too short ({len(batch_corrected)} chars), keeping original.")
                    corrected_chunks.append(chunk)
                else:
                    corrected_chunks.append(batch_corrected)
            except Exception as e:
                logger.error(f"  Chunk {i+1} correction failed: {e}")
                corrected_chunks.append(chunk)

        result = "\n\n".join(corrected_chunks)

        # Collapse loops within a paragraph first, then drop duplicate blocks
        # across the whole text (order matters: collapsing internal loops can
        # turn two near-identical blocks into exact duplicates).
        result = _collapse_internal_loops(result)
        result = _deduplicate_paragraphs(result)

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
            source_text = text
            hierarchy_threshold = getattr(self, "hierarchical_summary_chars", 18000)
            summary_chunk_chars = getattr(self, "summary_chunk_chars", 12000)
            if len(text) > hierarchy_threshold:
                chunks = self._split_summary_chunks(text, summary_chunk_chars)
                logger.info(
                    "Long transcript: generating %d section summaries before synthesis",
                    len(chunks),
                )
                section_summaries = []
                for index, chunk in enumerate(chunks, 1):
                    section_prompt = (
                        f"请提炼视频《{title}》第 {index}/{len(chunks)} 部分。"
                        "必须忠于原文，保留关键人物、术语、数字、案例、论据和结论；"
                        "不要补充原文没有的信息。输出简体中文的结构化要点，供最终总结使用。\n\n"
                        f"{chunk}"
                    )
                    section_summaries.append(
                        self.generate(
                            section_prompt,
                            temperature=0.25,
                            max_tokens=1600,
                            top_p=0.8,
                            top_k=20,
                            presence_penalty=1.5,
                            repetition_penalty=1.0,
                            think=False,
                            num_ctx=getattr(self, "correction_num_ctx", 12288),
                        )
                    )
                source_text = "\n\n".join(
                    f"### 第 {i} 部分提炼\n{summary}"
                    for i, summary in enumerate(section_summaries, 1)
                )
            text = source_text
            prompt_template = self._load_prompt("summarize")
            prompt = prompt_template.replace("{text}", text).replace("{title}", title or "未知标题").replace("{author}", author or "未知UP主")
            
            # Qwen3.5 non-thinking mode with official HF sampling params
            result = self.generate(
                prompt,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.8,
                top_k=20,
                presence_penalty=1.5,
                repetition_penalty=1.0,
                think=False,
                num_ctx=getattr(self, "summary_num_ctx", 32768),
            )
            
            if len(result) < 80:
                raise RuntimeError(f"Summary too short ({len(result)} chars)")
            
            return result
        except requests.Timeout:
            logger.error("    [Error] Ollama summarization timed out.")
            raise RuntimeError("Ollama summarization timed out")
        except Exception as e:
            logger.error(f"    [Error] Summarization failed: {e}")
            raise

    @staticmethod
    def _split_summary_chunks(text: str, target_chars: int) -> list[str]:
        """Split long text on paragraph boundaries for hierarchical summaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for paragraph in paragraphs:
            if current and current_len + len(paragraph) + 2 > target_chars:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(paragraph)
            current_len += len(paragraph) + 2
        if current:
            chunks.append("\n\n".join(current))
        return chunks

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
        corrected = self.correct_text_batched(text, progress_callback=progress_callback)
        
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
        response = client.generate("你好，请用一句话介绍自己。", max_tokens=100, think=False)
        logger.info(f"Ollama test successful: {response[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Ollama test failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
