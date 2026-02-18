"""Qwen3 ASR client for local speech-to-text transcription.

Uses Alibaba's Qwen3-ASR-1.7B model for high-quality multilingual speech recognition.
Supports automatic audio chunking for long files (12 GB VRAM safe).
Supports speaker diarization via pyannote + Qwen3-ASR hybrid pipeline.
"""

from __future__ import annotations

import gc
import re
import os
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── HuggingFace cache on E: drive (avoid C: space issues) ───────────
_HF_HOME = "E:/ai_models/huggingface"
_TEMP_DIR = "E:/temp"
Path(_TEMP_DIR).mkdir(parents=True, exist_ok=True)

if os.path.isdir(_HF_HOME):
    os.environ.setdefault("HF_HOME", _HF_HOME)

# Force TEMP to E: drive as C: is often full
os.environ["TEMP"] = _TEMP_DIR
os.environ["TMP"] = _TEMP_DIR

# Max audio chunk duration (seconds) - 3 min fits in 12 GB VRAM
MAX_CHUNK_SECONDS = 180
# Max speaker turn duration for diarized transcription (seconds)
MAX_TURN_SECONDS = 120
# Min gap (seconds) to treat consecutive same-speaker segments as one turn
MERGE_GAP_SECONDS = 3.0
# Duration (seconds) of sample used for quick speaker-count check
DIARIZE_PROBE_SECONDS = 120
# Batch inference: max turns per batch and max total audio seconds per batch
BATCH_SIZE = 8
BATCH_MAX_TOTAL_SECONDS = 240


@dataclass
class SpeakerTurn:
    """A contiguous speech segment from one speaker."""
    speaker: str
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class Qwen3ASRClient:
    """Local Qwen3 ASR speech-to-text client with optional speaker diarization."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        language: Optional[str] = None,
        convert_to_simplified: bool = True,
        ffmpeg_location: Optional[str] = None,
        diarize: bool = False,
        hf_token: Optional[str] = None,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ):
        """Initialize Qwen3 ASR client.
        
        Args:
            model_name: HuggingFace model ID.
            device: Device to use (e.g., "cuda:0", "cpu").
            language: Language name (e.g., "Chinese", "English") or None for auto.
            convert_to_simplified: If True, convert Traditional Chinese to Simplified.
            ffmpeg_location: Optional path to ffmpeg binary directory.
            diarize: If True, enable speaker diarization via pyannote.
            hf_token: HuggingFace token (required for pyannote diarization model).
            min_speakers: Minimum number of speakers for diarization.
            max_speakers: Maximum number of speakers for diarization.
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.convert_to_simplified = convert_to_simplified
        self.diarize = diarize
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._model = None
        self._diarize_pipeline = None
        self._converter = None
        
        # Add ffmpeg to PATH if provided
        if ffmpeg_location and os.path.isdir(ffmpeg_location):
            current_path = os.environ.get("PATH", "")
            if ffmpeg_location not in current_path:
                os.environ["PATH"] = ffmpeg_location + os.pathsep + current_path
                logger.info(f"Qwen3: Added ffmpeg to PATH: {ffmpeg_location}")
        
        logger.info(f"Qwen3: Initialized (model='{model_name}', diarize={diarize})")

    # ─── Model Management ────────────────────────────────────────────

    @property
    def model(self):
        """Lazy-load the Qwen3-ASR model."""
        if self._model is None:
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
                if torch.cuda.is_available():
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    logger.info(f"CUDA Memory before loading Qwen3: Free={free_mem/1024**2:.0f}MB, Total={total_mem/1024**2:.0f}MB")
            except ImportError:
                raise RuntimeError(
                    "qwen-asr not installed. Please install it:\n"
                    "  pip install qwen-asr"
                )
            
            logger.info(f"Loading Qwen3-ASR model '{self.model_name}'...")
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map=self.device,
                max_inference_batch_size=BATCH_SIZE,
                max_new_tokens=4096,          # Support long chunks
            )
            logger.info("Qwen3-ASR model loaded.")
        
        return self._model

    def _ensure_diarize_pipeline(self):
        """Load pyannote diarization pipeline."""
        if self._diarize_pipeline is None:
            if not self.hf_token:
                raise ValueError(
                    "HuggingFace token required for speaker diarization. "
                    "Set hf_token in config.yaml under whisperx section."
                )
            
            logger.info("Loading pyannote diarization pipeline...")
            try:
                from whisperx.diarize import DiarizationPipeline
            except (ImportError, AttributeError):
                from pyannote.audio import Pipeline as PyannotePipeline
                # Fallback: create our own wrapper
                class DiarizationPipeline:
                    def __init__(self, use_auth_token, device):
                        self.pipeline = PyannotePipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=use_auth_token,
                        ).to(device)
                    
                    def __call__(self, audio, min_speakers=None, max_speakers=None):
                        return self.pipeline(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            
            self._diarize_pipeline = DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device,
            )
            logger.info("Diarization pipeline loaded.")

    def _unload_diarize_pipeline(self):
        """Unload diarization pipeline to free VRAM for ASR."""
        if self._diarize_pipeline is not None:
            logger.info("Unloading diarization pipeline and clearing CUDA cache...")
            self._diarize_pipeline = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    free_mem, _ = torch.cuda.mem_get_info()
                    logger.info(f"  ✓ Diarization pipeline unloaded. Free VRAM: {free_mem/1024**2:.0f}MB")
            except ImportError:
                pass

    def unload_model(self):
        """Unload all models from memory to save VRAM."""
        if self._model is not None:
            logger.info(f"Unloading Qwen3-ASR model...")
            self._model = None
        
        self._unload_diarize_pipeline()
        
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  ✓ CUDA cache cleared")
        except ImportError:
            pass

    # ─── Audio Processing ────────────────────────────────────────────

    def _load_audio_mono_16k(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Load audio as mono 16kHz numpy array."""
        import librosa
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        return y, sr

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        import librosa
        return librosa.get_duration(path=str(audio_path))

    def _split_audio_chunks(self, audio_path: Path) -> list[str]:
        """Split audio into chunks if it exceeds MAX_CHUNK_SECONDS.
        
        Returns list of audio file paths (original if short enough, temp files if chunked).
        """
        duration = self._get_audio_duration(audio_path)
        
        if duration <= MAX_CHUNK_SECONDS + 30:  # Allow 30s buffer before splitting
            logger.info(f"  Audio duration: {duration:.0f}s (no chunking needed)")
            return [str(audio_path)]
        
        import soundfile as sf
        
        logger.info(f"  Audio duration: {duration:.0f}s -> splitting into {MAX_CHUNK_SECONDS}s chunks...")
        y, sr = self._load_audio_mono_16k(audio_path)
        
        chunk_samples = MAX_CHUNK_SECONDS * sr
        chunks = []
        
        for i in range(0, len(y), chunk_samples):
            chunk = y[i:i + chunk_samples]
            chunk_path = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix=f"qwen_chunk_{i // chunk_samples}_",
                delete=False, dir=str(audio_path.parent),
            )
            sf.write(chunk_path.name, chunk, sr)
            chunks.append(chunk_path.name)
            chunk_dur = len(chunk) / sr
            logger.info(f"    chunk {len(chunks)}: {chunk_dur:.0f}s")
        
        logger.info(f"  Split into {len(chunks)} chunks")
        return chunks

    def _save_audio_segment(self, y: np.ndarray, sr: int, start: float, end: float, parent_dir: str) -> str:
        """Cut and save an audio segment to a temp wav file."""
        import soundfile as sf
        
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="spk_seg_", delete=False, dir=parent_dir,
        )
        sf.write(tmp.name, segment, sr)
        return tmp.name

    def _slice_audio(self, y: np.ndarray, sr: int, start: float, end: float) -> tuple:
        """Slice audio array in memory. Returns (np.ndarray, sr) tuple for model."""
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        return (y[start_sample:end_sample], sr)

    # ─── Speaker Diarization ─────────────────────────────────────────

    def _parse_diarize_result(self, diarize_result) -> list[SpeakerTurn]:
        """Parse diarization output into SpeakerTurn list."""
        raw_turns: list[SpeakerTurn] = []
        if hasattr(diarize_result, 'iterrows'):
            for _, row in diarize_result.iterrows():
                raw_turns.append(SpeakerTurn(
                    speaker=str(row.get("speaker", "SPEAKER_00")),
                    start=float(row.get("start", 0)),
                    end=float(row.get("end", 0)),
                ))
        elif hasattr(diarize_result, 'itertracks'):
            for turn, _, speaker in diarize_result.itertracks(yield_label=True):
                raw_turns.append(SpeakerTurn(
                    speaker=str(speaker),
                    start=turn.start,
                    end=turn.end,
                ))
        else:
            logger.warning(f"Unknown diarization output type: {type(diarize_result)}")
        return raw_turns

    def _quick_speaker_count(self, audio_path: Path) -> tuple[int, bool]:
        """Fast speaker-count probe on the first DIARIZE_PROBE_SECONDS of audio.
        
        Returns (num_speakers, is_full_coverage) where is_full_coverage is True
        when the probe covered the entire audio (so result can be reused).
        """
        import whisperx
        
        duration = self._get_audio_duration(audio_path)
        probe_dur = min(duration, DIARIZE_PROBE_SECONDS)
        is_full = (probe_dur >= duration)
        
        if not is_full:
            y, sr = self._load_audio_mono_16k(audio_path)
            probe_samples = int(probe_dur * sr)
            probe_audio = y[:probe_samples].copy()
            if probe_audio.dtype != np.float32:
                probe_audio = probe_audio.astype(np.float32)
        else:
            probe_audio = whisperx.load_audio(str(audio_path))
        
        self._ensure_diarize_pipeline()
        logger.info(f"Quick speaker probe ({probe_dur:.0f}s sample)...")
        
        result = self._diarize_pipeline(
            probe_audio,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        
        raw = self._parse_diarize_result(result)
        speakers = set(t.speaker for t in raw)
        n = len(speakers)
        logger.info(f"  Probe found {n} speaker(s) in {len(raw)} segments")
        
        # Cache probe result when it covers the full audio to avoid re-running
        if is_full:
            self._cached_probe_turns = raw
        else:
            self._cached_probe_turns = None
        
        return n, is_full

    def _run_diarization(self, audio_path: Path) -> list[SpeakerTurn]:
        """Run pyannote speaker diarization.
        
        Reuses cached probe result when available (probe covered full audio).
        Returns list of merged SpeakerTurn with (speaker, start, end).
        """
        # Reuse cached probe if it covered the full audio
        cached = getattr(self, '_cached_probe_turns', None)
        if cached is not None:
            logger.info(f"Reusing probe diarization ({len(cached)} raw segments)")
            raw_turns = cached
            self._cached_probe_turns = None
        else:
            import whisperx
            audio = whisperx.load_audio(str(audio_path))
            self._ensure_diarize_pipeline()
            
            logger.info("Running full speaker diarization...")
            diarize_result = self._diarize_pipeline(
                audio,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            raw_turns = self._parse_diarize_result(diarize_result)
        
        logger.info(f"Diarization found {len(raw_turns)} raw segments")
        
        merged = self._merge_speaker_turns(raw_turns)
        speakers = set(t.speaker for t in merged)
        logger.info(f"  {len(speakers)} speakers detected, {len(merged)} merged turns")
        
        return merged

    def _merge_speaker_turns(self, turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
        """Merge consecutive same-speaker turns with small gaps.
        
        Also splits overly long turns at MAX_TURN_SECONDS.
        """
        if not turns:
            return []
        
        merged: list[SpeakerTurn] = [SpeakerTurn(
            speaker=turns[0].speaker,
            start=turns[0].start,
            end=turns[0].end,
        )]
        
        for turn in turns[1:]:
            prev = merged[-1]
            gap = turn.start - prev.end
            
            # Merge if same speaker and gap is small and combined duration acceptable
            if (turn.speaker == prev.speaker 
                    and gap <= MERGE_GAP_SECONDS
                    and (turn.end - prev.start) <= MAX_TURN_SECONDS):
                prev.end = turn.end  # Extend the previous turn
            else:
                merged.append(SpeakerTurn(
                    speaker=turn.speaker,
                    start=turn.start,
                    end=turn.end,
                ))
        
        return merged

    # ─── Transcription ───────────────────────────────────────────────

    def transcribe(self, audio_path: Path, initial_prompt: str = "") -> str:
        """Transcribe audio file to text (no diarization).
        
        Handles automatic chunking for long audio.
        """
        import torch
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing with Qwen3: {audio_path.name}...")
        
        chunk_paths = self._split_audio_chunks(audio_path)
        is_chunked = len(chunk_paths) > 1
        
        all_text = []
        
        for i, chunk_path in enumerate(chunk_paths):
            if is_chunked:
                logger.info(f"  Transcribing chunk {i + 1}/{len(chunk_paths)}...")
            
            results = self.model.transcribe(
                audio=chunk_path,
                language=self.language,
            )
            
            text = results[0].text.strip() if results else ""
            all_text.append(text)
            
            if is_chunked:
                logger.info(f"    -> {len(text)} chars")
                gc.collect()
                torch.cuda.empty_cache()
        
        # Clean up temp chunk files
        if is_chunked:
            for cp in chunk_paths:
                if cp != str(audio_path):
                    try:
                        os.unlink(cp)
                    except OSError:
                        pass
        
        full_text = "".join(all_text)
        
        if self.convert_to_simplified:
            full_text = self._to_simplified(full_text)
        
        logger.info(f"  ✓ Transcription complete: {len(full_text)} chars")
        return full_text

    def transcribe_with_diarization(self, audio_path: Path, initial_prompt: str = "") -> list[SpeakerTurn]:
        """Transcribe audio with speaker diarization.
        
        Flow:
          1. Quick probe first DIARIZE_PROBE_SECONDS to count speakers.
             If only 1 → skip diarization, use plain chunked transcription.
          2. Full diarization → unload → Qwen3-ASR per turn (in-memory).
        
        Returns:
            List of SpeakerTurn with text filled in.
        """
        import torch
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing with diarization: {audio_path.name}...")
        
        # ── Optimization D: quick single-speaker detection ────────────
        duration = self._get_audio_duration(audio_path)
        n_speakers, probe_is_full = self._quick_speaker_count(audio_path)
        
        if n_speakers <= 1:
            logger.info(
                f"  Single speaker detected – skipping full diarization, "
                f"using chunked mode ({duration:.0f}s audio)"
            )
            self._unload_diarize_pipeline()
            text = self.transcribe(audio_path, initial_prompt=initial_prompt)
            return [SpeakerTurn(speaker="SPEAKER_00", start=0, end=duration, text=text)]
        
        # ── Multi-speaker path: full diarization ─────────────────────
        # We already have the pipeline loaded from the probe; run on full audio
        turns = self._run_diarization(audio_path)
        
        if not turns:
            logger.warning("No speaker turns detected, falling back to plain transcription")
            self._unload_diarize_pipeline()
            text = self.transcribe(audio_path)
            return [SpeakerTurn(speaker="SPEAKER_00", start=0, end=0, text=text)]
        
        # Unload diarization to free VRAM for Qwen3-ASR
        self._unload_diarize_pipeline()
        
        # ── Optimization B+C: batch in-memory slices ──────────────────
        y, sr = self._load_audio_mono_16k(audio_path)
        
        # Build batches of turns (respecting BATCH_SIZE and BATCH_MAX_TOTAL_SECONDS)
        batches: list[list[int]] = []  # list of lists of turn indices
        cur_batch: list[int] = []
        cur_secs = 0.0
        
        for idx, turn in enumerate(turns):
            if cur_batch and (len(cur_batch) >= BATCH_SIZE or cur_secs + turn.duration > BATCH_MAX_TOTAL_SECONDS):
                batches.append(cur_batch)
                cur_batch = []
                cur_secs = 0.0
            cur_batch.append(idx)
            cur_secs += turn.duration
        if cur_batch:
            batches.append(cur_batch)
        
        logger.info(f"  Processing {len(turns)} turns in {len(batches)} batches")
        
        for batch_i, batch_indices in enumerate(batches):
            batch_turns = [turns[i] for i in batch_indices]
            audio_segments = [self._slice_audio(y, sr, t.start, t.end) for t in batch_turns]
            
            # Log batch info
            first, last = batch_turns[0], batch_turns[-1]
            logger.info(
                f"  Batch {batch_i+1}/{len(batches)}: {len(batch_indices)} turns "
                f"({first.start:.0f}-{last.end:.0f}s)"
            )
            
            try:
                results = self.model.transcribe(
                    audio=audio_segments,
                    language=self.language,
                )
                
                for j, turn in enumerate(batch_turns):
                    text = results[j].text.strip() if j < len(results) and results[j] else ""
                    if self.convert_to_simplified:
                        text = self._to_simplified(text)
                    turn.text = text
                    logger.debug(
                        f"    [{batch_indices[j]+1}/{len(turns)}] {turn.speaker} "
                        f"({turn.start:.1f}-{turn.end:.1f}s) -> {len(text)} chars"
                    )
            except Exception as e:
                logger.error(f"    ✗ Batch error: {e}. Falling back to sequential.")
                # Fallback: transcribe individually
                for j, turn in enumerate(batch_turns):
                    try:
                        seg = audio_segments[j]
                        res = self.model.transcribe(audio=seg, language=self.language)
                        text = res[0].text.strip() if res else ""
                        if self.convert_to_simplified:
                            text = self._to_simplified(text)
                        turn.text = text
                        logger.debug(
                            f"    [{batch_indices[j]+1}/{len(turns)}] {turn.speaker} "
                            f"({turn.start:.1f}-{turn.end:.1f}s) -> {len(text)} chars"
                        )
                    except Exception as e2:
                        logger.error(f"    ✗ Turn {batch_indices[j]+1} error: {e2}")
                        turn.text = ""
        
        # ── Optimization A: single cleanup at end, not per-turn ──────
        gc.collect()
        torch.cuda.empty_cache()
        
        # Filter out empty turns
        turns = [t for t in turns if t.text.strip()]
        
        total_chars = sum(len(t.text) for t in turns)
        logger.info(f"  ✓ Diarized transcription complete: {total_chars} chars, {len(turns)} turns")
        
        return turns

    def _detect_language_heuristic(self, text: str) -> str:
        """Heuristic to detect language from text if not provided by engine."""
        if not text:
            return "zh"
        
        # Count Chinese characters (CJK Unified Ideographs)
        cjk_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        # Count Japanese characters (Hiragana and Katakana)
        jp_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        # Total non-whitespace chars
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return "zh"
            
        cjk_ratio = cjk_count / total_chars
        jp_ratio = jp_count / total_chars
        
        if jp_ratio > 0.2:
            return "ja"
        if cjk_ratio > 0.3:
            return "zh"
        
        # Default to "en" if no CJK/JP found but text exists
        return "en"

    def transcribe_to_markdown(
        self,
        audio_path: Path,
        title: str = "",
        author: str = "",
        initial_prompt: str = ""
    ) -> tuple[str, str]:
        """Full transcription flow: Transcribe -> Format as MD.
        
        Args:
            audio_path: Path to audio file
            title: Video title for header
            author: UP主 name
            initial_prompt: Optional initial prompt for transcription style/punctuation
            
        Returns:
            Tuple of (Markdown text, language_code)
        """
        if self.diarize:
            turns = self.transcribe_with_diarization(audio_path, initial_prompt=initial_prompt)
            markdown = self._format_turns_to_markdown(turns, title=title, author=author)
            
            # Use heuristic on the first few turns if self.language is None
            detected_lang = self.language
            if not detected_lang:
                sample_text = " ".join([t.text for t in turns[:5]])
                detected_lang = self._detect_language_heuristic(sample_text)
            
            return markdown, detected_lang
        else:
            text = self.transcribe(audio_path, initial_prompt=initial_prompt)
            markdown = self._format_text_to_markdown(text, title=title, author=author)
            
            detected_lang = self.language
            if not detected_lang:
                detected_lang = self._detect_language_heuristic(text)
                
            return markdown, detected_lang

    def _format_turns_to_markdown(self, turns: list[SpeakerTurn], title: str = "", author: str = "") -> str:
        """Internal helper to format diarized turns to MD."""
        lines = [f"# {title}", ""]
        if author:
            lines.append(f"**UP主**: {author}")
            lines.append("")
        lines.append("---")
        lines.append("")
        
        # Count unique speakers and their contribution
        speaker_stats = {}  # speaker -> total_chars
        for turn in turns:
            txt = turn.text.strip()
            if txt:
                speaker_stats[turn.speaker] = speaker_stats.get(turn.speaker, 0) + len(txt)
        
        total_chars = sum(speaker_stats.values())
        
        # Determine significant speakers (at least 5% of content or 100 chars)
        significant_speakers = {
            spk for spk, count in speaker_stats.items() 
            if (total_chars > 0 and count / total_chars > 0.05) or (count > 100)
        }
        
        # Find the most prominent speaker (fallback for insignificant speakers)
        most_prominent_spk = max(speaker_stats.items(), key=lambda x: x[1])[0] if speaker_stats else None
        
        # If no significant speakers, use the most prominent one
        if not significant_speakers and most_prominent_spk:
            significant_speakers = {most_prominent_spk}
            
        num_significant = len(significant_speakers)
        
        if num_significant <= 1:
            # Single speaker: Just output text without labels
            full_text = "".join(t.text for t in turns)
            for para in self._split_into_paragraphs(full_text):
                lines.append(para)
                lines.append("")
        else:
            # Multi-speaker: Only map significant speakers
            speaker_map = self._build_speaker_map(significant_speakers)
            current_speaker = None
            current_texts = []
            
            for turn in turns:
                speaker = turn.speaker
                # Map insignificant speakers to current or most prominent
                if speaker not in significant_speakers:
                    speaker = current_speaker if current_speaker else most_prominent_spk
                
                if speaker != current_speaker:
                    if current_texts and current_speaker is not None:
                        label = speaker_map.get(current_speaker, current_speaker)
                        combined = "".join(current_texts)
                        for para in self._split_into_paragraphs(combined):
                            lines.append(f"**{label}** {para}")
                            lines.append("")
                    current_speaker = speaker
                    current_texts = [turn.text]
                else:
                    current_texts.append(turn.text)
            
            if current_texts and current_speaker is not None:
                label = speaker_map.get(current_speaker, current_speaker)
                combined = "".join(current_texts)
                for para in self._split_into_paragraphs(combined):
                    lines.append(f"**{label}** {para}")
                    lines.append("")
        
        return "\n".join(lines)

    def _format_text_to_markdown(self, text: str, title: str = "", author: str = "") -> str:
        """Internal helper to format plain text to MD."""
        lines = [f"# {title}", ""]
        if author:
            lines.append(f"**UP主**: {author}")
            lines.append("")
        lines.append("---")
        lines.append("")
        for para in self._split_into_paragraphs(text):
            lines.append(para)
            lines.append("")
        return "\n".join(lines)

    # ─── Utilities ───────────────────────────────────────────────────

    def _build_speaker_map(self, speakers: set[str]) -> dict[str, str]:
        """Map raw speaker IDs to readable labels like [说话人 A]."""
        # Maintain consistency with WhisperX format for LLM identification
        sorted_speakers = sorted(speakers)
        mapping = {}
        for i, spk in enumerate(sorted_speakers):
            char_tag = chr(65 + i) if i < 26 else str(i + 1)
            mapping[spk] = f"[说话人 {char_tag}]"
        return mapping

    def _to_simplified(self, text: str) -> str:
        """Convert Traditional Chinese to Simplified Chinese."""
        if self._converter is None:
            try:
                from opencc import OpenCC
                self._converter = OpenCC('t2s')
            except ImportError:
                logger.warning("opencc not installed, skipping conversion")
                return text
        return self._converter.convert(text)

    def _split_into_paragraphs(self, text: str, target_length: int = 500) -> list[str]:
        """Split text into paragraphs for readability."""
        if not text:
            return []
        
        sentence_endings = "。！？；.!?;"
        paragraphs = []
        current = []
        current_len = 0
        
        start = 0
        for i, char in enumerate(text):
            if char in sentence_endings:
                sentence = text[start:i+1].strip()
                if sentence:
                    if current_len + len(sentence) > target_length and current:
                        paragraphs.append("".join(current))
                        current = []
                        current_len = 0
                    current.append(sentence)
                    current_len += len(sentence)
                start = i + 1
        
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                current.append(remaining)
        
        if current:
            paragraphs.append("".join(current))
            
        return paragraphs if paragraphs else [text]
