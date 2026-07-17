"""Qwen3 ASR client for local speech-to-text transcription.

Uses Alibaba's Qwen3-ASR-1.7B model for high-quality multilingual speech recognition.
Supports automatic audio chunking for long files (12 GB VRAM safe).
Supports speaker diarization via pyannote + Qwen3-ASR hybrid pipeline.
"""

from __future__ import annotations

import gc
import hashlib
import json
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
_MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "BILIBILI_MODEL_CACHE",
        Path.home() / ".cache" / "bilibili_summarizer" / "ai_models",
    )
)
_HF_HOME = _MODEL_CACHE_ROOT / "huggingface"
_TEMP_DIR = Path(
    os.environ.get(
        "BILIBILI_TEMP",
        Path.home() / ".cache" / "bilibili_summarizer" / "temp",
    )
)
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

if _HF_HOME.is_dir():
    os.environ.setdefault("HF_HOME", str(_HF_HOME))

# Force TEMP to E: drive as C: is often full
os.environ["TEMP"] = str(_TEMP_DIR)
os.environ["TMP"] = str(_TEMP_DIR)

def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to default.

    Lets the scheduled runner (scripts/run_and_hibernate.py) shrink the ASR
    forward-pass size on a retry. After a sleep/hibernate resume the GPU can
    fall into a degraded state where a full-size 180s chunk stalls *inside* a
    single forward pass for tens of minutes (max_time can't interrupt that — it
    is only checked between generated tokens). Re-running in a fresh process
    with smaller chunks keeps each forward small enough to fit/finish.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


# Max audio chunk duration (seconds) - 3 min fits in 12 GB VRAM. Overridable via
# ASR_MAX_CHUNK_SECONDS so a degraded-GPU retry can use smaller chunks.
MAX_CHUNK_SECONDS = _env_int("ASR_MAX_CHUNK_SECONDS", 180)
# Generated-token cap per chunk. Generous so real long speech is never truncated
# (180s of dense Chinese ≈ 1000-1200 tokens). Runaway generation is bounded by
# GENERATE_MAX_TIME_SECONDS (wall-clock), not by this cap.
MAX_NEW_TOKENS = 2048
# Hard wall-clock cap on a single model.generate() call. HuggingFace checks this
# between token steps, so it genuinely interrupts a runaway/repetition loop that
# would otherwise generate to MAX_NEW_TOKENS while VRAM-thrashing (the root cause
# of the 27-39 min hangs). A normal batch finishes in ~50s.
GENERATE_MAX_TIME_SECONDS = 240
# Max speaker turn duration for diarized transcription (seconds)
MAX_TURN_SECONDS = 120
# Min gap (seconds) to treat consecutive same-speaker segments as one turn
MERGE_GAP_SECONDS = 3.0
# Duration (seconds) of sample used for quick speaker-count check
DIARIZE_PROBE_SECONDS = 120
# Split long probes across the beginning, middle and end. A video can open with
# a solo introduction and become an interview later, so probing only the first
# two minutes can silently route multi-speaker content to the plain path.
DIARIZE_PROBE_POINTS = 3
# Max time to wait for the pyannote diarization pipeline to load before treating
# it as a hang. The load is cache-only under HF_HUB_OFFLINE, so this is a
# defensive backstop against a stalled network call, not a normal-path budget.
DIARIZE_LOAD_TIMEOUT_SECONDS = _env_int("ASR_DIARIZE_LOAD_TIMEOUT", 120)
# A speaker only "counts" in the probe if it speaks at least this many seconds
# total within the probe window. Filters out fleeting false positives (intro
# music, jingles, brief noise) that otherwise misroute single-narrator videos
# to the slow diarization path. Mirrors the "significant speaker" filter the
# markdown formatter already applies.
MIN_SPEAKER_PROBE_SECONDS = 5.0
# Forced aligner (~0.6B) gives char/word-level timestamps. Loaded alongside the
# ASR model when diarize is on, so multi-speaker videos can be transcribed in one
# fast full-audio pass and then labelled by overlaying diarization turns on the
# word timestamps — instead of re-transcribing every speaker turn separately
# (which exploded to hours on long, conversational videos). ~1.2 GB extra VRAM.
FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
# Batch inference: max chunks per GPU batch. 6 (not 8) leaves VRAM headroom on a
# 12 GB card so the KV cache during generation doesn't fill VRAM and thrash.
BATCH_SIZE = _env_int("ASR_BATCH_SIZE", 6)
BATCH_MAX_TOTAL_SECONDS = _env_int("ASR_BATCH_MAX_TOTAL_SECONDS", 240)
# A small overlap protects words that straddle a hard chunk boundary. It is
# used only by plain transcription; timestamped forced-alignment chunks remain
# non-overlapping so speaker labels stay on an unambiguous timeline.
CHUNK_OVERLAP_SECONDS = _env_int("ASR_CHUNK_OVERLAP_SECONDS", 2)
# A long chunk with no recognized text is almost never a valid result for these
# videos; treating it as success creates partial transcripts that never retry.
MIN_NONEMPTY_CHUNK_SECONDS = 30


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


@dataclass
class AlignItem:
    """A timestamped transcription unit (mirrors the model's ForcedAlignItem,
    exposing just the fields the speaker-labelling logic needs). Used so chunked
    transcription can carry absolute (offset) timestamps regardless of whether
    the model's own item type is mutable."""
    text: str
    start_time: float
    end_time: float


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

    @staticmethod
    def build_context(title: str = "", author: str = "") -> str:
        """Build a compact ASR context containing likely proper nouns."""
        parts = []
        if title:
            parts.append(f"视频标题：{title.strip()}")
        if author:
            parts.append(f"UP主：{author.strip()}")
        return "；".join(parts)

    @staticmethod
    def _merge_chunk_text(previous: str, current: str, max_overlap_chars: int = 240) -> str:
        """Merge adjacent ASR chunks by removing an exact suffix/prefix overlap."""
        if not previous:
            return current
        if not current:
            return previous
        limit = min(len(previous), len(current), max_overlap_chars)
        for size in range(limit, 3, -1):
            if previous[-size:] == current[:size]:
                return previous + current[size:]
        return previous + current

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
            # When diarization is enabled, also load the forced aligner so the
            # multi-speaker path can label a single fast transcription pass via
            # word timestamps (see FORCED_ALIGNER_MODEL).
            extra = {}
            if self.diarize:
                extra["forced_aligner"] = FORCED_ALIGNER_MODEL
                logger.info(f"  Loading forced aligner '{FORCED_ALIGNER_MODEL}' for timestamp labelling")
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map=self.device,
                max_inference_batch_size=BATCH_SIZE,
                max_new_tokens=MAX_NEW_TOKENS,
                **extra,
            )
            # Hard wall-clock cap on each generate() call — the real guard that
            # interrupts runaway repetition loops (checked between token steps).
            #
            # CRITICAL: Qwen3ASRForConditionalGeneration.generate() is only a shell
            # that delegates to self.thinker.generate(), passing through just
            # max_new_tokens/eos_token_id. The actual transformers generation loop
            # (and the MaxTimeCriteria check) runs against the THINKER's own
            # generation_config — NOT the outer model's. Setting max_time on the
            # outer config alone is silently ignored, which caused 49-min hangs.
            # So set it on the thinker; also set the outer one for good measure.
            try:
                hf_model = self._model.model
                configs = []
                thinker = getattr(hf_model, "thinker", None)
                if thinker is not None and getattr(thinker, "generation_config", None) is not None:
                    configs.append(thinker.generation_config)
                if getattr(hf_model, "generation_config", None) is not None:
                    configs.append(hf_model.generation_config)
                for gc in configs:
                    gc.max_time = GENERATE_MAX_TIME_SECONDS
                if configs:
                    logger.info(
                        f"  Set generation max_time={GENERATE_MAX_TIME_SECONDS}s on "
                        f"{len(configs)} config(s) incl. thinker (runaway guard)"
                    )
                else:
                    logger.warning("  No generation_config found — max_time guard NOT set")
            except AttributeError as e:
                logger.warning(f"  Could not set generation max_time: {e}")
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
            
            # Watchdog: the model load resolves files through huggingface_hub,
            # which can make a network call with no timeout of its own. A
            # transient stall there froze the whole pipeline indefinitely (no
            # error, no traceback). HF_HUB_OFFLINE (set in main.py) prevents the
            # network call entirely; this bounded thread is the defensive backstop
            # so a hang surfaces as a TimeoutError instead of freezing forever.
            # The TimeoutError propagates to the Step C group handler, which marks
            # the group as error and moves on rather than hanging the run.
            self._diarize_pipeline = self._call_with_timeout(
                DIARIZE_LOAD_TIMEOUT_SECONDS,
                lambda: DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device,
                ),
            )
            logger.info("Diarization pipeline loaded.")

    @staticmethod
    def _call_with_timeout(timeout_s: float, fn):
        """Run ``fn`` in a daemon thread and return its result, raising
        TimeoutError if it does not finish within ``timeout_s``.

        Used to bound a model load that can otherwise block forever on a network
        call. signal.alarm is unavailable on Windows, so a thread is the portable
        option. A timed-out thread is left as a daemon (it cannot be force-killed)
        and will be reaped when the process exits — acceptable because a hung load
        means this run is already failing.
        """
        import threading

        result: dict = {}

        def _run():
            try:
                result["value"] = fn()
            except BaseException as e:  # surface load errors to the caller
                result["error"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            raise TimeoutError(
                f"Diarization pipeline load exceeded {timeout_s:.0f}s "
                f"(likely a stalled HuggingFace network call; HF_HUB_OFFLINE should prevent this)"
            )
        if "error" in result:
            raise result["error"]
        return result["value"]

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

    def _unload_asr_model(self):
        """Unload Qwen3-ASR model to free VRAM for diarization."""
        if self._model is not None:
            logger.info("Unloading Qwen3-ASR model to free VRAM for diarization...")
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    free_mem, _ = torch.cuda.mem_get_info()
                    logger.info(f"  ✓ ASR model unloaded. Free VRAM: {free_mem/1024**2:.0f}MB")
            except ImportError:
                pass

    def unload_model(self):
        """Unload all models from memory to save VRAM."""
        self._unload_asr_model()
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
        overlap_samples = min(CHUNK_OVERLAP_SECONDS * sr, max(0, chunk_samples // 4))
        stride_samples = max(1, chunk_samples - overlap_samples)
        chunks = []
        
        for i in range(0, len(y), stride_samples):
            chunk = y[i:i + chunk_samples]
            chunk_path = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix=f"qwen_chunk_{i // chunk_samples}_",
                delete=False, dir=str(audio_path.parent),
            )
            sf.write(chunk_path.name, chunk, sr)
            chunks.append(chunk_path.name)
            chunk_dur = len(chunk) / sr
            logger.info(f"    chunk {len(chunks)}: {chunk_dur:.0f}s")
            if i + chunk_samples >= len(y):
                break
        
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
        """Fast speaker-count probe distributed across the audio.
        
        Returns (num_speakers, is_full_coverage) where is_full_coverage is True
        when the probe covered the entire audio (so result can be reused).
        """
        import whisperx
        
        duration = self._get_audio_duration(audio_path)
        probe_dur = min(duration, DIARIZE_PROBE_SECONDS)
        is_full = probe_dur >= duration
        
        if not is_full:
            y, sr = self._load_audio_mono_16k(audio_path)
            point_count = max(1, DIARIZE_PROBE_POINTS)
            segment_dur = probe_dur / point_count
            max_start = max(0.0, duration - segment_dur)
            starts = np.linspace(0.0, max_start, point_count)
            segments = [
                y[int(start * sr):int((start + segment_dur) * sr)]
                for start in starts
            ]
            probe_audio = np.concatenate(segments).copy()
            if probe_audio.dtype != np.float32:
                probe_audio = probe_audio.astype(np.float32)
        else:
            probe_audio = whisperx.load_audio(str(audio_path))
        
        self._ensure_diarize_pipeline()
        probe_label = "full audio" if is_full else f"{DIARIZE_PROBE_POINTS} distributed samples"
        logger.info(f"Quick speaker probe ({probe_dur:.0f}s, {probe_label})...")
        
        result = self._diarize_pipeline(
            probe_audio,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        
        raw = self._parse_diarize_result(result)
        # Count only speakers with enough cumulative speech in the probe window;
        # fleeting segments (intro music, brief noise) shouldn't trigger the slow
        # diarization path on what is really a single-narrator video.
        speaker_secs: dict[str, float] = {}
        for t in raw:
            speaker_secs[t.speaker] = speaker_secs.get(t.speaker, 0.0) + max(0.0, t.end - t.start)
        significant = [s for s, secs in speaker_secs.items() if secs >= MIN_SPEAKER_PROBE_SECONDS]
        # Always count at least one speaker so a very short clip isn't dropped.
        n = max(1, len(significant))
        total_speakers = len(speaker_secs)
        if total_speakers != n:
            logger.info(
                f"  Probe found {total_speakers} raw speaker(s) in {len(raw)} segments; "
                f"{n} significant (>= {MIN_SPEAKER_PROBE_SECONDS:.0f}s)"
            )
        else:
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
        """Merge consecutive same-speaker turns with small gaps, then split any
        turn exceeding MAX_TURN_SECONDS into equal-length chunks."""
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

        # Split any turn that exceeds MAX_TURN_SECONDS (e.g. a raw diarization
        # segment that pyannote emits as one long block).
        result: list[SpeakerTurn] = []
        for turn in merged:
            if turn.duration <= MAX_TURN_SECONDS:
                result.append(turn)
            else:
                pos = turn.start
                while pos < turn.end:
                    chunk_end = min(pos + MAX_TURN_SECONDS, turn.end)
                    result.append(SpeakerTurn(speaker=turn.speaker, start=pos, end=chunk_end))
                    pos = chunk_end

        return result

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

            # Runaway generation is bounded by generation_config.max_time set at
            # model load, so a stuck chunk returns (possibly partial) text rather
            # than hanging — no external timeout wrapper needed.
            results = self.model.transcribe(
                audio=chunk_path,
                context=initial_prompt,
                language=self.language,
            )

            text = results[0].text.strip() if results else ""
            if all_text:
                all_text[-1] = self._merge_chunk_text(all_text[-1], text)
            else:
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

    def transcribe_files_batch(
        self,
        audio_paths: list[Path],
        contexts: Optional[list[str]] = None,
    ) -> list[str]:
        """Transcribe multiple audio files by batching all chunks across files.

        Skips diarization for throughput. Processes up to BATCH_SIZE chunks per
        GPU call, spreading work across videos for full GPU utilization.

        Returns one plain-text string per input path (empty string on failure).
        """
        import torch

        audio_paths = [Path(p) for p in audio_paths]
        if not audio_paths:
            return []
        if contexts is None:
            contexts = [""] * len(audio_paths)
        if len(contexts) != len(audio_paths):
            raise ValueError("contexts must match audio_paths length")

        # Build flat list of (file_idx, chunk_idx, chunk_count, chunk_array, sr)
        # from all files.
        file_chunk_counts: list[int] = []
        flat_chunks: list[tuple[int, int, int, np.ndarray, int]] = []
        file_failures: set[int] = set()

        for file_idx, audio_path in enumerate(audio_paths):
            if not audio_path.exists():
                logger.error(f"  Audio not found: {audio_path.name}")
                file_chunk_counts.append(0)
                file_failures.add(file_idx)
                continue

            # Isolate per-file load failures: a single corrupt/truncated m4a
            # (e.g. a partial download) raises here, and must NOT abort the whole
            # batch and drag every healthy file in the group down with it. The
            # file gets 0 chunks → "" result, which the caller marks as error.
            try:
                y, sr = self._load_audio_mono_16k(audio_path)
            except Exception as e:
                logger.error(f"  ✗ Failed to load {audio_path.name}: {e!r} — skipping file")
                file_chunk_counts.append(0)
                file_failures.add(file_idx)
                continue
            duration = len(y) / sr
            chunk_samples = MAX_CHUNK_SECONDS * sr
            overlap_samples = min(CHUNK_OVERLAP_SECONDS * sr, max(0, chunk_samples // 4))
            stride_samples = max(1, chunk_samples - overlap_samples)

            if duration <= MAX_CHUNK_SECONDS + 30:
                flat_chunks.append((file_idx, 0, 1, y, sr))
                file_chunk_counts.append(1)
            else:
                chunks: list[np.ndarray] = []
                for start in range(0, len(y), stride_samples):
                    chunks.append(y[start:start + chunk_samples])
                    if start + chunk_samples >= len(y):
                        break
                n = len(chunks)
                for chunk_idx, chunk in enumerate(chunks):
                    flat_chunks.append((file_idx, chunk_idx, n, chunk, sr))
                file_chunk_counts.append(n)
                logger.info(f"  {audio_path.name}: {duration:.0f}s → {n} chunks")

        total_chunks = len(flat_chunks)

        # Group chunks into GPU batches bounded by BOTH the chunk count (BATCH_SIZE)
        # and the total audio seconds (BATCH_MAX_TOTAL_SECONDS). The seconds cap is
        # the load-bearing one on a 12 GB card: batching purely by count let 6 chunks
        # * 180s = ~1080s of audio land in a single prefill, overflowing VRAM into
        # shared RAM. That forward pass then slows by orders of magnitude — a
        # multi-hour stall that generation_config.max_time CANNOT interrupt, because
        # max_time is only checked between generated tokens, never inside one forward
        # pass. Capping audio-seconds per batch prevents the thrash. Mirrors the
        # diarized turn-batching path below.
        batches: list[list[tuple[int, int, int, np.ndarray, int]]] = []
        cur: list[tuple[int, int, int, np.ndarray, int]] = []
        cur_secs = 0.0
        for chunk in flat_chunks:
            chunk_secs = len(chunk[3]) / chunk[4]
            if cur and (len(cur) >= BATCH_SIZE or cur_secs + chunk_secs > BATCH_MAX_TOTAL_SECONDS):
                batches.append(cur)
                cur, cur_secs = [], 0.0
            cur.append(chunk)
            cur_secs += chunk_secs
        if cur:
            batches.append(cur)

        total_batches = len(batches)
        logger.info(
            f"  Batch transcription: {len(audio_paths)} files, "
            f"{total_chunks} chunks, {total_batches} GPU batches"
        )

        # Accumulate chunk texts indexed by file
        chunk_results: dict[int, list[str]] = {i: [] for i in range(len(audio_paths))}

        for b_idx, batch in enumerate(batches):
            audio_segs = [(arr, sr) for _, _, _, arr, sr in batch]
            b_num = b_idx + 1
            batch_secs = sum(len(arr) / sr for _, _, _, arr, sr in batch)
            logger.info(
                f"  GPU batch {b_num}/{total_batches}: {len(batch)} chunks, {batch_secs:.0f}s audio"
            )

            try:
                results = self.model.transcribe(
                    audio=audio_segs,
                    context=[contexts[file_idx] for file_idx, _, _, _, _ in batch],
                    language=self.language,
                )
            except Exception as e:
                logger.error(f"  ✗ GPU batch {b_num} failed: {e!r} — skipping batch")
                for file_idx, _, _, _, _ in batch:
                    file_failures.add(file_idx)
                    chunk_results[file_idx].append("")
                continue

            for j, (file_idx, chunk_idx, chunk_count, arr, sr) in enumerate(batch):
                text = results[j].text.strip() if j < len(results) and results[j] else ""
                chunk_secs = len(arr) / sr
                if (
                    chunk_count > 1
                    and chunk_secs >= MIN_NONEMPTY_CHUNK_SECONDS
                    and not text
                ):
                    logger.error(
                        f"  Empty ASR chunk {chunk_idx + 1}/{chunk_count} "
                        f"for {audio_paths[file_idx].name}; marking file incomplete"
                    )
                    file_failures.add(file_idx)
                if self.convert_to_simplified:
                    text = self._to_simplified(text)
                existing = chunk_results[file_idx]
                if existing:
                    existing[-1] = self._merge_chunk_text(existing[-1], text)
                else:
                    existing.append(text)

            gc.collect()
            torch.cuda.empty_cache()

        return [
            "" if i in file_failures else "".join(chunk_results.get(i, []))
            for i in range(len(audio_paths))
        ]

    def transcribe_files_smart(
        self, items: list[tuple[Path, str, str]]
    ) -> list[tuple[str, str]]:
        """Hybrid transcription with per-video speaker routing.

        For each (audio_path, title, author):
          1. Probe speaker count once (pyannote loaded a single time for the whole batch).
          2. Single-speaker videos → fast cross-video batched transcription (no labels).
          3. Multi-speaker videos → full diarization with [说话人 A/B] labels.

        Returns one (markdown, language) tuple per input, in the original order.
        Falls back to plain batch transcription when diarization is not configured.
        """
        items = [(Path(p), t, a) for p, t, a in items]
        n = len(items)
        if n == 0:
            return []

        # No diarization configured → everything goes through the fast plain path.
        if not self.diarize or not self.hf_token:
            contexts = [self.build_context(t, a) for _, t, a in items]
            texts = self.transcribe_files_batch(
                [p for p, _, _ in items],
                contexts=contexts,
            )
            return [
                (
                    self._format_text_to_markdown(txt, title=t, author=a),
                    self.language or self._detect_language_heuristic(txt),
                )
                for (p, t, a), txt in zip(items, texts)
            ]

        # ── Phase A: probe speaker counts (pyannote loaded once) ──────────
        self._unload_asr_model()  # free VRAM so pyannote runs alone
        speaker_counts: list[int] = []
        for path, _, _ in items:
            try:
                cnt, _ = self._quick_speaker_count(path)
            except Exception as e:
                logger.warning(f"  Speaker probe failed for {path.name}: {e!r}; assuming single")
                cnt = 1
            speaker_counts.append(cnt)
        self._unload_diarize_pipeline()

        single_idx = [i for i, c in enumerate(speaker_counts) if c <= 1]
        multi_idx = [i for i, c in enumerate(speaker_counts) if c > 1]
        logger.info(
            f"  Speaker routing: {len(single_idx)} single-speaker (fast batch), "
            f"{len(multi_idx)} multi-speaker (diarized)"
        )

        results: list[tuple[str, str]] = [("", "")] * n

        # ── Phase B: single-speaker → fast cross-video batch ──────────────
        if single_idx:
            texts = self.transcribe_files_batch(
                [items[i][0] for i in single_idx],
                contexts=[self.build_context(items[i][1], items[i][2]) for i in single_idx],
            )
            for k, i in enumerate(single_idx):
                _, title, author = items[i]
                txt = texts[k] if k < len(texts) else ""
                md = self._format_text_to_markdown(txt, title=title, author=author)
                lang = self.language or self._detect_language_heuristic(txt)
                results[i] = (md, lang)

        # ── Phase C: multi-speaker → transcribe once, then label by timestamps ─
        for i in multi_idx:
            path, title, author = items[i]
            try:
                turns = self._diarize_and_label(
                    path,
                    context=self.build_context(title, author),
                )
                md = self._format_turns_to_markdown(turns, title=title, author=author)
                sample = " ".join(t.text for t in turns[:5])
                lang = self.language or self._detect_language_heuristic(sample)
                results[i] = (md, lang)
            except Exception as e:
                logger.error(f"  Diarized transcription failed for {path.name}: {e!r}")
                results[i] = ("", "")

        return results

    def _diarize_and_label(
        self,
        audio_path: Path,
        context: str = "",
    ) -> list[SpeakerTurn]:
        """Multi-speaker transcription via "transcribe once, then label".

        Prefers the fast forced-aligner path (one full-audio pass + timestamp
        overlay). Falls back to the per-turn path when the aligner is missing or
        produces no usable alignment.
        """
        has_aligner = getattr(self.model, "forced_aligner", None) is not None
        if has_aligner:
            try:
                turns = self._transcribe_then_label(Path(audio_path), context=context)
                if turns:
                    return turns
                logger.warning("  Aligner labelling produced no turns; falling back to per-turn")
            except Exception as e:
                logger.warning(f"  Aligner labelling failed ({e}); falling back to per-turn")
        return self.transcribe_with_diarization(Path(audio_path), initial_prompt=context)

    def _transcribe_then_label(
        self,
        audio_path: Path,
        context: str = "",
    ) -> list[SpeakerTurn]:
        """Transcribe the whole audio once (fast), then assign speaker labels by
        overlaying diarization turns onto word/char timestamps.

        This avoids re-transcribing every speaker turn (which exploded to hours on
        long conversational videos). Text is split from the ORIGINAL transcription
        at speaker-change boundaries, so punctuation/quality is preserved — the
        timestamps are only used to decide *where* each speaker change happens.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        duration = self._get_audio_duration(audio_path)

        # 1. Diarization timeline (unload ASR first so pyannote runs alone in VRAM)
        self._unload_asr_model()
        turns = self._run_diarization(audio_path)
        self._unload_diarize_pipeline()

        if not turns:
            logger.warning("  No speaker turns detected; using plain transcription")
            text = self.transcribe(audio_path, initial_prompt=context)
            return [SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=duration, text=text)]

        # 2. Transcribe with word/char timestamps, chunked to stay VRAM-safe.
        y, sr = self._load_audio_mono_16k(audio_path)
        text, items = self._transcribe_with_timestamps_chunked(
            y,
            sr,
            duration,
            checkpoint_dir=self._checkpoint_dir(audio_path),
            context=context,
        )
        if not text.strip() or not items:
            logger.warning("  Empty text or no alignment items; cannot label")
            return []

        # 3. Overlay diarization on the word timestamps → speaker-labelled segments
        labelled = self._assign_speakers_to_text(text, items, turns)
        if self.convert_to_simplified:
            for t in labelled:
                t.text = self._to_simplified(t.text)
        n_spk = len({t.speaker for t in labelled})
        total = sum(len(t.text) for t in labelled)
        logger.info(
            f"  ✓ Labelled transcription: {total} chars, {len(labelled)} segments, {n_spk} speaker(s)"
        )
        return labelled

    def _transcribe_with_timestamps_chunked(
        self,
        y,
        sr: int,
        duration: float,
        checkpoint_dir: Optional[Path] = None,
        context: str = "",
    ) -> tuple[str, list[AlignItem]]:
        """Transcribe `y` in <=MAX_CHUNK_SECONDS chunks, returning the concatenated
        text and alignment items whose timestamps are offset to ABSOLUTE time.

        A single full-audio transcribe of a long file (e.g. 75 min / 4533s) was
        the cause of a multi-hour Step C stall: it overflows 12 GB VRAM into
        shared RAM and crawls (~6% GPU util), AND emits no incremental log during
        the one giant forward pass, so the scheduled-run watchdog false-killed it
        at 20 min. Chunking mirrors the proven single-speaker batch path: each
        forward pass stays VRAM-safe and logs per chunk. Timestamps are shifted by
        each chunk's start so downstream diarization overlay still works on the
        absolute timeline.
        """
        chunk_samples = MAX_CHUNK_SECONDS * sr
        n_chunks = max(1, (len(y) + chunk_samples - 1) // chunk_samples)
        logger.info(
            f"  Transcribing {duration:.0f}s in {n_chunks} chunk(s) "
            f"(<= {MAX_CHUNK_SECONDS}s each) + timestamp labelling..."
        )
        text_parts: list[str] = []
        items: list[AlignItem] = []
        for ci in range(n_chunks):
            start = ci * chunk_samples
            seg = y[start:start + chunk_samples]
            t0 = start / sr
            checkpoint_file = (
                checkpoint_dir / f"chunk_{ci:05d}.json"
                if checkpoint_dir is not None
                else None
            )
            if n_chunks > 1:
                logger.info(f"    chunk {ci + 1}/{n_chunks} @ {t0:.0f}s ({len(seg) / sr:.0f}s)")
            cached = None
            if checkpoint_file is not None and checkpoint_file.exists():
                try:
                    cached = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                    logger.info(f"      resumed chunk {ci + 1}/{n_chunks} from checkpoint")
                except (OSError, json.JSONDecodeError):
                    checkpoint_file.unlink(missing_ok=True)

            if cached is None:
                results = self.model.transcribe(
                    audio=(seg, sr),
                    context=context,
                    language=self.language,
                    return_time_stamps=True,
                )
                r = results[0] if results else None
                chunk_text = (r.text or "") if r else ""
                chunk_items = []
                if r and r.time_stamps is not None:
                    chunk_items = [
                        {
                            "text": getattr(it, "text", "") or "",
                            "start_time": (getattr(it, "start_time", 0.0) or 0.0) + t0,
                            "end_time": (getattr(it, "end_time", 0.0) or 0.0) + t0,
                        }
                        for it in r.time_stamps
                    ]
                cached = {"text": chunk_text, "items": chunk_items}
                chunk_secs = len(seg) / sr
                if (
                    n_chunks > 1
                    and chunk_secs >= MIN_NONEMPTY_CHUNK_SECONDS
                    and not chunk_text.strip()
                    and not chunk_items
                ):
                    raise RuntimeError(
                        f"Empty ASR chunk {ci + 1}/{n_chunks} "
                        f"at {t0:.0f}s; refusing partial transcript"
                    )
                if checkpoint_file is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    tmp = checkpoint_file.with_suffix(".tmp")
                    tmp.write_text(
                        json.dumps(cached, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    os.replace(tmp, checkpoint_file)

            chunk_secs = len(seg) / sr
            if (
                n_chunks > 1
                and chunk_secs >= MIN_NONEMPTY_CHUNK_SECONDS
                and not (cached.get("text", "") or "").strip()
                and not cached.get("items", [])
            ):
                if checkpoint_file is not None:
                    checkpoint_file.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Empty ASR checkpoint chunk {ci + 1}/{n_chunks} "
                    f"at {t0:.0f}s; refusing partial transcript"
                )

            text_parts.append(cached.get("text", ""))
            items.extend(AlignItem(**item) for item in cached.get("items", []))
            gc.collect()
        return "".join(text_parts), items

    def _checkpoint_dir(self, audio_path: Path) -> Path:
        audio_path = Path(audio_path)
        stat = audio_path.stat()
        identity = (
            f"{audio_path.resolve()}|{stat.st_size}|{MAX_CHUNK_SECONDS}|"
            f"{CHUNK_OVERLAP_SECONDS}|{self.model_name}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        return audio_path.parent / ".asr_checkpoints" / digest

    def clear_checkpoint(self, audio_path: Path) -> None:
        checkpoint_dir = self._checkpoint_dir(Path(audio_path))
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def _speaker_at(self, t: float, turns_sorted: list[SpeakerTurn]) -> str:
        """Speaker whose turn covers time t; else the nearest turn's speaker."""
        best_spk = None
        best_gap = None
        for turn in turns_sorted:
            if turn.start <= t <= turn.end:
                return turn.speaker
            gap = (turn.start - t) if t < turn.start else (t - turn.end)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_spk = turn.speaker
        return best_spk if best_spk is not None else "SPEAKER_00"

    def _assign_speakers_to_text(
        self, text: str, items: list, turns: list[SpeakerTurn]
    ) -> list[SpeakerTurn]:
        """Split the original transcription into speaker-labelled segments.

        `items` are aligned units (ForcedAlignItem: text, start_time, end_time) in
        transcription order. Each is assigned a speaker via its midpoint time, then
        the ORIGINAL text is cut at speaker-change boundaries (keeping punctuation).
        """
        if not items:
            return []
        turns_sorted = sorted(turns, key=lambda x: x.start)
        spk = [
            self._speaker_at((it.start_time + it.end_time) / 2.0, turns_sorted)
            for it in items
        ]
        # Map each item to a char offset in `text` (monotonic forward search).
        positions: list[int] = []
        pos = 0
        for it in items:
            tok = (getattr(it, "text", "") or "").strip()
            idx = text.find(tok, pos) if tok else -1
            if idx == -1:
                positions.append(pos)
            else:
                positions.append(idx)
                pos = idx + len(tok)

        segments: list[SpeakerTurn] = []
        seg_start_char = 0
        cur_spk = spk[0]
        seg_start_time = items[0].start_time
        last_end_time = items[0].end_time
        for i in range(1, len(items)):
            if spk[i] != cur_spk:
                cut = positions[i]
                seg_text = text[seg_start_char:cut].strip()
                if seg_text:
                    segments.append(SpeakerTurn(
                        speaker=cur_spk, start=seg_start_time,
                        end=last_end_time, text=seg_text,
                    ))
                seg_start_char = cut
                cur_spk = spk[i]
                seg_start_time = items[i].start_time
            last_end_time = items[i].end_time
        # Final segment (includes any trailing text/punctuation).
        seg_text = text[seg_start_char:].strip()
        if seg_text:
            segments.append(SpeakerTurn(
                speaker=cur_spk, start=seg_start_time, end=last_end_time, text=seg_text,
            ))
        return segments

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

        # Unload ASR model first so pyannote runs alone in VRAM
        self._unload_asr_model()

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
            text = self.transcribe(audio_path, initial_prompt=initial_prompt)
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
                    context=[initial_prompt] * len(audio_segments),
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
                # Free fragmented VRAM before retrying individually
                gc.collect()
                torch.cuda.empty_cache()
                # Fallback: transcribe individually
                for j, turn in enumerate(batch_turns):
                    try:
                        seg = audio_segments[j]
                        res = self.model.transcribe(
                            audio=seg,
                            context=initial_prompt,
                            language=self.language,
                        )
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
        asr_context = initial_prompt or self.build_context(title, author)
        if self.diarize:
            turns = self.transcribe_with_diarization(audio_path, initial_prompt=asr_context)
            markdown = self._format_turns_to_markdown(turns, title=title, author=author)
            
            # Use heuristic on the first few turns if self.language is None
            detected_lang = self.language
            if not detected_lang:
                sample_text = " ".join([t.text for t in turns[:5]])
                detected_lang = self._detect_language_heuristic(sample_text)
            
            return markdown, detected_lang
        else:
            text = self.transcribe(audio_path, initial_prompt=asr_context)
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
