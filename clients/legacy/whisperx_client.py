"""WhisperX client for transcription with speaker diarization.

Uses whisperX for high-accuracy transcription with speaker labels.
Requires HuggingFace token for pyannote speaker diarization model.
"""

from __future__ import annotations

import gc
import os
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Fix for Torch 2.6+ weights_only loading and Windows symlink issues
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([ListConfig, DictConfig])
except ImportError:
    pass



class WhisperXClient:
    """WhisperX speech-to-text client with speaker diarization."""
    
    # Model size -> approximate VRAM (GB) for whisperX
    MODEL_SIZES = {
        "tiny": 1,
        "base": 1,
        "small": 2,
        "medium": 5,
        "large-v2": 10,
        "large-v3": 10,
    }
    
    # Default initial prompt for Chinese transcription with proper punctuation
    # This guides Whisper to output text with correct punctuation marks
    DEFAULT_INITIAL_PROMPT = (
        "以下是普通话的句子，使用简体中文输出，包含正确的标点符号。"
        "这是一段视频的语音转录，请注意使用逗号、句号、问号和感叹号。"
    )
    
    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        hf_token: Optional[str] = None,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ):
        """Initialize WhisperX client.
        
        Args:
            model: Whisper model size (tiny/base/small/medium/large-v2/large-v3)
            device: Device to use (auto/cpu/cuda)
            compute_type: Compute type (float16/int8)
            hf_token: HuggingFace token for pyannote diarization
            language: Language code (e.g., "zh" for Chinese), None for auto-detect
            initial_prompt: Initial prompt to guide transcription style (improves punctuation)
            min_speakers: Minimum number of speakers expected
            max_speakers: Maximum number of speakers expected
        """
        self.model_name = model
        self.compute_type = compute_type
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.language = language
        self.initial_prompt = initial_prompt if initial_prompt else self.DEFAULT_INITIAL_PROMPT
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Models will be loaded lazily
        self._whisper_model = None
        self._diarize_model = None
        self._align_model = None
        self._align_metadata = None
        
        logger.info(f"WhisperX: Initialized with model '{model}' on device '{self.device}'")
        
    def _ensure_whisper_model(self):
        """Load Whisper model if not already loaded."""
        if self._whisper_model is None:
            import whisperx
            logger.info(f"WhisperX: Loading model '{self.model_name}'...")
            # Optimized ASR options to reduce missed content
            asr_options = {
                "initial_prompt": self.initial_prompt,
                "beam_size": 5,  # Increase from default 1 for better accuracy
                "best_of": 5,    # Consider more candidates
                "patience": 1.0, # Wait longer before stopping beam search
                "suppress_blank": True,
            }
            # VAD parameters to capture more speech (lower threshold = more sensitive)
            vad_options = {
                "vad_onset": 0.3,           # Default 0.5, lower = more sensitive to speech start
                "vad_offset": 0.3,          # Default 0.363, lower = keep more speech
            }
            self._whisper_model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                asr_options=asr_options,
                vad_options=vad_options,
            )
            logger.info("WhisperX: Model loaded.")

            
    def _ensure_diarize_model(self):
        """Load diarization pipeline if not already loaded."""
        if self._diarize_model is None:
            import whisperx
            if not self.hf_token:
                raise ValueError(
                    "HuggingFace token required for speaker diarization. "
                    "Set hf_token in config or HF_TOKEN environment variable."
                )
            
            logger.info("WhisperX: Loading diarization model...")
            # Robust loading of DiarizationPipeline
            try:
                from whisperx.diarize import DiarizationPipeline
            except (ImportError, AttributeError):
                if hasattr(whisperx, "DiarizationPipeline"):
                    DiarizationPipeline = whisperx.DiarizationPipeline
                else:
                    raise ImportError("Could not find DiarizationPipeline in whisperx or whisperx.diarize")
            
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
            logger.info("WhisperX: Diarization model loaded.")
            
    def transcribe_with_diarization(
        self,
        audio_path: Path,
        batch_size: int = 16,
    ) -> dict:
        """Transcribe audio with speaker diarization.
        
        Args:
            audio_path: Path to audio file
            batch_size: Batch size for transcription
            
        Returns:
            Dictionary with 'segments' containing speaker-labeled transcription
        """
        import whisperx
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Load audio
        logger.info(f"WhisperX: Loading audio from {audio_path.name}...")
        audio = whisperx.load_audio(str(audio_path))
        
        # Step 1: Transcribe
        self._ensure_whisper_model()
        logger.info("WhisperX: Transcribing...")
        result = self._whisper_model.transcribe(
            audio,
            batch_size=batch_size,
            language=self.language
        )
        detected_language = result.get("language", self.language or "zh")
        logger.info(f"WhisperX: Detected language: {detected_language}")
        
        # Step 2: Align whisper output (optional - improves diarization accuracy)
        try:
            logger.info("WhisperX: Aligning transcription...")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )
            result = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
        except Exception as e:
            logger.error(f"WhisperX: Alignment failed ({e}), proceeding without alignment...")
            # Continue with unaligned segments - diarization will still work
        
        # Step 3: Speaker diarization
        self._ensure_diarize_model()
        logger.info("WhisperX: Performing speaker diarization...")
        diarize_segments = self._diarize_model(
            audio,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )
        
        # Step 4: Assign speaker labels to segments
        logger.info("WhisperX: Assigning speaker labels...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return result
    
    def transcribe_to_markdown(
        self,
        audio_path: Path,
        title: str = "",
        author: str = "",
        target_paragraph_length: int = 500,
    ) -> str:
        """Transcribe audio and format as Markdown with speaker labels and smart paragraphing.
        
        Args:
            audio_path: Path to audio file
            title: Optional title for the Markdown document
            author: Optional author/uploader name
            target_paragraph_length: Target length for each paragraph in characters
            
        Returns:
            Tuple of (Markdown formatted string, detected_language)
        """
        result = self.transcribe_with_diarization(audio_path)
        detected_lang = result.get("language", self.language or "zh")
        
        lines = []
        if title:
            lines.append(f"# {title}\n")
        
        # Count unique speakers and their contribution
        speaker_stats = {}  # speaker -> total_chars
        for segment in result.get("segments", []):
            spk = segment.get("speaker", "UNKNOWN")
            txt = segment.get("text", "").strip()
            if txt:
                speaker_stats[spk] = speaker_stats.get(spk, 0) + len(txt)
        
        total_chars = sum(speaker_stats.values())
        
        # Determine significant speakers (at least 5% of content or 100 chars)
        # These thresholds are heuristics to filter out "ghost" speakers
        significant_speakers = {
            spk for spk, count in speaker_stats.items() 
            if (total_chars > 0 and count / total_chars > 0.05) or (count > 100)
        }
        
        # Find the most prominent speaker (fallback for insignificant speakers)
        most_prominent_spk = max(speaker_stats.items(), key=lambda x: x[1])[0] if speaker_stats else None
        
        # If no significant speakers (extremely short?), use the most prominent one
        if not significant_speakers and most_prominent_spk:
            significant_speakers = {most_prominent_spk}
            
        num_significant = len(significant_speakers)

        # Group segments by speaker, then apply smart paragraphing
        current_speaker = None
        current_segments = []
        speaker_blocks = []  # List of (speaker, [segments])
        
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            # Map insignificant speakers to the current speaker if exists,
            # or fallback to most prominent speaker if it's the start.
            if speaker not in significant_speakers:
                if current_speaker is not None:
                    speaker = current_speaker
                else:
                    speaker = most_prominent_spk

            if speaker == current_speaker:
                current_segments.append(segment)
            else:
                if current_speaker is not None and current_segments:
                    speaker_blocks.append((current_speaker, current_segments))
                current_speaker = speaker
                current_segments = [segment]
        
        # Flush last block
        if current_speaker is not None and current_segments:
            speaker_blocks.append((current_speaker, current_segments))
        
        # Now apply smart paragraphing to each speaker block
        for speaker, segments in speaker_blocks:
            paragraphs = self._segments_to_paragraphs(segments, target_paragraph_length)
            
            for para in paragraphs:
                # ONLY use speaker labels if there's more than one significant speaker
                if num_significant > 1:
                    speaker_label = self._format_speaker_label(speaker)
                    lines.append(f"**{speaker_label}** {para}\n")
                else:
                    lines.append(f"{para}\n")
            
        return "\n".join(lines), detected_lang
    
    def _segments_to_paragraphs(
        self,
        segments: list[dict],
        target_length: int = 500,
    ) -> list[str]:
        """Convert Whisper segments into paragraphs of roughly target_length chars.
        
        Groups consecutive segments into paragraphs using segment boundaries
        as natural break points.
        
        Args:
            segments: List of Whisper segment dicts with 'text' field.
            target_length: Target paragraph length in characters.
            
        Returns:
            List of paragraph strings.
        """
        if not segments:
            return []
        
        paragraphs = []
        current_para = []
        current_len = 0
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            # If adding this segment would exceed target and we have content,
            # start a new paragraph
            if current_len + len(text) > target_length and current_para:
                paragraphs.append("".join(current_para))
                current_para = []
                current_len = 0
            
            current_para.append(text)
            current_len += len(text)
        
        # Don't forget the last paragraph
        if current_para:
            paragraphs.append("".join(current_para))
        
        return paragraphs

    
    def _format_speaker_label(self, speaker: str) -> str:
        """Format speaker ID to a readable label."""
        # Convert "SPEAKER_00" -> "[说话人 A]"
        if speaker.startswith("SPEAKER_"):
            try:
                num = int(speaker.split("_")[1])
                # Use letters A, B, C... for first 26 speakers
                if num < 26:
                    return f"[说话人 {chr(65 + num)}]"
                return f"[说话人 {num + 1}]"
            except (IndexError, ValueError):
                pass
        return f"[{speaker}]"
    
    def unload_model(self):
        """Unload all models to free GPU memory."""
        logger.info("WhisperX: Unloading models...")
        
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            
        if self._diarize_model is not None:
            del self._diarize_model
            self._diarize_model = None
            
        if self._align_model is not None:
            del self._align_model
            self._align_model = None
            self._align_metadata = None
            
        # Force garbage collection and CUDA cache clear
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("WhisperX: Models unloaded, GPU memory freed.")
