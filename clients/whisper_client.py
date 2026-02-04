"""Whisper client for local speech-to-text transcription.

Uses OpenAI's open-source Whisper model for local transcription.
Automatically detects GPU/CPU and uses appropriate device.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Suppress whisper's FP16 warning on CPU
os.environ.setdefault("WHISPER_WARN_FP16", "0")


class WhisperClient:
    """Local Whisper speech-to-text client."""
    
    # Available models and their approximate VRAM requirements
    MODELS = {
        "tiny": "~1 GB",
        "base": "~1 GB", 
        "small": "~2 GB",
        "medium": "~5 GB",
        "large": "~10 GB",
        "large-v2": "~10 GB",
        "large-v3": "~10 GB",
    }
    
    # Prompt to encourage punctuation in Chinese transcription
    CHINESE_PUNCTUATION_PROMPT = (
        "以下是普通话的句子，包含标点符号。"
        "今天天气很好，我们一起去公园玩吧！你觉得怎么样？"
        "好的，没问题。我们下午三点出发，记得带上水和零食。"
    )
    
    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
        language: Optional[str] = None,
        convert_to_simplified: bool = True,
        ffmpeg_location: Optional[str] = None,
    ):
        """Initialize Whisper client.
        
        Args:
            model_name: Whisper model to use (tiny/base/small/medium/large/large-v3).
            device: Device to use ("auto", "cpu", "cuda", "mps").
            language: Language code (e.g., "zh"). None = auto-detect.
            convert_to_simplified: If True, convert Traditional Chinese to Simplified.
            ffmpeg_location: Optional path to ffmpeg binary directory.
        """
        self.model_name = model_name
        self.language = language  # None = auto-detect
        self.convert_to_simplified = convert_to_simplified
        self._model = None
        self._converter = None
        
        # Add ffmpeg to PATH if provided
        if ffmpeg_location and os.path.isdir(ffmpeg_location):
            current_path = os.environ.get("PATH", "")
            if ffmpeg_location not in current_path:
                os.environ["PATH"] = ffmpeg_location + os.pathsep + current_path
                print(f"Whisper: Added ffmpeg to PATH: {ffmpeg_location}")
        
        # Determine device
        if device == "auto":
            self.device = self._detect_device()
        else:
            self.device = device
        
        print(f"Whisper: Using model '{model_name}' on device '{self.device}'")
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Whisper: CUDA GPU detected: {gpu_name}")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("Whisper: Apple MPS detected")
                return "mps"
            else:
                print("Whisper: Using CPU")
                return "cpu"
        except ImportError:
            return "cpu"
    
    @property
    def model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            try:
                import whisper
            except ImportError:
                raise RuntimeError(
                    "openai-whisper not installed. Please install it:\n"
                    "  pip install openai-whisper\n"
                    "Note: This also requires ffmpeg to be installed."
                )
            
            print(f"Loading Whisper model '{self.model_name}'...")
            self._model = whisper.load_model(self.model_name, device=self.device)
            print("Whisper model loaded.")
        
        return self._model

    def unload_model(self):
        """Unload the model from memory to save VRAM."""
        if self._model is not None:
            print(f"Unloading Whisper model '{self.model_name}'...")
            self._model = None
            
            # Try to force garbage collection and empty CUDA cache
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("  ✓ CUDA cache cleared")
            except ImportError:
                pass
    
    def transcribe(
        self,
        audio_path: Path,
        include_timestamps: bool = False,
    ) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (.m4a, .mp3, .wav, etc.)
            include_timestamps: If True, include timestamps in output.
            
        Returns:
            Transcribed text as a string.
        """
        result, detected_lang = self._transcribe_raw(audio_path)
        
        if include_timestamps:
            # Format with timestamps
            lines = []
            for segment in result.get("segments", []):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                lines.append(f"[{start} --> {end}] {text}")
            text = "\n".join(lines)
        else:
            # Just the text
            text = result.get("text", "").strip()
        
        # Convert to Simplified Chinese if enabled and language is Chinese
        if self.convert_to_simplified and detected_lang in ("zh", "Chinese"):
            text = self._to_simplified(text)
        
        return text
    
    def _transcribe_raw(self, audio_path: Path) -> tuple[dict, str]:
        """Transcribe and return raw result with segments.
        
        Returns:
            Tuple of (whisper_result_dict, detected_language)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing: {audio_path.name}...")
        
        # Build transcribe options
        transcribe_opts = {
            "verbose": False,
        }
        
        # Language: None = auto-detect, otherwise use specified
        if self.language:
            transcribe_opts["language"] = self.language
            
        # Add initial_prompt to encourage punctuation for Chinese
        if self.language in (None, "zh", "Chinese"):
            transcribe_opts["initial_prompt"] = self.CHINESE_PUNCTUATION_PROMPT
        
        # Transcribe with Whisper
        result = self.model.transcribe(str(audio_path), **transcribe_opts)
        
        # Get detected language
        detected_lang = result.get("language", self.language)
        if self.language is None:
            print(f"  Detected language: {detected_lang}")
        
        return result, detected_lang
    
    def _to_simplified(self, text: str) -> str:
        """Convert Traditional Chinese to Simplified Chinese."""
        if self._converter is None:
            try:
                from opencc import OpenCC
                self._converter = OpenCC('t2s')  # Traditional to Simplified
            except ImportError:
                print("Warning: opencc not installed, skipping Traditional to Simplified conversion")
                return text
        return self._converter.convert(text)
    
    def transcribe_to_markdown(
        self,
        audio_path: Path,
        title: str,
        author: str = "",
    ) -> str:
        """Transcribe and format as Markdown.
        
        Args:
            audio_path: Path to audio file.
            title: Title for the document.
            author: Author/creator name.
            
        Returns:
            Markdown formatted transcription.
        """
        # Get raw result with segments for better paragraph splitting
        result, detected_lang = self._transcribe_raw(audio_path)
        
        # Use segments to create natural paragraphs
        paragraphs = self._segments_to_paragraphs(
            result.get("segments", []),
            target_length=500,
        )
        
        # Convert to Simplified Chinese if enabled
        if self.convert_to_simplified and detected_lang in ("zh", "Chinese"):
            paragraphs = [self._to_simplified(p) for p in paragraphs]
        
        # Build markdown
        lines = [f"# {title}", ""]
        
        if author:
            lines.append(f"**UP主**: {author}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        for para in paragraphs:
            lines.append(para)
            lines.append("")
        
        if not paragraphs:
            return f"# {title}\n\n**UP主**: {author}\n\n---\n\n(转录失败或无文字内容)"
            
        return "\n".join(lines)
    
    def _segments_to_paragraphs(
        self,
        segments: list[dict],
        target_length: int = 500,
    ) -> list[str]:
        """Convert Whisper segments into paragraphs.
        
        Groups consecutive segments into paragraphs of roughly target_length chars.
        Uses segment boundaries as natural break points.
        
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
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _split_into_paragraphs(
        self,
        text: str,
        target_length: int = 500,
    ) -> list[str]:
        """Split text into paragraphs for readability.
        
        Args:
            text: Raw transcription text.
            target_length: Target paragraph length in characters.
            
        Returns:
            List of paragraph strings.
        """
        if not text:
            return []
        
        # Chinese punctuation for sentence splitting
        sentence_endings = "。！？；.!?;"
        
        paragraphs = []
        current = []
        current_len = 0
        
        # Split into rough sentences
        sentences = []
        start = 0
        for i, char in enumerate(text):
            if char in sentence_endings:
                sentences.append(text[start:i+1].strip())
                start = i + 1
        
        # Add remaining text
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                sentences.append(remaining)
        
        # Group sentences into paragraphs
        for sentence in sentences:
            if current_len + len(sentence) > target_length and current:
                paragraphs.append("".join(current))
                current = []
                current_len = 0
            
            current.append(sentence)
            current_len += len(sentence)
        
        if current:
            paragraphs.append("".join(current))
        
        return paragraphs if paragraphs else [text]
