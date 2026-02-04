"""Qwen3 ASR client for local speech-to-text transcription.

Uses Alibaba's Qwen3-ASR-1.7B model for high-quality multilingual speech recognition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

class Qwen3ASRClient:
    """Local Qwen3 ASR speech-to-text client."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        language: Optional[str] = None,
        convert_to_simplified: bool = True,
        ffmpeg_location: Optional[str] = None,
    ):
        """Initialize Qwen3 ASR client.
        
        Args:
            model_name: HuggingFace model ID.
            device: Device to use (e.g., "cuda:0", "cpu").
            language: Language name (e.g., "Chinese", "English") or None for auto.
            convert_to_simplified: If True, convert Traditional Chinese to Simplified.
            ffmpeg_location: Optional path to ffmpeg binary directory.
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.convert_to_simplified = convert_to_simplified
        self._model = None
        self._converter = None
        
        # Add ffmpeg to PATH if provided (Qwen3-ASR uses librosa which needs ffmpeg/ffprobe)
        if ffmpeg_location and os.path.isdir(ffmpeg_location):
            current_path = os.environ.get("PATH", "")
            if ffmpeg_location not in current_path:
                os.environ["PATH"] = ffmpeg_location + os.pathsep + current_path
                print(f"Qwen3: Added ffmpeg to PATH: {ffmpeg_location}")
        
        print(f"Qwen3: Initialized with model '{model_name}' on device '{self.device}'")

    @property
    def model(self):
        """Lazy-load the Qwen3-ASR model."""
        if self._model is None:
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
            except ImportError:
                raise RuntimeError(
                    "qwen-asr not installed. Please install it:\n"
                    "  pip install qwen-asr"
                )
            
            print(f"Loading Qwen3-ASR model '{self.model_name}'...")
            # Use bfloat16 for better performance/accuracy on 3080 Ti
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map=self.device,
                max_inference_batch_size=32,
                max_new_tokens=4096,  # Support long audio (B站视频可能较长)
            )
            print("Qwen3-ASR model loaded.")
        
        return self._model

    def unload_model(self):
        """Unload the model from memory to save VRAM."""
        if self._model is not None:
            print(f"Unloading Qwen3-ASR model '{self.model_name}'...")
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
    ) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Transcribed text as a string.
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing with Qwen3: {audio_path.name}...")
        
        # Transcribe
        results = self.model.transcribe(
            audio=str(audio_path),
            language=self.language,
        )
        
        if not results:
            return ""
            
        text = results[0].text.strip()
        
        # Convert to Simplified Chinese if enabled
        if self.convert_to_simplified:
            text = self._to_simplified(text)
        
        return text

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
        text = self.transcribe(audio_path)
        
        # Split into paragraphs (reuse simple sentence splitting)
        paragraphs = self._split_into_paragraphs(text)
        
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
        
        return "\n".join(lines)

    def _to_simplified(self, text: str) -> str:
        """Convert Traditional Chinese to Simplified Chinese."""
        if self._converter is None:
            try:
                from opencc import OpenCC
                self._converter = OpenCC('t2s')
            except ImportError:
                print("Warning: opencc not installed, skipping conversion")
                return text
        return self._converter.convert(text)

    def _split_into_paragraphs(self, text: str, target_length: int = 500) -> list[str]:
        """Split text into paragraphs for readability."""
        if not text:
            return []
        
        # Simple sentence splitting
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
