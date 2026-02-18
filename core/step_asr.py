"""Transcription step for the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base_step import BaseStep
from .models import QueueItem
from .state import now_iso
from utils import safe_filename

class StepASR(BaseStep):
    """Step C: Transcribe downloaded audio with Whisper/Qwen3-ASR."""
    
    def run(self, max_items: Optional[int] = None) -> dict[str, int]:
        max_items = self.get_max_items(max_items)
        engine_name = self.config.get("asr_engine", "whisper").upper()
        self.logger.info(f"=== Step C: Transcribe with {engine_name} (max {max_items}) ===")
        
        queue = self.load_queue()
        if not queue:
            return {"error": 1}
        
        stats = {
            "processed": 0,
            "transcribed": 0,
            "no_audio": 0,
            "already_transcribed": 0,
            "error": 0,
        }
        
        processed_bvids = []
        for item in queue:
            if stats["processed"] >= max_items:
                break
            
            status = self.state.get_status(item.bvid)
            # Only process downloaded items
            if status not in ("downloaded", "transcribing"):
                if status in ("transcript_ready", "correcting", "corrected", "summarizing", "summarized", "success", "uploaded"):
                    stats["already_transcribed"] += 1
                    stats["processed"] += 1
                continue
            
            video_state = self.state.get_video_state(item.bvid)
            if not video_state.audio_path or not Path(video_state.audio_path).exists():
                self.logger.error(f"✗ No audio file: {item.title[:50]}")
                stats["no_audio"] += 1
                continue
            
            audio_path = Path(video_state.audio_path)
            self.logger.info(f"Transcribing: {item.title[:60]}...")
            self.state.update(item.bvid, status="transcribing", title=item.title, up_name=item.up_name, last_attempt=now_iso())
            
            try:
                transcript_md, detected_lang = self.pipeline.asr_client.transcribe_to_markdown(
                    audio_path,
                    title=item.title,
                    author=item.up_name,
                )
                
                safe_title = safe_filename(item.title)
                transcript_path = self.pipeline.output_dir / f"{safe_title}.md"
                transcript_path.write_text(transcript_md, encoding="utf-8")
                
                self.state.update(
                    item.bvid,
                    status="transcript_ready",
                    title=item.title,
                    up_name=item.up_name,
                    transcript_md=str(transcript_path),
                    language=detected_lang,
                    last_attempt=now_iso(),
                )
                self.logger.info(f"  ✓ Saved: {transcript_path.name}")
                stats["transcribed"] += 1
                processed_bvids.append(item.bvid)
            except Exception as e:
                self.state.update(
                    item.bvid,
                    status="error",
                    error=f"Transcription failed: {str(e)}",
                    last_attempt=now_iso(),
                )
                self.logger.error(f"  ✗ Transcription failed: {e}")
                stats["error"] += 1
            
            stats["processed"] += 1

        # Explicitly unload model after the batch
        if hasattr(self.pipeline.asr_client, "unload_model"):
            self.pipeline.asr_client.unload_model()
            
        self.logger.info(f"=== Step C Complete ===")
        self.logger.info(f"Transcribed: {stats['transcribed']}, Already: {stats['already_transcribed']}, Errors: {stats['error']}")
        return stats, processed_bvids
