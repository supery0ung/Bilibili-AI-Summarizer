"""LLM correction and summarization step for the pipeline."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from .base_step import BaseStep
from .models import QueueItem
from .state import now_iso
from clients.ollama_client import build_final_markdown

class StepLLM(BaseStep):
    """Steps D & E: Correct and Summarize transcripts with LLM."""
    
    def run(self, max_items: Optional[int] = None) -> dict[str, int]:
        max_items = self.get_max_items(max_items)
        self.logger.info(f"=== Steps D & E: LLM Correct & Summarize (max {max_items}) ===")
        
        queue = self.load_queue()
        if not queue:
            return {"error": 1}
        
        stats = {
            "processed": 0,
            "corrected": 0,
            "summarized": 0,
            "error": 0,
        }
        
        processed_bvids = []
        for item in queue:
            if stats["processed"] >= max_items:
                break
            
            status = self.state.get_status(item.bvid)
            # Only process items with raw transcripts
            if status not in ("transcript_ready", "correcting", "corrected", "summarizing"):
                continue
            
            video_state = self.state.get_video_state(item.bvid)
            if not video_state.transcript_md or not Path(video_state.transcript_md).exists():
                continue
            
            self.logger.info(f"LLM Processing: {item.title[:60]}...")
            
            try:
                # Step D: Correction
                if status in ("transcript_ready", "correcting"):
                    self.state.update(item.bvid, status="correcting", last_attempt=now_iso())
                    corrected_path = self._correct_item(item, video_state)
                    if corrected_path:
                        stats["corrected"] += 1
                        # Update video_state manually for the next sub-step
                        video_state.corrected_md = str(corrected_path)
                        status = "corrected"
                
                # Step E: Summarization
                if status in ("corrected", "summarizing"):
                    self.state.update(item.bvid, status="summarizing", last_attempt=now_iso())
                    final_path = self._summarize_item(item, video_state)
                    if final_path:
                        stats["summarized"] += 1
                
                processed_bvids.append(item.bvid)

            except Exception as e:
                self.state.update(item.bvid, status="error", error=f"LLM step failed: {e}")
                self.logger.error(f"  ✗ LLM step failed: {e}")
                stats["error"] += 1
            
            stats["processed"] += 1

        # Explicitly unload model after the batch
        if hasattr(self.pipeline.ollama, "unload_model"):
            self.pipeline.ollama.unload_model()
            
        return stats, processed_bvids

    def _correct_item(self, item: QueueItem, video_state: Any) -> Optional[Path]:
        transcript_path = Path(video_state.transcript_md)
        content = transcript_path.read_text(encoding="utf-8")
        
        # Extract body (after ---)
        body_start = content.find("---")
        if body_start != -1:
            body_start = content.find("\n", body_start) + 1
            raw_text = content[body_start:].strip()
        else:
            raw_text = content
        
        # Identify speakers
        self.logger.info(f"  Identifying speakers...")
        speaker_map = self.pipeline.ollama.identify_speakers(
            raw_text, title=item.title, author=item.up_name
        )
        
        # Batch Correction (NEW: much faster)
        self.logger.info(f"  Correcting text (batch mode)...")
        corrected_text = self.pipeline.ollama.correct_text_batched(
            raw_text, 
            title=item.title, 
            author=item.up_name, 
            speaker_map=speaker_map,
            language=getattr(video_state, 'language', 'zh')
        )
        
        # Build corrected markdown
        corrected_md = f"# {item.title}\n\n**UP主**: {item.up_name}\n\n---\n\n{corrected_text}"
        corrected_path = transcript_path.with_suffix(".corrected.md")
        corrected_path.write_text(corrected_md, encoding="utf-8")
        
        self.state.update(item.bvid, status="corrected", corrected_md=str(corrected_path))
        self.logger.info(f"  ✓ Corrected: {corrected_path.name}")
        return corrected_path

    def _summarize_item(self, item: QueueItem, video_state: Any) -> Optional[Path]:
        corrected_path = Path(video_state.corrected_md)
        content = corrected_path.read_text(encoding="utf-8")
        
        body_start = content.find("---")
        if body_start != -1:
            body_start = content.find("\n", body_start) + 1
            corrected_text = content[body_start:].strip()
        else:
            corrected_text = content
        
        self.logger.info("  Generating summary...")
        summary = self.pipeline.ollama.summarize(corrected_text, title=item.title, author=item.up_name)
        
        final_md = build_final_markdown(
            title=item.title, author=item.up_name, summary=summary, corrected_text=corrected_text
        )
        final_path = corrected_path.with_name(corrected_path.name.replace(".corrected.md", ".final.md"))
        final_path.write_text(final_md, encoding="utf-8")
        
        # Step F picks up 'summarized' or 'success'
        self.state.update(
            item.bvid, 
            status="summarized",
            summary_md=str(final_path)
        )
        self.logger.info(f"  ✓ Summarized: {final_path.name}")
        return final_path
