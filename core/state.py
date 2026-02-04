"""State management for the pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from .models import VideoState, QueueItem, VideoInfo


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    """Parse ISO format datetime string."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


class StateManager:
    """Manages pipeline state persistence."""
    
    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self._state: dict[str, Any] = {"videos": {}}
        self._load()
    
    def _load(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                self._state = json.loads(self.state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                self._state = {"videos": {}}
        if "videos" not in self._state:
            self._state["videos"] = {}
    
    def _save(self) -> None:
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def get_video_state(self, bvid: str) -> VideoState:
        """Get state for a specific video."""
        data = self._state["videos"].get(bvid, {})
        return VideoState.from_dict(bvid, data)
    
    def get_status(self, bvid: str) -> str:
        """Get status for a specific video."""
        return self.get_video_state(bvid).status
    
    def update(self, bvid: str, **fields) -> None:
        """Update state for a specific video."""
        if bvid not in self._state["videos"]:
            self._state["videos"][bvid] = {}
        
        self._state["videos"][bvid].update(fields)
        self._save()
    
    def mark_seen(self, bvid: str, pubdate: Optional[int] = None) -> None:
        """Mark a video as seen (update first_seen/last_seen)."""
        now = now_iso()
        if bvid not in self._state["videos"]:
            self._state["videos"][bvid] = {
                "first_seen": now,
                "status": "new",
            }
        
        self._state["videos"][bvid]["last_seen"] = now
        if pubdate is not None:
            self._state["videos"][bvid]["pubdate"] = pubdate
            
        self._save()
    
    def get_pending_items(self, status: str = "new") -> list[str]:
        """Get list of bvids with the specified status."""
        return [
            bvid for bvid, data in self._state["videos"].items()
            if data.get("status") == status
        ]
    
    def get_all_bvids_with_summaries(self) -> list[str]:
        """Get list of bvids that have a summary_md file path, regardless of status."""
        return [
            bvid for bvid, data in self._state["videos"].items()
            if data.get("summary_md")
        ]
    
    def get_all_bvids_with_transcripts(self) -> list[str]:
        """Get list of bvids that have a transcript_md file path, regardless of status."""
        return [
            bvid for bvid, data in self._state["videos"].items()
            if data.get("transcript_md") or data.get("summary_md")
        ]
    
    def build_queue(
        self,
        videos: list[VideoInfo],
        max_items: int = 50,
        max_age_days: int = 30,
    ) -> list[QueueItem]:
        """Build processing queue from video list.
        
        - Skip videos that already have status != 'new'
        - Skip videos older than max_age_days
        - Cap at max_items
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days)
        queue: list[QueueItem] = []
        
        for video in videos:
            bvid = video.bvid
            if not bvid:
                continue
            
            # Mark as seen
            self.mark_seen(bvid, pubdate=video.pubdate)
            
            state = self.get_video_state(bvid)
            
            # Skip if already uploaded
            if state.status == "uploaded":
                continue
            
            # Skip terminal states
            if state.status == "skipped_old":
                continue
            
            # Check age
            if state.first_seen:
                try:
                    first_seen_dt = parse_iso(state.first_seen)
                    if first_seen_dt < cutoff:
                        self.update(bvid, status="skipped_old")
                        continue
                except ValueError:
                    pass
            
            # Only queue "new", "error", "downloading", "transcribing", or "success" (for re-upload)
            if state.status not in ("new", "error", "downloading", "transcribing", "success"):
                continue
            
            queue.append(QueueItem.from_video_info(video))
            
            if len(queue) >= max_items:
                break
        
        return queue
    
    def get_summary_stats(self) -> dict[str, int]:
        """Get summary statistics of all videos."""
        stats: dict[str, int] = {}
        for data in self._state["videos"].values():
            status = data.get("status", "unknown")
            stats[status] = stats.get(status, 0) + 1
        return stats
