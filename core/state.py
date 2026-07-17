"""State management for the pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import threading
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
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {"videos": {}}
        self._load()
    
    def _load(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                self._state = json.loads(self.state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                raise RuntimeError(
                    f"State file is unreadable; refusing to start with empty state: "
                    f"{self.state_file}"
                ) from e
        if "videos" not in self._state:
            self._state["videos"] = {}
    
    def _save(self) -> None:
        """Atomically save state to disk.

        Download workers update state concurrently.  Serialize the in-memory
        mutation and replace the JSON only after a complete temporary file has
        been flushed, so readers never observe a partial document.
        """
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, ensure_ascii=False, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.state_file.name}.",
            suffix=".tmp",
            dir=self.state_file.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.state_file)
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
    
    def get_video_state(self, bvid: str) -> VideoState:
        """Get state for a specific video."""
        with self._lock:
            data = dict(self._state["videos"].get(bvid, {}))
            return VideoState.from_dict(bvid, data)
    
    def get_status(self, bvid: str) -> str:
        """Get status for a specific video."""
        return self.get_video_state(bvid).status
    
    def update(self, bvid: str, **fields) -> None:
        """Update state for a specific video."""
        with self._lock:
            if bvid not in self._state["videos"]:
                self._state["videos"][bvid] = {}
            self._state["videos"][bvid].update(fields)
            self._save()
    
    def mark_seen(self, bvid: str, pubdate: Optional[int] = None) -> None:
        """Mark a video as seen (update first_seen/last_seen)."""
        with self._lock:
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
        with self._lock:
            return [
                bvid for bvid, data in self._state["videos"].items()
                if data.get("status") == status
            ]
    
    def get_all_bvids_with_summaries(self) -> list[str]:
        """Get list of bvids that have a summary_md file path, regardless of status."""
        with self._lock:
            return [
                bvid for bvid, data in self._state["videos"].items()
                if data.get("summary_md")
            ]
    
    def get_all_bvids_with_transcripts(self) -> list[str]:
        """Get list of bvids that have a transcript_md file path, regardless of status."""
        with self._lock:
            return [
                bvid for bvid, data in self._state["videos"].items()
                if data.get("transcript_md") or data.get("summary_md")
            ]
    
    def build_queue(
        self,
        videos: list[VideoInfo],
        max_items: int = 50,
    ) -> list[QueueItem]:
        """Build processing queue from video list.
        
        - Skip videos that already have status != 'new'
        - Cap at max_items
        """
        with self._lock:
            queue: list[QueueItem] = []
            now = now_iso()
            changed = False

            for video in videos:
                bvid = video.bvid
                if not bvid:
                    continue

                data = self._state["videos"].setdefault(
                    bvid, {"first_seen": now, "status": "new"}
                )
                data["last_seen"] = now
                if video.pubdate is not None:
                    data["pubdate"] = video.pubdate
                changed = True
                state = VideoState.from_dict(bvid, data)

                if state.status == "uploaded":
                    continue
                if state.status in ("skipped_old", "skipped_removed"):
                    continue

                queueable = (
                    "new", "error", "downloading", "downloaded", "transcribing", "transcript_ready",
                    "correcting", "corrected", "summarizing", "summarized", "success"
                )
                if state.status not in queueable:
                    continue

                queue.append(QueueItem.from_video_info(video))
                if len(queue) >= max_items:
                    break

            if changed:
                self._save()
            return queue
    
    def get_summary_stats(self) -> dict[str, int]:
        """Get summary statistics of all videos."""
        with self._lock:
            stats: dict[str, int] = {}
            for data in self._state["videos"].values():
                status = data.get("status", "unknown")
                stats[status] = stats.get(status, 0) + 1
            return stats

    def reset_non_uploaded_items(self) -> int:
        """Reset all videos that are not in terminal states back to 'new'.
        
        Only counts items that actually changed state or had errors cleared.
        
        Returns:
            The number of items reset.
        """
        with self._lock:
            count = 0
            terminal_states = {"uploaded", "success", "skipped_old", "skipped_ai", "skipped_removed", "skipped"}

            for bvid, data in self._state["videos"].items():
                current_status = data.get("status", "new")
                has_error = "error" in data

                if current_status not in terminal_states:
                    if current_status == "new" and not has_error:
                        continue
                    data["status"] = "new"
                    if has_error:
                        del data["error"]
                    count += 1
            
            if count > 0:
                self._save()
            return count

    def recover_interrupted_items(self) -> int:
        """Resume interrupted work from the latest durable artifact.

        Retries must not turn completed transcripts back into ``new`` items:
        their audio has already been deleted, so that would force a fresh
        download and transcription.  Prefer the most advanced file that still
        exists and only fall back to ``new`` when nothing durable remains.
        """
        with self._lock:
            terminal_states = {
                "uploaded", "success", "skipped_old", "skipped_ai",
                "skipped_removed", "skipped",
            }
            changed = 0

            for data in self._state["videos"].values():
                current_status = data.get("status", "new")
                if current_status in terminal_states:
                    continue

                def file_exists(field: str) -> bool:
                    value = data.get(field)
                    return bool(value) and Path(value).exists()

                if file_exists("epub_path"):
                    recovered_status = "success"
                elif file_exists("summary_md"):
                    recovered_status = "summarized"
                elif file_exists("corrected_md"):
                    recovered_status = "corrected"
                elif file_exists("transcript_md"):
                    recovered_status = "transcript_ready"
                elif file_exists("audio_path"):
                    recovered_status = "downloaded"
                else:
                    recovered_status = "new"

                if recovered_status != current_status or data.get("error"):
                    data["status"] = recovered_status
                    data.pop("error", None)
                    changed += 1

            if changed:
                self._save()
            return changed


def collect_epub_targets(state: "StateManager", force_all: bool = False) -> list[tuple[str, str]]:
    """Return (bvid, md_path) pairs that Step F should convert to EPUB.

    Automatic mode only emits items that finished summarization (status
    'summarized'/'summary_ready') — those are the only ones whose markdown
    carries the 核心摘要/要点列表 sections. It never falls back to a raw
    transcript, so an item that was transcribed but not summarized is skipped
    rather than shipped without a summary.

    force_all (the explicit --force-all override) regenerates every item that
    has any transcript, falling back to the raw transcript when no summary
    exists.
    """
    if force_all:
        bvids = state.get_all_bvids_with_transcripts()
    else:
        bvids = state.get_pending_items("summarized") + state.get_pending_items("summary_ready")

    targets: list[tuple[str, str]] = []
    for bvid in bvids:
        vs = state.get_video_state(bvid)
        md_path = vs.summary_md or (vs.transcript_md if force_all else None)
        if not md_path:
            continue
        targets.append((bvid, md_path))
    return targets
