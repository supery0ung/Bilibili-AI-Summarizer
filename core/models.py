"""Data models for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoInfo:
    """Video information from Bilibili."""
    
    bvid: str
    title: str
    url: str
    duration: int  # seconds
    up_name: str
    aid: Optional[int] = None
    cid: Optional[int] = None
    pubdate: Optional[int] = None # Unix timestamp
    youtube_url: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, item: dict) -> "VideoInfo":
        """Create VideoInfo from Bilibili API response item."""
        bvid = item.get("bvid", "")
        return cls(
            bvid=bvid,
            title=item.get("title", ""),
            url=f"https://www.bilibili.com/video/{bvid}",
            duration=item.get("duration", 0),
            up_name=item.get("owner", {}).get("name", ""),
            aid=item.get("aid"),
            cid=item.get("cid"),
            pubdate=item.get("pubdate"),
        )


@dataclass
class VideoState:
    """State of a video in the pipeline."""
    
    bvid: str
    # Status flow: new → downloading → downloaded → transcribing → transcript_ready 
    #            → correcting → corrected → summarizing → summarized → success
    status: str = "new"
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    pubdate: Optional[int] = None
    first_attempt: Optional[str] = None  # First processing attempt time
    last_attempt: Optional[str] = None
    audio_path: Optional[str] = None     # Downloaded audio file path
    transcript_md: Optional[str] = None  # Whisper transcript (Step C)
    corrected_md: Optional[str] = None   # Corrected transcript (Step D)
    summary_md: Optional[str] = None     # Legacy: Gemini summary (kept for compatibility)
    epub_path: Optional[str] = None
    title: Optional[str] = None          # Video title
    up_name: Optional[str] = None        # Author name
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "pubdate": self.pubdate,
            "first_attempt": self.first_attempt,
            "last_attempt": self.last_attempt,
            "audio_path": self.audio_path,
            "transcript_md": self.transcript_md,
            "corrected_md": self.corrected_md,
            "summary_md": self.summary_md,
            "epub_path": self.epub_path,
            "title": self.title,
            "up_name": self.up_name,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, bvid: str, data: dict) -> "VideoState":
        return cls(
            bvid=bvid,
            status=data.get("status", "new"),
            first_seen=data.get("first_seen"),
            last_seen=data.get("last_seen"),
            pubdate=data.get("pubdate"),
            first_attempt=data.get("first_attempt"),
            last_attempt=data.get("last_attempt"),
            audio_path=data.get("audio_path"),
            transcript_md=data.get("transcript_md"),
            corrected_md=data.get("corrected_md"),
            summary_md=data.get("summary_md"),
            epub_path=data.get("epub_path"),
            title=data.get("title"),
            up_name=data.get("up_name"),
            error=data.get("error"),
        )


@dataclass
class QueueItem:
    """An item in the processing queue."""
    
    bvid: str
    title: str
    url: str
    duration: int
    up_name: str
    pubdate: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "bvid": self.bvid,
            "title": self.title,
            "url": self.url,
            "duration": self.duration,
            "up_name": self.up_name,
            "pubdate": self.pubdate,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueueItem":
        return cls(
            bvid=data.get("bvid", ""),
            title=data.get("title", ""),
            url=data.get("url", ""),
            duration=data.get("duration", 0),
            up_name=data.get("up_name", ""),
            pubdate=data.get("pubdate"),
        )
    
    @classmethod
    def from_video_info(cls, video: VideoInfo) -> "QueueItem":
        return cls(
            bvid=video.bvid,
            title=video.title,
            url=video.url,
            duration=video.duration,
            up_name=video.up_name,
            pubdate=video.pubdate,
        )
