"""Video filtering logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .models import VideoInfo


class VideoFilter:
    """Filter videos based on configurable rules."""
    
    def __init__(self, config: dict[str, Any]):
        self.min_seconds = int(config.get("min_seconds", 0))
        self.up_deny = [
            str(x).lower() 
            for x in (config.get("up_deny_contains") or []) 
            if str(x).strip()
        ]
        self.title_deny_patterns: list[re.Pattern] = []
        for pat in (config.get("title_deny_regex") or []):
            if pat and str(pat).strip():
                try:
                    self.title_deny_patterns.append(
                        re.compile(str(pat), flags=re.IGNORECASE)
                    )
                except re.error:
                    continue
        self.keep_when_uncertain = bool(config.get("keep_when_uncertain", True))
    
    @classmethod
    def from_yaml(cls, path: Path) -> "VideoFilter":
        """Load filter from YAML file."""
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(config)
    
    def should_keep(self, video: VideoInfo) -> bool:
        """Check if video passes all filter rules."""
        # Check duration
        if self.min_seconds and video.duration < self.min_seconds:
            return False
        
        # Check UP name deny list
        up_lower = video.up_name.lower() if video.up_name else ""
        if up_lower and any(deny in up_lower for deny in self.up_deny):
            return False
        
        # Check title deny patterns
        if video.title and any(p.search(video.title) for p in self.title_deny_patterns):
            return False
        
        # Check required fields
        if not video.bvid or not video.url:
            return self.keep_when_uncertain
        
        return True
    
    def filter_all(self, videos: list[VideoInfo]) -> list[VideoInfo]:
        """Filter a list of videos."""
        return [v for v in videos if self.should_keep(v)]
