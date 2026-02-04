"""Bilibili API client."""

from __future__ import annotations

import requests
from typing import Optional

from core.models import VideoInfo


class BilibiliClient:
    """Client for Bilibili API."""
    
    API_BASE = "https://api.bilibili.com"
    WATCHLATER_URL = f"{API_BASE}/x/v2/history/toview"
    VIDEO_INFO_URL = f"{API_BASE}/x/web-interface/view"
    
    def __init__(
        self,
        sessdata: str,
        bili_jct: str,
        dedeuserid: str,
        buvid3: str,
    ):
        self.session = requests.Session()
        self.session.cookies.set("SESSDATA", sessdata, domain=".bilibili.com")
        self.session.cookies.set("bili_jct", bili_jct, domain=".bilibili.com")
        self.session.cookies.set("DedeUserID", dedeuserid, domain=".bilibili.com")
        self.session.cookies.set("BUVID3", buvid3, domain=".bilibili.com")
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
        })
    
    def get_watchlater_list(self) -> list[VideoInfo]:
        """Fetch the watchlater (稀后再看) list.
        
        Returns:
            List of VideoInfo objects.
        
        Raises:
            requests.RequestException: On network errors.
            ValueError: On API errors.
        """
        resp = self.session.get(self.WATCHLATER_URL)
        resp.raise_for_status()
        
        data = resp.json()
        if data.get("code") != 0:
            raise ValueError(f"Bilibili API error: {data.get('message', 'Unknown error')}")
        
        items = data.get("data", {}).get("list", [])
        videos = []
        
        for item in items:
            try:
                video = VideoInfo.from_api_response(item)
                if video.bvid:
                    videos.append(video)
            except Exception:
                continue
        
        return videos
    
    def get_video_info(self, bvid: str) -> Optional[VideoInfo]:
        """Get detailed info for a single video.
        
        Args:
            bvid: The video's BV ID.
            
        Returns:
            VideoInfo object or None if not found.
        """
        resp = self.session.get(self.VIDEO_INFO_URL, params={"bvid": bvid})
        resp.raise_for_status()
        
        data = resp.json()
        if data.get("code") != 0:
            return None
        
        video_data = data.get("data", {})
        return VideoInfo(
            bvid=video_data.get("bvid", bvid),
            title=video_data.get("title", ""),
            url=f"https://www.bilibili.com/video/{bvid}",
            duration=video_data.get("duration", 0),
            up_name=video_data.get("owner", {}).get("name", ""),
            aid=video_data.get("aid"),
            cid=video_data.get("cid"),
            pubdate=video_data.get("pubdate"),
        )
    
    def check_auth(self) -> bool:
        """Check if authentication is valid.
        
        Returns:
            True if authenticated, False otherwise.
        """
        try:
            resp = self.session.get(f"{self.API_BASE}/x/web-interface/nav")
            resp.raise_for_status()
            data = resp.json()
            return data.get("code") == 0 and data.get("data", {}).get("isLogin", False)
        except Exception:
            return False
