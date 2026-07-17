"""Bilibili API client."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from typing import Optional
from urllib3.util.retry import Retry

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
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.request_timeout = (10, 30)
        self.session.cookies.set("SESSDATA", sessdata, domain=".bilibili.com")
        self.session.cookies.set("bili_jct", bili_jct, domain=".bilibili.com")
        self.session.cookies.set("DedeUserID", dedeuserid, domain=".bilibili.com")
        self.session.cookies.set("BUVID3", buvid3, domain=".bilibili.com")
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
        })
        self._wbi_keys: Optional[tuple[str, str]] = None
    
    def get_watchlater_list(self) -> list[VideoInfo]:
        """Fetch the watchlater (稀后再看) list.
        
        Returns:
            List of VideoInfo objects.
        
        Raises:
            requests.RequestException: On network errors.
            ValueError: On API errors.
        """
        resp = self.session.get(self.WATCHLATER_URL, timeout=self.request_timeout)
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

    def _get_wbi_keys(self) -> tuple[str, str]:
        """Fetch img_key and sub_key for WBI signing."""
        if self._wbi_keys:
            return self._wbi_keys
        
        # Must use the same session/cookies to get keys for this user
        resp = self.session.get(
            f"{self.API_BASE}/x/web-interface/nav", timeout=self.request_timeout
        )
        resp.raise_for_status()
        data = resp.json()
        
        wbi_img = data.get("data", {}).get("wbi_img", {})
        img_url = wbi_img.get("img_url", "")
        sub_url = wbi_img.get("sub_url", "")
        
        if not img_url or not sub_url:
            from utils import get_logger
            get_logger("bilibili").error(f"Could not find WBI keys in nav response: {data}")
            raise ValueError("Could not find WBI keys in nav response")
            
        img_key = img_url.split("/")[-1].split(".")[0]
        sub_key = sub_url.split("/")[-1].split(".")[0]
        
        from utils import get_logger
        get_logger("bilibili").info(f"Fetched WBI keys: img_key={img_key[:4]}..., sub_key={sub_key[:4]}...")
        
        self._wbi_keys = (img_key, sub_key)
        return self._wbi_keys
    
    def get_video_info(self, bvid: str) -> Optional[VideoInfo]:
        """Get detailed info for a single video.
        
        Args:
            bvid: The video's BV ID.
            
        Returns:
            VideoInfo object or None if not found.
        """
        params = {"bvid": bvid}
        try:
            img_key, sub_key = self._get_wbi_keys()
            from utils.bilibili_wbi_signer import enc_wbi
            params = enc_wbi(params, img_key, sub_key)
        except Exception as e:
            from utils import get_logger
            get_logger("bilibili").warning(f"Failed to sign WBI: {e}")

        resp = self.session.get(
            self.VIDEO_INFO_URL, params=params, timeout=self.request_timeout
        )
        resp.raise_for_status()
        
        data = resp.json()
        if data.get("code") != 0:
            from utils import get_logger
            get_logger("bilibili").warning(f"Bilibili API error: code={data.get('code')}, msg={data.get('message')}, bvid={bvid}")
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
            resp = self.session.get(
                f"{self.API_BASE}/x/web-interface/nav", timeout=self.request_timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("code") == 0 and data.get("data", {}).get("isLogin", False)
        except Exception:
            return False
