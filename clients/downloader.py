"""Video downloader using yt-dlp with classified, bounded retries."""

from __future__ import annotations

import subprocess
import shutil
import sys
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Error types returned by download()
_ERR_OK        = "ok"
_ERR_SKIP      = "skip"       # Permanent: video unavailable/paid/deleted → mark skipped
_ERR_AUTH      = "auth"       # Cookie expired or invalid → warn user
_ERR_OUTDATED  = "outdated"   # yt-dlp extractor broken → report and retry safely
_ERR_TRANSIENT = "transient"  # Network hiccup → plain retry

# Patterns in yt-dlp stderr that map to each error type
_SKIP_PATTERNS = [
    "this video does not exist",
    "video unavailable",
    "已失效视频",
    "视频不见了",
    "大会员专享",
    "付费内容",
    "paid",
    "members only",
    "该内容已删除",
]
_AUTH_PATTERNS = [
    "login required",
    "需要登录",
    "http error 412",
    "please log in",
    "sign in",
]
_OUTDATED_PATTERNS = [
    "unable to extract",
    "unsupported url",
    "did not get any data blocks",
    "extractor",
    "no video formats",
    "requested format is not available",
]


def _classify_error(stderr: str) -> str:
    low = stderr.lower()
    for p in _SKIP_PATTERNS:
        if p in low:
            return _ERR_SKIP
    for p in _AUTH_PATTERNS:
        if p in low:
            return _ERR_AUTH
    for p in _OUTDATED_PATTERNS:
        if p in low:
            return _ERR_OUTDATED
    return _ERR_TRANSIENT


def _is_bilibili_url(url: str) -> bool:
    low = url.lower()
    return "bilibili.com" in low or "b23.tv" in low


class VideoDownloader:
    """Download videos/audio from Bilibili using yt-dlp."""

    def __init__(
        self,
        output_dir: Path,
        audio_only: bool = True,
        cookies_file: Optional[Path] = None,
        cookies_browser: Optional[str] = None,
        ffmpeg_location: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.audio_only = audio_only
        self.cookies_file = cookies_file
        self.cookies_browser = cookies_browser
        self.ffmpeg_location = ffmpeg_location

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ytdlp_path = shutil.which("yt-dlp")
        if not self.ytdlp_path:
            venv_scripts = Path(sys.executable).parent
            candidate = venv_scripts / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
            if candidate.exists():
                self.ytdlp_path = str(candidate)

        if not self.ytdlp_path:
            raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cmd(self, url: str, output_template: str) -> list[str]:
        cmd = [
            self.ytdlp_path,
            "--no-warnings",
            "--no-progress",
            "-o", output_template,
        ]
        if self.audio_only:
            # Bilibili may return HTTP 412 unless the request looks like it
            # comes from the site.  When only audio is needed for ASR, also
            # avoid downloading huge 4K/1080p streams just to extract audio.
            if _is_bilibili_url(url):
                cmd.extend(["-f", "16/bestaudio/best[height<=360]/worst"])
            cmd.extend(["-x", "--audio-format", "m4a", "--audio-quality", "0"])
        else:
            cmd.extend(["-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"])

        if _is_bilibili_url(url):
            cmd.extend([
                "--add-headers", "Origin:https://www.bilibili.com",
                "--add-headers", "Referer:https://www.bilibili.com/",
            ])

        if self.cookies_browser:
            cmd.extend(["--cookies-from-browser", self.cookies_browser])
        elif self.cookies_file and self.cookies_file.exists():
            cmd.extend(["--cookies", str(self.cookies_file)])

        if self.ffmpeg_location:
            cmd.extend(["--ffmpeg-location", self.ffmpeg_location])

        cmd.append(url)
        return cmd

    def _find_output(self, filename: str, ext: str) -> Optional[Path]:
        expected = self.output_dir / f"{filename}.{ext}"
        if expected.exists():
            return expected
        for f in self.output_dir.glob(f"{filename}.*"):
            if f.suffix in (".m4a", ".mp3", ".wav", ".mp4", ".mkv", ".webm"):
                return f
        return None

    def _update_ytdlp(self) -> bool:
        """Do not mutate the active environment during parallel downloads."""
        logger.warning(
            "  yt-dlp extractor appears outdated. Runtime self-update is disabled "
            "because replacing packages while downloads are active can corrupt the "
            "environment. Update yt-dlp before the next pipeline run."
        )
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, url: str, filename: str) -> tuple[Optional[Path], str]:
        """Download a single video/audio file.

        Returns:
            (path, error_type) where error_type is one of:
            'ok', 'skip', 'auth', 'outdated', 'transient'
        """
        ext = "m4a" if self.audio_only else "mp4"
        output_template = str(self.output_dir / f"{filename}.%(ext)s")
        cmd = self._build_cmd(url, output_template)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=600,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                error_type = _classify_error(stderr)
                logger.error(f"  yt-dlp [{error_type}]: {stderr}")
                return None, error_type

            path = self._find_output(filename, ext)
            if path:
                return path, _ERR_OK
            logger.error(f"  Output file not found after download: {filename}")
            return None, _ERR_TRANSIENT

        except subprocess.TimeoutExpired:
            logger.error(f"  Download timeout (10 min) for {url}")
            return None, _ERR_TRANSIENT
        except Exception as e:
            logger.error(f"  Download exception: {e}")
            return None, _ERR_TRANSIENT

    def download_with_retry(
        self,
        url: str,
        filename: str,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
    ) -> tuple[Optional[Path], str]:
        """Download with smart retry and self-healing.

        Recovery logic:
        - skip      → return immediately, no retry (caller should mark skipped)
        - auth      → log warning, no retry (caller should warn user)
        - outdated  → report that yt-dlp needs an out-of-band update, then retry
        - transient → retry up to max_retries with exponential backoff

        Returns:
            (path, final_error_type)
        """
        _outdated_retried = False

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = retry_base_delay * (2 ** (attempt - 1))  # 5s, 10s, 20s
                logger.info(f"  Retry {attempt}/{max_retries} (waiting {delay:.0f}s)...")
                time.sleep(delay)

            path, err = self.download(url, filename)

            if err == _ERR_OK:
                return path, _ERR_OK

            if err == _ERR_SKIP:
                logger.info("  Video unavailable/paid/deleted — permanently skipping.")
                return None, _ERR_SKIP

            if err == _ERR_AUTH:
                logger.warning(
                    "  Auth error — cookies may be expired. "
                    "Update sessdata/bili_jct in config.yaml."
                )
                return None, _ERR_AUTH

            if err == _ERR_OUTDATED and not _outdated_retried:
                updated = self._update_ytdlp()
                _outdated_retried = True
                if updated:
                    logger.info("  Retrying after yt-dlp update...")
                    path, err = self.download(url, filename)
                    if err == _ERR_OK:
                        return path, _ERR_OK
                    if err == _ERR_SKIP:
                        return None, _ERR_SKIP

            # transient (or outdated that still failed) → keep looping

        return None, _ERR_TRANSIENT
