"""Video downloader using yt-dlp.

Supports downloading Bilibili videos, with option to extract audio only.
"""

from __future__ import annotations

import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional


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
        """Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded files.
            audio_only: If True, extract audio only (saves space).
            cookies_file: Optional path to cookies file for authenticated downloads.
            cookies_browser: Optional browser to read cookies from (chrome/edge/firefox).
            ffmpeg_location: Optional path to ffmpeg binary directory.
        """
        self.output_dir = Path(output_dir)
        self.audio_only = audio_only
        self.cookies_file = cookies_file
        self.cookies_browser = cookies_browser
        self.ffmpeg_location = ffmpeg_location
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find yt-dlp executable (check venv Scripts dir first on Windows)
        self.ytdlp_path = shutil.which("yt-dlp")
        if not self.ytdlp_path:
            # Try to find in the same directory as the Python interpreter (venv)
            venv_scripts = Path(sys.executable).parent
            ytdlp_candidate = venv_scripts / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
            if ytdlp_candidate.exists():
                self.ytdlp_path = str(ytdlp_candidate)
        
        if not self.ytdlp_path:
            raise RuntimeError(
                "yt-dlp not found. Please install it:\n"
                "  pip install yt-dlp\n"
                "  or: brew install yt-dlp"
            )
    
    def download(self, url: str, filename: str) -> Optional[Path]:
        """Download video/audio from URL.
        
        Args:
            url: Video URL (e.g., https://www.bilibili.com/video/BV1xxx)
            filename: Base filename (without extension) for the output.
            
        Returns:
            Path to downloaded file, or None if download failed.
        """
        # Build output template
        if self.audio_only:
            ext = "m4a"  # Best audio format for Bilibili
            output_template = str(self.output_dir / f"{filename}.%(ext)s")
        else:
            ext = "mp4"
            output_template = str(self.output_dir / f"{filename}.%(ext)s")
        
        # Build yt-dlp command
        cmd = [
            self.ytdlp_path,
            "--no-warnings",
            "--no-progress",
            "-o", output_template,
        ]
        
        if self.audio_only:
            cmd.extend([
                "-x",  # Extract audio
                "--audio-format", "m4a",
                "--audio-quality", "0",  # Best quality
            ])
        else:
            cmd.extend([
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            ])
        
        # Add cookies if provided
        if self.cookies_browser:
            cmd.extend(["--cookies-from-browser", self.cookies_browser])
        elif self.cookies_file and self.cookies_file.exists():
            cmd.extend(["--cookies", str(self.cookies_file)])
        
        # Add ffmpeg location if provided
        if self.ffmpeg_location:
            cmd.extend(["--ffmpeg-location", self.ffmpeg_location])
        
        cmd.append(url)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,  # 10 minute timeout
            )
            
            if result.returncode != 0:
                print(f"yt-dlp error: {result.stderr}")
                return None
            
            # Find the downloaded file
            expected_path = self.output_dir / f"{filename}.{ext}"
            if expected_path.exists():
                return expected_path
            
            # Try to find any matching file (yt-dlp might use different extension)
            for f in self.output_dir.glob(f"{filename}.*"):
                if f.suffix in (".m4a", ".mp3", ".wav", ".mp4", ".mkv", ".webm"):
                    return f
            
            print(f"Downloaded file not found for {filename}")
            return None
            
        except subprocess.TimeoutExpired:
            print(f"Download timeout for {url}")
            return None
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    def download_with_retry(
        self,
        url: str,
        filename: str,
        max_retries: int = 2,
    ) -> Optional[Path]:
        """Download with retry logic.
        
        Args:
            url: Video URL.
            filename: Base filename.
            max_retries: Maximum retry attempts.
            
        Returns:
            Path to downloaded file, or None if all attempts failed.
        """
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"  Retry attempt {attempt}/{max_retries}...")
            
            result = self.download(url, filename)
            if result:
                return result
        
        return None
