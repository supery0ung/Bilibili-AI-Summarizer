"""Download step for the pipeline."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Optional

from .base_step import BaseStep
from .models import QueueItem
from .state import now_iso
from utils import safe_filename

class StepDownloader(BaseStep):
    """Step B: Download audio from videos."""
    
    def run(self, max_items: Optional[int] = None) -> dict[str, int]:
        max_items = self.get_max_items(max_items)
        self.logger.info(f"=== Step B: Download Audio (max {max_items}) ===")
        
        queue = self.load_queue()
        if not queue:
            return {"error": 1}
        
        stats = {
            "processed": 0,
            "downloaded": 0,
            "already_downloaded": 0,
            "download_failed": 0,
        }
        
        # Filter items that need downloading
        to_download = []
        for item in queue:
            if stats["processed"] >= max_items:
                break
            
            status = self.state.get_status(item.bvid)
            if status not in ("new", "downloading"):
                if status == "downloaded":
                    stats["already_downloaded"] += 1
                continue
            
            video_state = self.state.get_video_state(item.bvid)
            if video_state.audio_path and Path(video_state.audio_path).exists():
                self.logger.info(f"✓ Already downloaded: {item.title[:50]}")
                self.state.update(item.bvid, status="downloaded")
                stats["already_downloaded"] += 1
                stats["processed"] += 1
                continue
                
            to_download.append(item)
            stats["processed"] += 1

        if not to_download:
            self.logger.info("No new audio files to download.")
            return stats, []

        # Use ThreadPoolExecutor for parallel downloads
        max_workers = self.config.get("download", {}).get("max_parallel", 3)
        self.logger.info(f"Starting parallel downloads (workers={max_workers})...")
        
        processed_bvids = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(self._download_item, item): item 
                for item in to_download
            }
            
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    success = future.result()
                    if success:
                        stats["downloaded"] += 1
                        processed_bvids.append(item.bvid)
                    else:
                        stats["download_failed"] += 1
                except Exception as e:
                    self.logger.error(f"Error downloading {item.bvid}: {e}")
                    stats["download_failed"] += 1
        
        self.logger.info(f"=== Step B Complete ===")
        self.logger.info(f"Downloaded: {stats['downloaded']}, Already: {stats['already_downloaded']}, Failed: {stats['download_failed']}")
        return stats, processed_bvids

    def _download_item(self, item: QueueItem) -> bool:
        """Download a single item and update state."""
        self.logger.info(f"Downloading: {item.title[:60]}...")
        self.state.update(item.bvid, status="downloading", title=item.title, up_name=item.up_name, last_attempt=now_iso())
        
        safe_name = safe_filename(f"{item.bvid}_{item.title[:50]}")
        audio_path = self.pipeline.downloader.download_with_retry(item.url, safe_name)
        
        if audio_path:
            self.state.update(
                item.bvid,
                status="downloaded",
                title=item.title,
                up_name=item.up_name,
                audio_path=str(audio_path),
            )
            self.logger.info(f"  ✓ Saved: {audio_path.name}")
            return True
        else:
            self.state.update(
                item.bvid,
                status="error",
                error="Download failed",
                last_attempt=now_iso(),
            )
            self.logger.error(f"  ✗ Download failed: {item.title[:50]}")
            return False
