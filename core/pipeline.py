"""Main pipeline orchestrator.

Updated for Whisper transcription workflow:
- Step A: Fetch + Filter + Build queue (unchanged)
- Step B: Download video + Whisper transcribe (NEW)
- Step C: Generate EPUB (unchanged)
- Step D: Upload to WeChat Reading (unchanged)
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from clients.bilibili_api_client import BilibiliClient
from clients.downloader import VideoDownloader
from clients.qwen_asr_client import Qwen3ASRClient
from clients.weread_browser import WeReadBrowserClient
from clients.ollama_client import OllamaClient, build_final_markdown
from .models import VideoInfo, QueueItem
from .filter import VideoFilter
from .state import StateManager, now_iso, collect_epub_targets
from .step_downloader import StepDownloader
from .step_asr import StepASR
from .step_llm import StepLLM
from utils import convert_md_to_epub, safe_filename, get_logger


def today_ymd() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def now_iso() -> str:
    """Return current time in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


logger = get_logger("pipeline")


class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(
        self,
        config_path: Path,
        headless: bool = False,
    ):
        self.root = config_path.parent
        self.config = self._load_config(config_path)
        self.headless = headless
        self.logger = logger
        
        # Initialize paths
        self.output_dir = self.root / self.config.get("output", {}).get("transcripts_dir", "output/transcripts")
        self.media_dir = self.root / self.config.get("download", {}).get("output_dir", "output/media")
        self.epub_dir = self.root / self.config.get("output", {}).get("epub_dir", "output/epub")
        self.debug_dir = self.root / self.config.get("output", {}).get("debug_dir", "output/debug")
        self.state_file = self.root / self.config.get("output", {}).get("state_file", "output/pipeline_state.json")
        self.queue_file = self.root / self.config.get("output", {}).get("queue_file", "output/pipeline_queue.json")
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.epub_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.state = StateManager(self.state_file)
        
        # Lazy-load clients
        self._bilibili: Optional[BilibiliClient] = None
        self._downloader: Optional[VideoDownloader] = None
        self._asr_client: Optional[Any] = None
        self._ollama: Optional[OllamaClient] = None
        self._filter: Optional[VideoFilter] = None
    
    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                "Please copy config.example.yaml to config.yaml and fill in your credentials."
            )
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    def _save_queue(self, queue: list[QueueItem], run_id: str | None = None) -> None:
        self._atomic_write_json(
            self.queue_file,
            {
                "generated_at": now_iso(),
                "run_id": run_id,
                "queue": [item.to_dict() for item in queue],
            },
        )

    def _queue_item_from_state(self, bvid: str, data: dict[str, Any]) -> QueueItem:
        """Reconstruct enough queue metadata for downstream resume steps."""
        return QueueItem(
            bvid=bvid,
            title=data.get("title") or bvid,
            url=f"https://www.bilibili.com/video/{bvid}",
            duration=int(data.get("duration") or 0),
            up_name=data.get("up_name") or "",
            pubdate=data.get("pubdate"),
            is_manual=bool(data.get("is_manual", False)),
        )

    def _build_downstream_queue(
        self,
        preferred_items: list[QueueItem],
        statuses: set[str],
        max_items: int,
    ) -> list[QueueItem]:
        """Build a queue for LLM stages from current items plus saved backlog.

        Step D/E are queue-driven, so a daily run that replaces the queue with
        new Watch Later items can strand older transcript_ready/corrected work.
        Prefer items touched in this run, then fill from persisted state.
        """
        queue: list[QueueItem] = []
        seen: set[str] = set()

        def add(item: QueueItem) -> None:
            if item.bvid in seen or len(queue) >= max_items:
                return
            if self.state.get_status(item.bvid) not in statuses:
                return
            queue.append(item)
            seen.add(item.bvid)

        for item in preferred_items:
            add(item)

        videos = getattr(self.state, "_state", {}).get("videos", {})
        backlog = sorted(
            videos.items(),
            key=lambda row: (row[1].get("last_attempt") or "", row[0]),
            reverse=True,
        )
        for bvid, data in backlog:
            if len(queue) >= max_items:
                break
            if data.get("status") not in statuses or bvid in seen:
                continue
            queue.append(self._queue_item_from_state(bvid, data))
            seen.add(bvid)

        return queue

    def _run_downstream_backlog(
        self,
        *,
        run_id: str,
        preferred_items: list[QueueItem],
        max_items: int,
        upload: bool,
        stage_name: str,
    ) -> set[str]:
        """Run correction, summarization, EPUB, and upload for ready backlog."""
        touched_bvids: set[str] = set()

        self.logger.info(f"[{stage_name}] Downstream backlog: correcting/summarizing...")
        correct_queue = self._build_downstream_queue(
            preferred_items,
            {"transcript_ready", "correcting"},
            max_items,
        )
        if correct_queue:
            self.logger.info(f"Queued {len(correct_queue)} transcript backlog item(s) for correction")
            self._save_queue(correct_queue, run_id=f"{run_id}-{stage_name}-correct")
            _, d_bvids = self.run_step_d_correct(max_items=max_items, unload_after=False)
            touched_bvids.update(d_bvids)

        summarize_queue = self._build_downstream_queue(
            preferred_items,
            {"corrected", "summarizing"},
            max_items,
        )
        if summarize_queue:
            self.logger.info(f"Queued {len(summarize_queue)} corrected backlog item(s) for summarization")
            self._save_queue(summarize_queue, run_id=f"{run_id}-{stage_name}-summarize")
            _, e_bvids = self.run_step_e_summarize(
                max_items=max_items,
                unload_after=False,
            )
            touched_bvids.update(e_bvids)

        if hasattr(self.ollama, "unload_model"):
            self.ollama.unload_model()

        self.logger.info(f"[{stage_name}] Downstream backlog: generating EPUBs...")
        preferred_bvids = [
            item.bvid
            for item in preferred_items
            if self.state.get_status(item.bvid) in {"summarized", "success"}
        ]
        target_bvids = list(dict.fromkeys(preferred_bvids + list(touched_bvids)))
        if target_bvids:
            _, f_bvids = self.run_step_f_epub(target_bvids=target_bvids)
            touched_bvids.update(f_bvids)

        if upload:
            success_backlog = (
                self.state.get_pending_items("success")
                if hasattr(self.state, "get_pending_items")
                else []
            )
            upload_scope = list(dict.fromkeys(target_bvids + success_backlog))
            if not upload_scope:
                return touched_bvids
            self.logger.info(f"[{stage_name}] Downstream backlog: uploading...")
            priority_list = list(dict.fromkeys(list(touched_bvids) + upload_scope))
            upload_stats = self.run_step_g_upload(
                max_items=max_items,
                priority_bvids=priority_list,
                only_bvids=upload_scope,
            )
            if upload_stats.get("uploaded"):
                touched_bvids.update(priority_list)

        return touched_bvids
    
    @property
    def bilibili(self) -> BilibiliClient:
        """Get or create Bilibili client."""
        if self._bilibili is None:
            bc = self.config.get("bilibili", {})
            self._bilibili = BilibiliClient(
                sessdata=bc.get("sessdata", ""),
                bili_jct=bc.get("bili_jct", ""),
                dedeuserid=bc.get("dedeuserid", ""),
                buvid3=bc.get("buvid3", ""),
            )
        return self._bilibili
    
    def _generate_cookies_file(self) -> Path | None:
        """Generate a Netscape cookies.txt from bilibili config credentials."""
        bc = self.config.get("bilibili", {})
        sessdata = bc.get("sessdata", "").strip()
        bili_jct = bc.get("bili_jct", "").strip()
        dedeuserid = bc.get("dedeuserid", "").strip()
        buvid3 = bc.get("buvid3", "").strip()
        if not sessdata:
            return None

        # Extract expiry from SESSDATA (format: value,expiry,suffix)
        import urllib.parse
        decoded = urllib.parse.unquote(sessdata)
        parts = decoded.split(",")
        try:
            expiry = int(parts[1]) if len(parts) >= 2 else 2147483647
        except ValueError:
            expiry = 2147483647

        cookies_path = self.root / "output" / "bilibili_cookies.txt"
        cookies_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Netscape HTTP Cookie File",
            f".bilibili.com\tTRUE\t/\tTRUE\t{expiry}\tSESSDATA\t{sessdata}",
            f".bilibili.com\tTRUE\t/\tFALSE\t{expiry}\tbili_jct\t{bili_jct}",
            f".bilibili.com\tTRUE\t/\tFALSE\t{expiry}\tDedeUserID\t{dedeuserid}",
            f".bilibili.com\tTRUE\t/\tFALSE\t{expiry}\tbuvid3\t{buvid3}",
        ]
        cookies_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return cookies_path

    @property
    def downloader(self) -> VideoDownloader:
        """Get or create video downloader."""
        if self._downloader is None:
            dc = self.config.get("download", {})
            cookies_browser = dc.get("cookies_browser")
            cookies_file = dc.get("cookies_file")
            # Auto-generate cookies.txt from bilibili config if no other cookie source set
            if not cookies_browser and not cookies_file:
                generated = self._generate_cookies_file()
                if generated:
                    self.logger.info(f"  ✓ Using auto-generated cookies from config: {generated.name}")
                    cookies_file = str(generated)
            self._downloader = VideoDownloader(
                output_dir=self.media_dir,
                audio_only=dc.get("audio_only", True),
                cookies_browser=cookies_browser,
                cookies_file=Path(cookies_file) if cookies_file else None,
                ffmpeg_location=dc.get("ffmpeg_location"),
            )
        return self._downloader
    
    @property
    def asr_client(self) -> Qwen3ASRClient:
        """Get or create the Qwen3-ASR client."""
        if self._asr_client is None:
            dc = self.config.get("download", {})  # Get ffmpeg_location
            qc = self.config.get("qwen3", {})
            # hf_token (for pyannote diarization) historically lived under the
            # whisperx config section; accept it there or under qwen3.
            hf_token = qc.get("hf_token") or self.config.get("whisperx", {}).get("hf_token")
            self._asr_client = Qwen3ASRClient(
                model_name=qc.get("model", "Qwen/Qwen3-ASR-1.7B"),
                device=qc.get("device", "cuda:0"),
                language=qc.get("language"),
                ffmpeg_location=dc.get("ffmpeg_location"),
                diarize=qc.get("diarize", False),
                hf_token=hf_token,
                min_speakers=qc.get("min_speakers", 1),
                max_speakers=qc.get("max_speakers", 5),
            )
        return self._asr_client

    @property
    def ollama(self) -> OllamaClient:
        """Get or create Ollama client for LLM processing."""
        if self._ollama is None:
            oc = self.config.get("ollama", {})
            self._ollama = OllamaClient(
                model=oc.get("model", "qwen3:8b"),
                base_url=oc.get("base_url", "http://localhost:11434"),
                prompts_dir=self.root / "prompts",
                correction_num_ctx=oc.get("correction_num_ctx", 12288),
                summary_num_ctx=oc.get("summary_num_ctx", 32768),
                keep_alive=oc.get("keep_alive", "30m"),
                hierarchical_summary_chars=oc.get(
                    "hierarchical_summary_chars", 18000
                ),
                summary_chunk_chars=oc.get("summary_chunk_chars", 12000),
            )
        return self._ollama
    
    @property
    def video_filter(self) -> VideoFilter:
        """Get or create video filter."""
        if self._filter is None:
            filters_path = self.root / "filters.yaml"
            self._filter = VideoFilter.from_yaml(filters_path)
        return self._filter

    def _extract_bvid(self, text: str) -> str | None:
        """Extract BVID from Bilibili URL or raw string."""
        if not text:
            return None
        # Support BV1xxxxxx format
        match = re.search(r"(BV[a-zA-Z0-9]{10})", text, re.IGNORECASE)
        return match.group(1) if match else None

    def run_step_a(self, manual_urls: list[str] | None = None) -> list[QueueItem]:
        """Step A: Fetch + Filter + Build queue.
        
        Args:
            manual_urls: Optional list of specific Bilibili URLs to process.
                         These bypass all filters.
        
        Returns:
            List of queued items.
        """
        self.logger.info("=== Step A: Fetch + Filter + Build Queue ===")
        
        # Check auth
        self.logger.info("Checking Bilibili authentication...")
        if not self.bilibili.check_auth():
            raise ValueError("Bilibili authentication failed. Please check your cookies in config.yaml")
        self.logger.info("✓ Authenticated")
        
        manual_videos = []
        # 1. Handle manual URLs (Bypass filters)
        if manual_urls:
            self.logger.info(f"Adding {len(manual_urls)} manual URLs...")
            for url in manual_urls:
                bvid = self._extract_bvid(url)
                if not bvid:
                    self.logger.warning(f"  ⚠ Could not extract BVID from: {url}")
                    continue
                
                info = self.bilibili.get_video_info(bvid)
                if not info:
                    self.logger.warning(f"  ⚠ Could not fetch info for BVID: {bvid}. Using fallback info.")
                    # Fallback: create basic Info so we can still process it
                    info = VideoInfo(
                        bvid=bvid,
                        title=f"Manual Video {bvid}",
                        url=url,
                        duration=0,
                    up_name="Manual Upload",
                    is_manual=True
                )
            
                if info:
                    info.is_manual = True
                    self.logger.info(f"  + Added (manual/forced): {info.title[:60]}")
                    
                    # Force status to 'new' to ensure it's picked up by the queue
                    # even if it was previously processed or skipped.
                    self.logger.info(f"    Manual Mode: Reseting status to 'new'")
                    self.state.update(bvid, status="new", error=None, is_manual=True)
                    
                    manual_videos.append(info)

            # If manual mode, skip watchlater and return early
            self.logger.info(f"Manual mode: skipping watchlater list and filters.")
            queue = self.state.build_queue(manual_videos, max_items=9999)
            
            # Save queue
            self._save_queue(queue)
            return queue
        self.logger.info("Fetching watchlater list...")
        watchlater_videos = self.bilibili.get_watchlater_list()
        self.logger.info(f"✓ Found {len(watchlater_videos)} videos in watchlater")

        # Sync state with current Watch Later:
        # - Videos 'new' in state but gone from Watch Later → skipped_removed
        # - Videos 'skipped_removed' that re-appeared in Watch Later → new
        watchlater_bvids = {v.bvid for v in watchlater_videos}
        removed_count = 0
        restored_count = 0
        for bvid, data in self.state._state["videos"].items():
            status = data.get("status")
            if status == "new" and bvid not in watchlater_bvids:
                self.state.update(bvid, status="skipped_removed")
                removed_count += 1
            elif status == "skipped_removed" and bvid in watchlater_bvids:
                self.state.update(bvid, status="new")
                restored_count += 1
        if removed_count:
            self.logger.info(f"  ↳ Skipped {removed_count} video(s) no longer in Watch Later")
        if restored_count:
            self.logger.info(f"  ↳ Restored {restored_count} video(s) re-added to Watch Later")

        # 3. Filter watchlater
        self.logger.info("Applying filters to watchlater list...")
        filtered = self.video_filter.filter_all(watchlater_videos)
        self.logger.info(f"✓ {len(filtered)} videos passed filters")
        
        # Combine: Manual videos (unfiltered) + Filtered watchlater
        all_videos = manual_videos.copy()
        manual_bvids = {v.bvid for v in manual_videos}
        for v in filtered:
            if v.bvid not in manual_bvids:
                all_videos.append(v)
        
        # Build queue (store all, limit at transcribe step)
        self.logger.info("Building processing queue...")
        queue = self.state.build_queue(
            all_videos,
            max_items=9999,  # Store all, limit at transcribe step
        )
        self.logger.info(f"✓ {len(queue)} videos queued for processing")
        
        # Save queue
        self._save_queue(queue)
        
        return queue
    
    def run_step_b_download(self, max_items: int | None = None) -> tuple[dict[str, int], list[str]]:
        """Step B: Download audio from videos."""
        return StepDownloader(self).run(max_items)

    def run_step_ba_ai_filter(self, max_items: int | None = None) -> dict[str, int]:
        """Step BA: Use AI to filter videos based on title and author.
        
        This runs after download but before transcription to save VRAM and time.
        
        Args:
            max_items: Maximum items to filter in this run.
            
        Returns:
            Statistics dict with counts.
        """
        if max_items is None:
            max_items = self.config.get("pipeline", {}).get("max_items_per_run", 20)
            
        self.logger.info(f"=== Step BA: AI Filtering (max {max_items}) ===")
        
        # Load queue
        if not self.queue_file.exists():
            self.logger.error("No queue file found. Run 'fetch' first.")
            return {"error": 1}
        
        queue_data = json.loads(self.queue_file.read_text(encoding="utf-8"))
        queue = [QueueItem.from_dict(item) for item in queue_data.get("queue", [])]
        
        stats = {
            "processed": 0,
            "kept": 0,
            "skipped": 0,
            "error": 0,
        }
        
        for item in queue:
            if stats["processed"] >= max_items:
                break
            
            status = self.state.get_status(item.bvid)
            
            # We filter new videos before download
            if status != "new":
                continue
                
            if item.is_manual:
                self.logger.info(f"AI Filtering: {item.title[:60]}... (Manual, BYPASSING)")
                stats["kept"] += 1
                continue
                
            self.logger.info(f"AI Filtering: {item.title[:60]}...")
            
            try:
                # Use Ollama to decide
                should_keep = self.ollama.should_filter(item.title, item.up_name)
                
                if should_keep:
                    self.logger.info("  ✓ Decision: KEEP")
                    # Stay in 'downloaded' so next step picks it up
                    stats["kept"] += 1
                else:
                    self.logger.info("  ✗ Decision: SKIP (AI Filtered)")
                    self.state.update(
                        item.bvid, 
                        status="skipped_ai", 
                        title=item.title, 
                        up_name=item.up_name,
                        last_attempt=now_iso()
                    )
                    stats["skipped"] += 1
                    
            except Exception as e:
                self.logger.warning(f"  ⚠ AI Filtering error: {e}")
                # Default to keep on error
                stats["kept"] += 1
                stats["error"] += 1
            
            stats["processed"] += 1
            
        self.logger.info(f"=== Step BA Complete ===")
        self.logger.info(f"Kept: {stats['kept']}")
        self.logger.info(f"Skipped: {stats['skipped']}")
        
        return stats
    
    def run_step_c_transcribe(self, max_items: int | None = None) -> tuple[dict[str, int], list[str]]:
        """Step C: Transcribe downloaded audio with Whisper."""
        return StepASR(self).run(max_items)
    
    def run_step_d_correct(
        self,
        max_items: int | None = None,
        unload_after: bool = True,
    ) -> tuple[dict[str, int], list[str]]:
        """Step D: Correct transcripts with LLM."""
        return StepLLM(self).run(
            max_items,
            mode="correct",
            unload_after=unload_after,
        )

    def run_step_e_summarize(
        self,
        max_items: int | None = None,
        unload_after: bool = True,
    ) -> tuple[dict[str, int], list[str]]:
        """Step E: Summarize corrected transcripts with LLM."""
        return StepLLM(self).run(
            max_items,
            mode="summarize",
            unload_after=unload_after,
        )
    
    def run_step_f_epub(
        self,
        force_all: bool = False,
        target_bvids: list[str] | None = None,
    ) -> tuple[dict[str, int], list[str]]:
        """Step F: Convert transcripts to EPUB.
        
        Args:
            force_all: If True, regenerate EPUBs for all items with transcripts.
            
        Returns:
            Tuple of (statistics dict, list of processed BVIDs).
        """
        self.logger.info("=== Step F: Generate EPUBs ===")

        # collect_epub_targets enforces the policy: automatic runs only emit
        # summarized items (which carry 核心摘要/要点列表); only force_all may fall
        # back to a raw transcript. This prevents shipping summary-less EPUBs when
        # Step E (queue-driven) is starved while Step F reads from state.
        targets = collect_epub_targets(self.state, force_all=force_all)
        if target_bvids is not None:
            allowed = set(target_bvids)
            targets = [(bvid, path) for bvid, path in targets if bvid in allowed]
        if force_all:
            self.logger.info(f"Force-regenerating {len(targets)} EPUBs...")
        else:
            self.logger.info(f"Found {len(targets)} summarized items ready for EPUB conversion")

        stats = {"converted": 0, "error": 0}
        processed_bvids = []

        for bvid, md_path in targets:
            video_state = self.state.get_video_state(bvid)

            md_file = Path(md_path)
            if not md_file.exists():
                self.logger.error(f"  ✗ Markdown file missing: {md_file}")
                stats["error"] += 1
                continue
            
            # Extract title from content (first # heading or first non-empty line)
            content = md_file.read_text(encoding="utf-8")
            title = None
            
            # 1. Try to find standard Markdown header
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            
            # 2. Fallback: First non-empty line
            if not title:
                for line in content.splitlines():
                    if line.strip():
                        title = line.strip()
                        break
            
            # 3. Final fallback: BVID
            extracted_title = title or bvid
            
            # Generate EPUB
            safe_title = safe_filename(extracted_title)
            new_epub_path = self.epub_dir / f"{safe_title}.epub"
            
            # Check for old BVID-named file to cleanup
            old_epub_path_str = video_state.epub_path
            if old_epub_path_str:
                old_path = Path(old_epub_path_str)
                if old_path.exists() and old_path != new_epub_path:
                    if old_path.name == f"{bvid}.epub":
                        self.logger.info(f"  Cleanup: Deleting old BVID-named file: {old_path.name}")
                        try:
                            old_path.unlink()
                        except:
                            pass

            # Prepare publication time string
            pub_time_str = ""
            if video_state.pubdate:
                from datetime import datetime
                dt = datetime.fromtimestamp(video_state.pubdate)
                pub_time_str = dt.strftime("%Y-%m-%d %H:%M")

            try:
                convert_md_to_epub(
                    md_file, 
                    new_epub_path, 
                    extracted_title, 
                    author=video_state.up_name or "",
                    pub_time=pub_time_str
                )
                # Update status and new path
                update_fields = {
                    "epub_path": str(new_epub_path),
                    "title": extracted_title,
                    "up_name": video_state.up_name  # Preserve up_name if already in state
                }
                if video_state.status in ("summarized", "corrected", "transcript_ready", "summary_ready"):
                    update_fields["status"] = "success"
                
                self.state.update(bvid, **update_fields)
                self.logger.info(f"  ✓ {new_epub_path.name}")
                stats["converted"] += 1
                processed_bvids.append(bvid)
            except Exception as e:
                self.logger.error(f"  ✗ Failed to create EPUB: {e}")
                stats["error"] += 1
        
        self.logger.info(f"=== Step F Complete ===")
        self.logger.info(f"Converted: {stats['converted']}")
        self.logger.info(f"Errors: {stats['error']}")
        
        return stats, processed_bvids
    
    def run_step_g_upload(
        self,
        max_items: int | None = None,
        priority_bvids: list[str] | None = None,
        only_bvids: list[str] | None = None,
    ) -> dict[str, int]:
        """Step G: Upload EPUBs to WeChat Reading.
        
        Args:
            max_items: Maximum items to upload in this run.
            priority_bvids: BVIDs to upload first (e.g. from current run).
            
        Returns:
            Statistics dict with counts.
        """
        if max_items is None:
            max_items = self.config.get("pipeline", {}).get("max_items_per_run", 20)
            
        self.logger.info(f"=== Step G: Upload to WeChat Reading (max {max_items}) ===")
        
        ready_to_upload = self.state.get_pending_items("success")
        if only_bvids is not None:
            allowed = set(only_bvids)
            ready_to_upload = [bvid for bvid in ready_to_upload if bvid in allowed]
        
        # Prioritize current-run items first
        if priority_bvids:
            priority_set = set(priority_bvids)
            current_run = [b for b in ready_to_upload if b in priority_set]
            backlog = [b for b in ready_to_upload if b not in priority_set]
            ready_to_upload = current_run + backlog
        
        self.logger.info(f"Found {len(ready_to_upload)} EPUBs ready for upload")
        
        # Initialize browser client
        weread = WeReadBrowserClient(headless=self.headless, output_dir=str(self.debug_dir))
        
        stats = {"uploaded": 0, "error": 0}
        processed_count = 0
        
        try:
            for bvid in ready_to_upload:
                if processed_count >= max_items:
                    break
                
                processed_count += 1
                video_state = self.state.get_video_state(bvid)
                if not video_state.epub_path:
                    continue
                
                epub_path = Path(video_state.epub_path)
                if not epub_path.exists():
                    self.logger.error(f"  ✗ EPUB missing: {epub_path}")
                    stats["error"] += 1
                    continue
                
                self.logger.info(f"Uploading: {epub_path.name}")
                
                # Internal retry for specific file
                success = False
                for attempt in range(2):
                    if attempt > 0:
                        self.logger.info(f"  Retrying upload (attempt {attempt + 1})...")
                        time.sleep(2)
                        
                    try:
                        if weread.upload_epub(str(epub_path)):
                            success = True
                            break
                    except Exception as e:
                        self.logger.warning(f"  Upload attempt crashed: {e}")
                        weread.close()
                        weread = WeReadBrowserClient(headless=self.headless, output_dir=str(self.debug_dir))
                        
                if success:
                    self.logger.info("  ✓ Upload success")
                    stats["uploaded"] += 1
                    self.state.update(bvid, status="uploaded", title=video_state.title, up_name=video_state.up_name)
                else:
                    self.logger.error("  ✗ Upload failed after retries")
                    stats["error"] += 1
        finally:
            weread.close()
            
        self.logger.info(f"=== Step G Complete ===")
        self.logger.info(f"Uploaded: {stats['uploaded']}")
        self.logger.info(f"Errors: {stats['error']}")
        
        return stats
    
    def run_all(
        self, 
        max_items: int | None = None, 
        upload: bool = False,
        manual_urls: list[str] | None = None
    ) -> dict[str, Any]:
        """Run complete pipeline in batch stages (Step A -> G).
        
        This batch approach is more memory-efficient for 12GB VRAM cards:
        1. Batch Download (Step B)
        2. Batch AI Filter (Step BA) -> Unload LLM
        3. Batch Transcribe (Step C) -> Unload ASR
        4. Batch Correct + Summarize (Step D, E) -> Unload LLM
        5. Batch EPUB generation (Step F)
        6. Batch Upload (Step G)
        
        Args:
            max_items: Maximum items to process in this run.
            upload: If True, also upload to WeChat Reading.
            manual_urls: Specific URLs to process (bypass filters).
            
        Returns:
            Combined statistics and processing summary.
        """
        if max_items is None:
            max_items = self.config.get("pipeline", {}).get("max_items_per_run", 20)
            
        results: dict[str, Any] = {
            "processed_bvids": [],
            "stats": {},
            "success_count": 0,
        }
        
        self.logger.info("="*60)
        self.logger.info(f"=== Starting Batch Pipeline (max_items={max_items}) ===")
        if manual_urls:
            self.logger.info(f"Manual URLs: {len(manual_urls)}")
            # If manual mode, restrict max_items to the number of URLs to prevent processing backlog
            manual_count = len(manual_urls)
            if max_items is None or max_items > manual_count:
                self.logger.info(f"Restricting max_items to {manual_count} for manual mode.")
                max_items = manual_count
        self.logger.info("="*60)

        # --- Step A: Fetch ---
        queue = self.run_step_a(manual_urls=manual_urls)
        if not queue:
            self.logger.info("No new items to process.")
            return results
        
        # Limit queue to max_items for the whole pipeline run
        current_batch = queue[:max_items]
        run_id = f"{today_ymd()}-{uuid.uuid4().hex[:8]}"
        self._save_queue(current_batch, run_id=run_id)
        results["queued"] = len(current_batch)
        results["run_id"] = run_id
        
        # Track all BVIDs touched/successfully processed in this run
        all_touched_bvids = set()
        
        # --- Step BA: AI Filter ---
        self.logger.info("[Stage 1/6] Batch AI Filtering...")
        # run_step_ba_ai_filter returns stats dict
        self.run_step_ba_ai_filter(max_items=max_items)

        # Push already-transcribed backlog forward before new ASR work. Long
        # videos can consume the whole scheduled window; existing transcripts
        # should still become EPUBs/uploads first.
        all_touched_bvids.update(
            self._run_downstream_backlog(
                run_id=run_id,
                preferred_items=[],
                max_items=max_items,
                upload=upload,
                stage_name="Stage 1.5/6",
            )
        )
            
        # --- Step B: Download ---
        self.logger.info("[Stage 2/6] Batch Downloading...")
        _, b_bvids = self.run_step_b_download(max_items=max_items)
        all_touched_bvids.update(b_bvids)
        
        # --- Step C: Transcribe ---
        self.logger.info("[Stage 3/6] Batch Transcribing (ASR)...")
        _, c_bvids = self.run_step_c_transcribe(max_items=max_items)
        all_touched_bvids.update(c_bvids)
        
        # Release GPU memory used by Whisper/ASR
        if hasattr(self.asr_client, "unload_model"):
            self.asr_client.unload_model()
            
        # --- Step D-G: Correct, summarize, EPUB, upload for newly transcribed items ---
        self.logger.info("[Stage 4-6/6] Batch Downstream Processing...")
        all_touched_bvids.update(
            self._run_downstream_backlog(
                run_id=run_id,
                preferred_items=current_batch,
                max_items=max_items,
                upload=upload,
                stage_name="Stage 4-6/6",
            )
        )
        
        # Collect final status
        self.logger.info("="*60)
        self.logger.info("=== Batch Pipeline Complete ===")
        self.logger.info("="*60)
        
        # Count successes from state
        processed_bvids = [item.bvid for item in current_batch]
        for bvid in processed_bvids:
            status = self.state.get_status(bvid)
            if status in ("success", "uploaded"):
                results["success_count"] += 1
            results["processed_bvids"].append(bvid)
            
        self.logger.info(f"Items in this run: {len(current_batch)}")
        self.logger.info(f"Final Successes: {results['success_count']}")
        
        # Clean up debug files
        self.cleanup_debug_files()
            
        return results

    def cleanup_debug_files(self):
        """Delete temporary PNG screenshots from debug_dir."""
        if not self.debug_dir or not self.debug_dir.exists():
            return
            
        self.logger.info(f"Cleaning up debug files in {self.debug_dir}...")
        files = list(self.debug_dir.glob("*.png"))
        count = 0
        for f in files:
            try:
                f.unlink()
                count += 1
            except Exception as e:
                self.logger.warning(f"  Failed to delete {f.name}: {e}")
        
        if count > 0:
            self.logger.info(f"  ✓ Deleted {count} temporary screenshots.")
    
    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Status information.
        """
        stats = self.state.get_summary_stats()
        
        # Count EPUBs
        epub_count = len(list(self.epub_dir.glob("*.epub")))
        
        # Count media files
        media_count = len(list(self.media_dir.glob("*.*")))
        
        return {
            "video_stats": stats,
            "epub_count": epub_count,
            "media_count": media_count,
            "state_file": str(self.state_file),
            "queue_file": str(self.queue_file),
        }
