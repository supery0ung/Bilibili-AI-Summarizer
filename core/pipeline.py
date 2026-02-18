"""Main pipeline orchestrator.

Updated for Whisper transcription workflow:
- Step A: Fetch + Filter + Build queue (unchanged)
- Step B: Download video + Whisper transcribe (NEW)
- Step C: Generate EPUB (unchanged)
- Step D: Upload to WeChat Reading (unchanged)
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from clients.bilibili import BilibiliClient
from clients.downloader import VideoDownloader
from clients.legacy.whisper_client import WhisperClient
from clients.legacy.whisperx_client import WhisperXClient
from clients.qwen_asr_client import Qwen3ASRClient
from clients.weread_browser import WeReadBrowserClient
from clients.ollama_client import OllamaClient, build_final_markdown
from .models import VideoInfo, QueueItem
from .filter import VideoFilter
from .state import StateManager, now_iso
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
        self.state_file = self.root / self.config.get("output", {}).get("state_file", "output/pipeline_state.json")
        self.queue_file = self.root / self.config.get("output", {}).get("queue_file", "output/pipeline_queue.json")
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.epub_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    @property
    def downloader(self) -> VideoDownloader:
        """Get or create video downloader."""
        if self._downloader is None:
            dc = self.config.get("download", {})
            self._downloader = VideoDownloader(
                output_dir=self.media_dir,
                audio_only=dc.get("audio_only", True),
                cookies_browser=dc.get("cookies_browser"),
                ffmpeg_location=dc.get("ffmpeg_location"),
            )
        return self._downloader
    
    @property
    def asr_client(self) -> WhisperClient | WhisperXClient | Qwen3ASRClient:
        """Get or create the configured ASR client."""
        if self._asr_client is None:
            engine = self.config.get("asr_engine", "whisper")
            dc = self.config.get("download", {})  # Get ffmpeg_location
            whisperx_cfg = self.config.get("whisperx", {})
            
            if engine == "qwen3":
                qc = self.config.get("qwen3", {})
                self._asr_client = Qwen3ASRClient(
                    model_name=qc.get("model", "Qwen/Qwen3-ASR-1.7B"),
                    device=qc.get("device", "cuda:0"),
                    language=qc.get("language"),
                    ffmpeg_location=dc.get("ffmpeg_location"),
                    diarize=qc.get("diarize", False),
                    hf_token=whisperx_cfg.get("hf_token"),
                    min_speakers=qc.get("min_speakers", 1),
                    max_speakers=qc.get("max_speakers", 5),
                )
            elif whisperx_cfg.get("enabled", False):
                # Use WhisperX with speaker diarization
                wc = self.config.get("whisper", {})
                self._asr_client = WhisperXClient(
                    model=wc.get("model", "large-v3"),
                    device=wc.get("device", "auto"),
                    compute_type="float16",
                    hf_token=whisperx_cfg.get("hf_token"),
                    language=wc.get("language"),
                    min_speakers=whisperx_cfg.get("min_speakers", 1),
                    max_speakers=whisperx_cfg.get("max_speakers", 5),
                )
            else:
                # Default to Whisper - all settings from config
                wc = self.config.get("whisper", {})
                self._asr_client = WhisperClient(
                    model_name=wc.get("model", "large-v3"),
                    device=wc.get("device", "auto"),
                    language=wc.get("language"),  # None = auto-detect
                    ffmpeg_location=dc.get("ffmpeg_location"),
                )
        return self._asr_client
    
    @property
    def whisper(self) -> WhisperClient:
        """Deprecated: Use asr_client instead. Kept for backward compatibility."""
        return self.asr_client if isinstance(self.asr_client, WhisperClient) else self.asr_client
    
    @property
    def ollama(self) -> OllamaClient:
        """Get or create Ollama client for LLM processing."""
        if self._ollama is None:
            oc = self.config.get("ollama", {})
            self._ollama = OllamaClient(
                model=oc.get("model", "qwen3:8b"),
                base_url=oc.get("base_url", "http://localhost:11434"),
                prompts_dir=self.root / "prompts",
            )
        return self._ollama
    
    @property
    def video_filter(self) -> VideoFilter:
        """Get or create video filter."""
        if self._filter is None:
            filters_path = self.root / "filters.yaml"
            self._filter = VideoFilter.from_yaml(filters_path)
        return self._filter
    
    def run_step_a(self) -> list[QueueItem]:
        """Step A: Fetch + Filter + Build queue.
        
        Returns:
            List of queued items.
        """
        self.logger.info("=== Step A: Fetch + Filter + Build Queue ===")
        
        # Check auth
        self.logger.info("Checking Bilibili authentication...")
        if not self.bilibili.check_auth():
            raise ValueError("Bilibili authentication failed. Please check your cookies in config.yaml")
        self.logger.info("✓ Authenticated")
        
        # Fetch watchlater list
        self.logger.info("Fetching watchlater list...")
        videos = self.bilibili.get_watchlater_list()
        self.logger.info(f"✓ Found {len(videos)} videos in watchlater")
        
        # Filter
        self.logger.info("Applying filters...")
        filtered = self.video_filter.filter_all(videos)
        self.logger.info(f"✓ {len(filtered)} videos passed filters")
        
        # Build queue (store all filtered videos, limit at transcribe step)
        self.logger.info("Building processing queue...")
        queue = self.state.build_queue(
            filtered,
            max_items=9999,  # Store all, limit at transcribe step
        )
        self.logger.info(f"✓ {len(queue)} videos queued for processing")
        
        # Save queue
        queue_data = {
            "generated_at": now_iso(),
            "queue": [item.to_dict() for item in queue],
        }
        self.queue_file.write_text(
            json.dumps(queue_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
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
    
    def run_step_d_correct(self, max_items: int | None = None) -> tuple[dict[str, int], list[str]]:
        """Step D: Correct transcripts with LLM."""
        return StepLLM(self).run(max_items)

    def run_step_e_summarize(self, max_items: int | None = None) -> tuple[dict[str, int], list[str]]:
        """Step E: Summarize corrected transcripts with LLM."""
        return StepLLM(self).run(max_items)
    
    def run_steps_b_to_e(self, max_items: int | None = None, skip_llm: bool = False) -> dict[str, int]:
        """Run Steps B → E: Download + Transcribe + Correct + Summarize.
        
        Args:
            max_items: Maximum items to process in this run.
            skip_llm: If True, skip the LLM processing steps (Steps D & E).
            
        Returns:
            Statistics dict with counts.
        """
        if max_items is None:
            max_items = self.config.get("pipeline", {}).get("max_items_per_run", 20)
        # Run all steps
        download_stats, b_bvids = self.run_step_b_download(max_items)
        transcribe_stats, c_bvids = self.run_step_c_transcribe(max_items)
        
        correct_stats = {"corrected": 0}
        summarize_stats = {"summarized": 0}
        
        if not skip_llm:
            correct_stats, d_bvids = self.run_step_d_correct(max_items)
            summarize_stats, e_bvids = self.run_step_e_summarize(max_items)
        
        # Combine stats
        return {
            "processed": download_stats.get("processed", 0) + transcribe_stats.get("processed", 0),
            "transcript_ready": transcribe_stats.get("transcribed", 0),
            "corrected": correct_stats.get("corrected", 0),
            "summarized": summarize_stats.get("summarized", 0),
            "download_failed": download_stats.get("download_failed", 0),
            "error": (download_stats.get("error", 0) + transcribe_stats.get("error", 0) + 
                     correct_stats.get("error", 0) + summarize_stats.get("error", 0)),
        }
    
    def run_step_f_epub(self, force_all: bool = False) -> tuple[dict[str, int], list[str]]:
        """Step F: Convert transcripts to EPUB.
        
        Args:
            force_all: If True, regenerate EPUBs for all items with transcripts.
            
        Returns:
            Tuple of (statistics dict, list of processed BVIDs).
        """
        self.logger.info("=== Step F: Generate EPUBs ===")
        
        if force_all:
            pending = self.state.get_all_bvids_with_transcripts()
            self.logger.info(f"Force-regenerating {len(pending)} EPUBs...")
        else:
            # Include all ready states
            pending = self.state.get_pending_items("summarized")  # After Step E
            pending.extend(self.state.get_pending_items("corrected"))  # Can also use corrected
            pending.extend(self.state.get_pending_items("transcript_ready"))  # Raw transcript
            pending.extend(self.state.get_pending_items("summary_ready"))  # Legacy
            self.logger.info(f"Found {len(pending)} transcripts ready for EPUB conversion")
        
        stats = {"converted": 0, "error": 0}
        processed_bvids = []
        
        for bvid in pending:
            video_state = self.state.get_video_state(bvid)
            
            # Use summary_md if available, fallback to transcript_md
            md_path = video_state.summary_md or video_state.transcript_md
            if not md_path:
                continue
            
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
    
    def run_step_g_upload(self, max_items: int | None = None, priority_bvids: list[str] | None = None) -> dict[str, int]:
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
        
        # Prioritize current-run items first
        if priority_bvids:
            priority_set = set(priority_bvids)
            current_run = [b for b in ready_to_upload if b in priority_set]
            backlog = [b for b in ready_to_upload if b not in priority_set]
            ready_to_upload = current_run + backlog
        
        self.logger.info(f"Found {len(ready_to_upload)} EPUBs ready for upload")
        
        # Initialize browser client
        weread = WeReadBrowserClient(headless=self.headless)
        
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
                        
                    if weread.upload_epub(str(epub_path)):
                        success = True
                        break
                        
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
    
    def process_single_video(self, item: QueueItem, upload: bool = False) -> dict[str, Any]:
        """Process a single video through the complete pipeline (B→F or B→G).
        
        Args:
            item: Queue item to process.
            upload: If True, also upload to WeChat Reading (Step G).
            
        Returns:
            Result dict with status and any error message.
        """
        result = {
            "bvid": item.bvid,
            "title": item.title,
            "status": "unknown",
            "error": None,
        }
        
        try:
            # Step B: Download
            self.logger.info("="*60)
            self.logger.info(f"Processing: {item.title[:50]}")
            self.logger.info("="*60)
            
            status = self.state.get_status(item.bvid)
            video_state = self.state.get_video_state(item.bvid)
            
            # --- Step B: Download ---
            if status in ("new", "downloading"):
                self.logger.info("[Step B] Downloading audio...")
                self.state.update(item.bvid, status="downloading", title=item.title, up_name=item.up_name, last_attempt=now_iso())
                
                safe_name = safe_filename(f"{item.bvid}_{item.title[:50]}")
                audio_path = self.downloader.download_with_retry(item.url, safe_name)
                
                if not audio_path:
                    self.state.update(item.bvid, status="error", error="Download failed")
                    result["status"] = "error"
                    result["error"] = "Download failed"
                    return result
                
                self.state.update(item.bvid, status="downloaded", title=item.title, up_name=item.up_name, audio_path=str(audio_path))
                video_state = self.state.get_video_state(item.bvid)
                self.logger.info(f"  ✓ Downloaded: {audio_path.name}")
            
            # --- Step C: Transcribe ---
            status = self.state.get_status(item.bvid)
            if status in ("downloaded", "transcribing"):
                self.logger.info(f"[Step C] Transcribing with {self.config.get('asr_engine', 'whisper').upper()}...")
                self.state.update(item.bvid, status="transcribing", title=item.title, up_name=item.up_name, last_attempt=now_iso())
                
                audio_path = Path(video_state.audio_path)
                transcript_md = self.asr_client.transcribe_to_markdown(
                    audio_path,
                    title=item.title,
                    author=item.up_name,
                )
                
                safe_title = safe_filename(item.title)
                transcript_path = self.output_dir / f"{safe_title}.md"
                transcript_path.write_text(transcript_md, encoding="utf-8")
                
                self.state.update(item.bvid, status="transcript_ready", transcript_md=str(transcript_path))
                video_state = self.state.get_video_state(item.bvid)
                self.logger.info(f"  ✓ Transcribed: {transcript_path.name}")
                
                # --- NEW: Memory Management ---
                # Unload ASR model to save VRAM for LLM (Ollama)
                if hasattr(self.asr_client, "unload_model"):
                    self.asr_client.unload_model()
            
            # --- Step D: Correct ---
            status = self.state.get_status(item.bvid)
            if status in ("transcript_ready", "correcting"):
                self.logger.info("[Step D] Correcting with Qwen3...")
                self.state.update(item.bvid, status="correcting", last_attempt=now_iso())
                
                transcript_path = Path(video_state.transcript_md)
                content = transcript_path.read_text(encoding="utf-8")
                
                body_start = content.find("---")
                if body_start != -1:
                    body_start = content.find("\n", body_start) + 1
                    raw_text = content[body_start:].strip()
                else:
                    raw_text = content
                
                # Step D.1: Identify speakers globally
                self.logger.info(f"  Identifying speakers for: {item.title[:40]}...")
                speaker_map = self.ollama.identify_speakers(
                    raw_text,
                    title=item.title,
                    author=item.up_name
                )
                
                if speaker_map:
                    formatted_map = ", ".join([f"{k}→{v}" for k, v in speaker_map.items()])
                    self.logger.info(f"  ✓ Found speakers: {formatted_map}")
                    
                    # Code-level global replacement before LLM
                    for raw_tag, real_name in speaker_map.items():
                        id_match = re.search(r"(?:说话人|SPEAKER)[\s_]*([A-Z0-9]+)", raw_tag, re.I)
                        if id_match:
                            tag_id = id_match.group(1)
                            pattern = rf"\*\*\[?(?:说话人|SPEAKER)[\s_\u3000]*{tag_id}\]?\*\*"
                            count = len(re.findall(pattern, raw_text))
                            if count > 0:
                                self.logger.info(f"    Global replacing {tag_id} -> **[{real_name}]** ({count} occurrences)")
                                raw_text = re.sub(pattern, f"**[{real_name}]**", raw_text)
                else:
                    self.logger.info("  ℹ No speakers identified.")
                
                # Step D.2: LLM correction
                def progress_cb(current, total):
                    print(f"  Correcting paragraph {current}/{total}...", end="\r")
                
                corrected_text = self.ollama.correct_text(
                    raw_text,
                    title=item.title,
                    author=item.up_name,
                    speaker_map=speaker_map,
                    progress_callback=progress_cb
                )
                sys.stdout.write("\n")
                
                # Step D.3: Re-apply speaker replacements after LLM
                # LLM may revert or corrupt labels during paragraph-by-paragraph processing
                if speaker_map:
                    for raw_tag, real_name in speaker_map.items():
                        id_match = re.search(r"(?:说话人|SPEAKER)[\s_]*([A-Z0-9]+)", raw_tag, re.I)
                        if id_match:
                            tag_id = id_match.group(1)
                            pattern = rf"\*\*\[?(?:说话人|SPEAKER)[\s_\u3000]*{tag_id}\]?\*\*"
                            count = len(re.findall(pattern, corrected_text))
                            if count > 0:
                                self.logger.info(f"    Post-LLM fix: {tag_id} -> **[{real_name}]** ({count} remaining)")
                                corrected_text = re.sub(pattern, f"**[{real_name}]**", corrected_text)
                
                corrected_md_lines = [
                    f"# {item.title}",
                    "",
                    f"**UP主**: {item.up_name}",
                    "",
                    "---",
                    "",
                    corrected_text,
                ]
                corrected_md = "\n".join(corrected_md_lines)
                
                corrected_path = transcript_path.with_suffix(".corrected.md")
                corrected_path.write_text(corrected_md, encoding="utf-8")
                
                self.state.update(item.bvid, status="corrected", corrected_md=str(corrected_path))
                video_state = self.state.get_video_state(item.bvid)
                self.logger.info(f"  ✓ Corrected: {corrected_path.name}")
            
            # --- Step E: Summarize ---
            status = self.state.get_status(item.bvid)
            if status in ("corrected", "summarizing"):
                self.logger.info("[Step E] Summarizing with Qwen3...")
                self.state.update(item.bvid, status="summarizing", last_attempt=now_iso())
                
                corrected_path = Path(video_state.corrected_md)
                content = corrected_path.read_text(encoding="utf-8")
                
                body_start = content.find("---")
                if body_start != -1:
                    body_start = content.find("\n", body_start) + 1
                    corrected_text = content[body_start:].strip()
                else:
                    corrected_text = content
                
                summary = self.ollama.summarize(corrected_text, title=item.title, author=item.up_name)
                
                final_md = build_final_markdown(
                    title=item.title,
                    author=item.up_name,
                    summary=summary,
                    corrected_text=corrected_text,
                )
                
                final_path = corrected_path.with_name(
                    corrected_path.name.replace(".corrected.md", ".final.md")
                )
                final_path.write_text(final_md, encoding="utf-8")
                
                self.state.update(item.bvid, status="summarized", transcript_md=str(final_path))
                video_state = self.state.get_video_state(item.bvid)
                self.logger.info(f"  ✓ Summarized: {final_path.name}")
            
            # --- Step F: EPUB ---
            status = self.state.get_status(item.bvid)
            if status in ("summarized", "corrected", "transcript_ready"):
                self.logger.info("[Step F] Generating EPUB...")
                
                md_path = video_state.transcript_md
                md_file = Path(md_path)
                
                safe_title = safe_filename(item.title)
                epub_path = self.epub_dir / f"{safe_title}.epub"
                
                convert_md_to_epub(md_file, epub_path, item.title)
                self.state.update(item.bvid, status="success", epub_path=str(epub_path))
                video_state = self.state.get_video_state(item.bvid)
                self.logger.info(f"  ✓ EPUB: {epub_path.name}")
            
            # --- Step G: Upload (optional) ---
            status = self.state.get_status(item.bvid)
            if upload and status == "success":
                self.logger.info("[Step G] Uploading to WeChat Reading...")
                epub_path = Path(video_state.epub_path)
                
                weread = WeReadBrowserClient(headless=self.headless)
                try:
                    if weread.upload_epub(str(epub_path)):
                        self.state.update(item.bvid, status="uploaded")
                        self.logger.info("  ✓ Uploaded!")
                        result["status"] = "uploaded"
                    else:
                        self.logger.error("  ✗ Upload failed")
                        result["status"] = "success"  # EPUB was created successfully
                finally:
                    weread.close()
            else:
                result["status"] = self.state.get_status(item.bvid)
            
            return result
            
        except BaseException as e:
            err_msg = str(e) or repr(e)
            self.state.update(item.bvid, status="error", error=err_msg, last_attempt=now_iso())
            result["status"] = "error"
            result["error"] = err_msg
            self.logger.exception(f"  ✗ Error processing {item.bvid}: {err_msg}")
            return result
    
    def run_all(self, max_items: int | None = None, upload: bool = False) -> dict[str, Any]:
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
        self.logger.info("="*60)

        # --- Step A: Fetch ---
        queue = self.run_step_a()
        if not queue:
            self.logger.info("No new items to process.")
            return results
        
        # Limit queue to max_items for the whole pipeline run
        current_batch = queue[:max_items]
        results["queued"] = len(current_batch)
        
        # Track all BVIDs touched/successfully processed in this run
        all_touched_bvids = set()
        
        # --- Step BA: AI Filter ---
        self.logger.info("[Stage 1/6] Batch AI Filtering...")
        # run_step_ba_ai_filter returns stats dict
        self.run_step_ba_ai_filter(max_items=max_items)
        
        # Release GPU memory used by Qwen for filtering
        if hasattr(self.ollama, "unload_model"):
            self.ollama.unload_model()
            
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
            
        # --- Step D & E: Correct & Summarize ---
        self.logger.info("[Stage 4/6] Batch Correcting & Summarizing (LLM)...")
        # StepLLM.run() handles both D and E for all items in the batch
        _, e_bvids = self.run_step_e_summarize(max_items=max_items)
        all_touched_bvids.update(e_bvids)
        
        if hasattr(self.ollama, "unload_model"):
            self.ollama.unload_model()
        
        # --- Step F: EPUB ---
        self.logger.info("[Stage 5/6] Batch Generating EPUBs...")
        _, f_bvids = self.run_step_f_epub()
        all_touched_bvids.update(f_bvids)
        
        # --- Step G: Upload ---
        if upload:
            self.logger.info("[Stage 6/6] Batch Uploading to WeChat Reading...")
            # Combine current batch from queue AND items successfully processed in other steps
            # This ensures items recently backlog-processed in Step F also get priority.
            priority_list = list(all_touched_bvids)
            # Mix in the current batch BVIDs as secondary priority if they didn't finish today
            for item in current_batch:
                if item.bvid not in all_touched_bvids:
                    priority_list.append(item.bvid)
            
            upload_stats = self.run_step_g_upload(max_items=max_items, priority_bvids=priority_list)
            results["stats"]["upload"] = upload_stats
        
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
        
        # Auto-reset non-uploaded items for the next run
        reset_count = self.state.reset_non_uploaded_items()
        if reset_count > 0:
            self.logger.info(f"Auto-Reset: {reset_count} items reset to 'new' for retry.")
            
        return results
    
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
