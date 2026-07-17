"""Transcription step for the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base_step import BaseStep
from .models import QueueItem
from .state import now_iso
from utils import safe_filename

# Process this many audio files per batch group — limits peak RAM and saves
# progress incrementally so a crash only loses the current group.
BATCH_GROUP_SIZE = 8
LONG_VIDEO_SECONDS = 60 * 60


class StepASR(BaseStep):
    """Step C: Transcribe downloaded audio with Qwen3-ASR (hybrid speaker routing)."""

    @staticmethod
    def _build_groups(
        pending: list[tuple[QueueItem, Path]],
    ) -> list[list[tuple[QueueItem, Path]]]:
        short_items = [pair for pair in pending if pair[0].duration < LONG_VIDEO_SECONDS]
        long_items = [pair for pair in pending if pair[0].duration >= LONG_VIDEO_SECONDS]
        groups = [
            short_items[i:i + BATCH_GROUP_SIZE]
            for i in range(0, len(short_items), BATCH_GROUP_SIZE)
        ]
        groups.extend([[pair] for pair in long_items])
        return groups

    def run(self, max_items: Optional[int] = None) -> dict[str, int]:
        max_items = self.get_max_items(max_items)
        engine_name = self.config.get("asr_engine", "qwen3").upper()
        self.logger.info(f"=== Step C: Transcribe with {engine_name} (max {max_items}) ===")

        queue = self.load_queue()
        if not queue:
            return {"error": 1}

        stats = {
            "processed": 0,
            "transcribed": 0,
            "no_audio": 0,
            "already_transcribed": 0,
            "error": 0,
        }

        # Collect items that need transcription
        pending: list[tuple[QueueItem, Path]] = []
        for item in queue:
            if stats["processed"] + len(pending) >= max_items:
                break

            status = self.state.get_status(item.bvid)
            if status not in ("downloaded", "transcribing"):
                if status in ("transcript_ready", "correcting", "corrected",
                              "summarizing", "summarized", "success", "uploaded"):
                    stats["already_transcribed"] += 1
                continue

            video_state = self.state.get_video_state(item.bvid)
            if not video_state.audio_path or not Path(video_state.audio_path).exists():
                self.logger.error(f"✗ No audio file: {item.title[:50]}")
                stats["no_audio"] += 1
                continue

            pending.append((item, Path(video_state.audio_path)))
            self.state.update(item.bvid, status="transcribing", title=item.title,
                              up_name=item.up_name, last_attempt=now_iso())

        asr = self.pipeline.asr_client
        use_smart = bool(pending) and hasattr(asr, "transcribe_files_smart")

        if use_smart:
            groups = self._build_groups(pending)
            long_count = sum(1 for item, _ in pending if item.duration >= LONG_VIDEO_SECONDS)
            n_groups = len(groups)
            self.logger.info(
                f"Hybrid transcription: {len(pending)} files in {n_groups} groups "
                f"({long_count} long video(s) isolated)"
            )

            for g, group in enumerate(groups):
                self.logger.info(f"Group {g + 1}/{n_groups}: {len(group)} files")

                items = [(p, item.title, item.up_name) for item, p in group]
                try:
                    results = asr.transcribe_files_smart(items)
                except Exception as e:
                    self.logger.error(f"Group {g + 1} failed: {e!r} — marking as error")
                    for item, _ in group:
                        self.state.update(item.bvid, status="error",
                                          error=f"Batch group failed: {e!r}",
                                          last_attempt=now_iso())
                        stats["error"] += 1
                        stats["processed"] += 1
                    continue

                if len(results) != len(group):
                    self.logger.error(
                        f"Group {g + 1} returned {len(results)} results for "
                        f"{len(group)} inputs"
                    )
                    results = list(results) + [("", "")] * (len(group) - len(results))

                for (item, audio_path), (md, lang) in zip(group, results):
                    if not md:
                        self.state.update(item.bvid, status="error",
                                          error="Transcription returned empty result",
                                          last_attempt=now_iso())
                        self.logger.error(f"  ✗ Empty result: {item.title[:50]}")
                        stats["error"] += 1
                        stats["processed"] += 1
                        continue
                    self._save_transcript(item, audio_path, md, lang, stats)
        else:
            # Sequential fallback (e.g. a client without the hybrid method)
            for item, audio_path in pending:
                self.logger.info(f"Transcribing: {item.title[:60]}...")
                try:
                    transcript_md, detected_lang = asr.transcribe_to_markdown(
                        audio_path, title=item.title, author=item.up_name,
                    )
                    self._save_transcript(item, audio_path, transcript_md, detected_lang, stats)
                except Exception as e:
                    self.state.update(item.bvid, status="error",
                                      error=f"Transcription failed: {str(e)}",
                                      last_attempt=now_iso())
                    self.logger.error(f"  ✗ Transcription failed: {e}")
                    stats["error"] += 1
                stats["processed"] += 1

        # Unload model after the batch
        if hasattr(asr, "unload_model"):
            asr.unload_model()

        self.logger.info(f"=== Step C Complete ===")
        self.logger.info(
            f"Transcribed: {stats['transcribed']}, "
            f"Already: {stats['already_transcribed']}, "
            f"Errors: {stats['error']}"
        )
        processed_bvids = [
            item.bvid for item, _ in pending
            if self.state.get_status(item.bvid) == "transcript_ready"
        ]
        return stats, processed_bvids

    def _save_transcript(self, item, audio_path, transcript_md, detected_lang, stats):
        safe_title = safe_filename(item.title)
        transcript_path = self.pipeline.output_dir / f"{safe_title}.md"
        transcript_path.write_text(transcript_md, encoding="utf-8")

        self.state.update(
            item.bvid,
            status="transcript_ready",
            title=item.title,
            up_name=item.up_name,
            transcript_md=str(transcript_path),
            language=detected_lang,
            last_attempt=now_iso(),
            audio_path=None,
        )
        if hasattr(self.pipeline.asr_client, "clear_checkpoint"):
            self.pipeline.asr_client.clear_checkpoint(audio_path)
        try:
            audio_path.unlink()
            self.logger.info(f"  ✓ Deleted audio: {audio_path.name}")
        except OSError as e:
            self.logger.warning(f"  ⚠ Could not delete audio {audio_path.name}: {e}")
        self.logger.info(f"  ✓ Saved: {transcript_path.name}")
        stats["transcribed"] += 1
        stats["processed"] += 1
