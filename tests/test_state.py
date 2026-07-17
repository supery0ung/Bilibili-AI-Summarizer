"""Tests for StateManager persistence and transitions."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import pytest

from core.models import VideoInfo
from core.state import StateManager


class TestStateManagerBasics:
    def test_update_and_get(self, tmp_state_manager: StateManager):
        """update() stores fields, get_video_state() retrieves them."""
        sm = tmp_state_manager
        sm.update("BV1a", status="downloaded", audio_path="/audio.m4a")

        vs = sm.get_video_state("BV1a")
        assert vs.status == "downloaded"
        assert vs.audio_path == "/audio.m4a"

    def test_summary_md_persists(self, tmp_state_manager: StateManager):
        """After update(summary_md=path), get_video_state().summary_md == path."""
        sm = tmp_state_manager
        sm.update("BV1b", status="summarized", summary_md="/path/to/final.md")

        vs = sm.get_video_state("BV1b")
        assert vs.summary_md == "/path/to/final.md"
        assert vs.status == "summarized"

    def test_get_pending_items(self, tmp_state_manager: StateManager):
        """get_pending_items returns correct bvids for a given status."""
        sm = tmp_state_manager
        sm.update("BV1x", status="summarized")
        sm.update("BV1y", status="summarized")
        sm.update("BV1z", status="corrected")

        pending = sm.get_pending_items("summarized")
        assert set(pending) == {"BV1x", "BV1y"}

        corrected = sm.get_pending_items("corrected")
        assert corrected == ["BV1z"]

    def test_concurrent_updates_keep_valid_complete_json(
        self, tmp_state_manager: StateManager
    ):
        sm = tmp_state_manager

        def update_one(i: int):
            sm.update(f"BV{i:03d}", status="downloaded", title=f"Video {i}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(update_one, range(50)))

        raw = json.loads(sm.state_file.read_text(encoding="utf-8"))
        assert len(raw["videos"]) == 50
        assert all(v["status"] == "downloaded" for v in raw["videos"].values())

    def test_corrupt_state_fails_loudly_instead_of_resetting(self, tmp_path: Path):
        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text("{broken", encoding="utf-8")

        with pytest.raises(RuntimeError, match="refusing to start with empty state"):
            StateManager(state_file)


class TestStatusTransitions:
    def test_full_status_flow(self, tmp_state_manager: StateManager):
        """Status can progress through the full pipeline flow."""
        sm = tmp_state_manager
        bvid = "BV1flow"

        steps = [
            "new", "downloading", "downloaded",
            "transcribing", "transcript_ready",
            "correcting", "corrected",
            "summarizing", "summarized",
            "success", "uploaded",
        ]
        for status in steps:
            sm.update(bvid, status=status)
            assert sm.get_status(bvid) == status


class TestInterruptedRecovery:
    def test_recovers_latest_existing_artifact(
        self, tmp_state_manager: StateManager, tmp_path: Path
    ):
        transcript = tmp_path / "video.md"
        corrected = tmp_path / "video.corrected.md"
        summary = tmp_path / "video.final.md"
        epub = tmp_path / "video.epub"
        for path in (transcript, corrected, summary, epub):
            path.write_text("content", encoding="utf-8")

        sm = tmp_state_manager
        sm.update("empty", status="transcribing", audio_path=str(tmp_path / "missing.m4a"))
        sm.update("transcript", status="transcribing", transcript_md=str(transcript))
        sm.update("corrected", status="correcting", corrected_md=str(corrected))
        sm.update("summary", status="summarizing", summary_md=str(summary))
        sm.update("epub", status="error", epub_path=str(epub), error="interrupted")

        assert sm.recover_interrupted_items() == 5
        assert sm.get_status("empty") == "new"
        assert sm.get_status("transcript") == "transcript_ready"
        assert sm.get_status("corrected") == "corrected"
        assert sm.get_status("summary") == "summarized"
        assert sm.get_status("epub") == "success"

    def test_preserves_terminal_states(self, tmp_state_manager: StateManager):
        sm = tmp_state_manager
        sm.update("uploaded", status="uploaded", error="historical note")
        sm.update("skipped", status="skipped_ai")

        assert sm.recover_interrupted_items() == 0
        assert sm.get_status("uploaded") == "uploaded"
        assert sm.get_status("skipped") == "skipped_ai"

    def test_recovers_existing_audio_as_downloaded(
        self, tmp_state_manager: StateManager, tmp_path: Path
    ):
        audio = tmp_path / "video.m4a"
        audio.write_bytes(b"audio")
        sm = tmp_state_manager
        sm.update("BV1audio", status="transcribing", audio_path=str(audio))

        assert sm.recover_interrupted_items() == 1
        assert sm.get_status("BV1audio") == "downloaded"


class TestBuildQueue:
    def _make_video(self, bvid: str) -> VideoInfo:
        return VideoInfo(
            bvid=bvid,
            title=f"Title {bvid}",
            url=f"https://bilibili.com/video/{bvid}",
            duration=300,
            up_name="Author",
            pubdate=1700000000,
        )

    def test_skips_uploaded(self, tmp_state_manager: StateManager):
        """build_queue() skips videos with 'uploaded' status."""
        sm = tmp_state_manager
        sm.update("BV1up", status="uploaded")

        queue = sm.build_queue([self._make_video("BV1up")])
        assert len(queue) == 0

    def test_includes_error(self, tmp_state_manager: StateManager):
        """build_queue() includes videos with 'error' status for retry."""
        sm = tmp_state_manager
        sm.update("BV1err", status="error")

        queue = sm.build_queue([self._make_video("BV1err")])
        assert len(queue) == 1
        assert queue[0].bvid == "BV1err"

    def test_includes_downloaded_for_transcription(
        self, tmp_state_manager: StateManager
    ):
        """Downloaded audio must remain queueable after an interrupted run."""
        sm = tmp_state_manager
        sm.update("BV1audio", status="downloaded", audio_path="/audio.m4a")

        queue = sm.build_queue([self._make_video("BV1audio")])
        assert len(queue) == 1
        assert queue[0].bvid == "BV1audio"
