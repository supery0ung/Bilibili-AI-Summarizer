"""Tests for StateManager persistence and transitions."""

from pathlib import Path

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
