"""Tests for data models (VideoInfo, VideoState, QueueItem)."""

from core.models import VideoInfo, VideoState, QueueItem


class TestVideoState:
    def test_roundtrip(self):
        """VideoState.to_dict() → from_dict() preserves all fields."""
        state = VideoState(
            bvid="BV1abc",
            status="summarized",
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-02T00:00:00Z",
            pubdate=1700000000,
            first_attempt="2026-01-01T01:00:00Z",
            last_attempt="2026-01-01T02:00:00Z",
            audio_path="/path/to/audio.m4a",
            transcript_md="/path/to/transcript.md",
            corrected_md="/path/to/corrected.md",
            summary_md="/path/to/final.md",
            epub_path="/path/to/book.epub",
            title="测试标题",
            up_name="测试作者",
            language="Chinese",
            error=None,
        )
        d = state.to_dict()
        restored = VideoState.from_dict("BV1abc", d)

        assert restored.bvid == state.bvid
        assert restored.status == state.status
        assert restored.summary_md == state.summary_md
        assert restored.transcript_md == state.transcript_md
        assert restored.corrected_md == state.corrected_md
        assert restored.language == state.language
        assert restored.audio_path == state.audio_path
        assert restored.epub_path == state.epub_path
        assert restored.title == state.title
        assert restored.up_name == state.up_name

    def test_defaults(self):
        """VideoState defaults are sensible."""
        state = VideoState(bvid="BV1xyz")
        assert state.status == "new"
        assert state.summary_md is None
        assert state.transcript_md is None
        assert state.language is None


class TestQueueItem:
    def test_roundtrip(self):
        """QueueItem.to_dict() → from_dict() preserves fields."""
        item = QueueItem(
            bvid="BV1test",
            title="测试视频",
            url="https://bilibili.com/video/BV1test",
            duration=300,
            up_name="测试UP",
            pubdate=1700000000,
        )
        d = item.to_dict()
        restored = QueueItem.from_dict(d)

        assert restored.bvid == item.bvid
        assert restored.title == item.title
        assert restored.duration == item.duration
        assert restored.up_name == item.up_name
        assert restored.pubdate == item.pubdate


class TestVideoInfo:
    def test_from_api_response(self):
        """VideoInfo.from_api_response() parses a sample API dict."""
        api_item = {
            "bvid": "BV1fromapi",
            "title": "API视频标题",
            "duration": 1200,
            "owner": {"name": "API作者"},
            "aid": 12345,
            "cid": 67890,
            "pubdate": 1700000000,
        }
        video = VideoInfo.from_api_response(api_item)

        assert video.bvid == "BV1fromapi"
        assert video.title == "API视频标题"
        assert video.duration == 1200
        assert video.up_name == "API作者"
        assert video.aid == 12345
        assert video.pubdate == 1700000000
        assert "BV1fromapi" in video.url
