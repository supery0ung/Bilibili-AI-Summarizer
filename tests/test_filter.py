"""Tests for VideoFilter rules."""

from core.filter import VideoFilter
from core.models import VideoInfo


def _video(
    bvid="BV1x",
    title="普通视频标题",
    duration=600,
    up_name="普通UP主",
) -> VideoInfo:
    return VideoInfo(
        bvid=bvid,
        title=title,
        url=f"https://bilibili.com/video/{bvid}",
        duration=duration,
        up_name=up_name,
    )


class TestVideoFilter:
    def test_filter_short_duration(self, sample_filter_config: dict):
        """Videos under min_seconds are dropped."""
        f = VideoFilter(sample_filter_config)
        short = _video(duration=30)
        assert f.should_keep(short) is False

    def test_filter_up_deny(self, sample_filter_config: dict):
        """Deny-listed UP names cause filtering."""
        f = VideoFilter(sample_filter_config)
        denied = _video(up_name="某某广告号official")
        assert f.should_keep(denied) is False

    def test_filter_title_regex(self, sample_filter_config: dict):
        """Title deny regex patterns work."""
        f = VideoFilter(sample_filter_config)
        gaming = _video(title="今日游戏直播精彩集锦")
        assert f.should_keep(gaming) is False

    def test_filter_keeps_valid(self, sample_filter_config: dict):
        """A normal, valid video passes all filters."""
        f = VideoFilter(sample_filter_config)
        good = _video(title="深度分析AI发展", duration=1200, up_name="硅谷101")
        assert f.should_keep(good) is True
