"""Shared fixtures for pipeline regression tests."""

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import VideoInfo, VideoState, QueueItem
from core.state import StateManager
from clients.ollama_client import build_final_markdown


# ---------------------------------------------------------------------------
# Sample text content
# ---------------------------------------------------------------------------

SAMPLE_TITLE = "测试视频：AI 技术分析"
SAMPLE_AUTHOR = "测试UP主"
SAMPLE_BODY = (
    "大家好，今天我们来聊一下人工智能的最新进展。\n\n"
    "首先，大语言模型在 2024 年取得了重大突破。\n\n"
    "其次，多模态模型也在快速发展。\n\n"
    "最后，让我们总结一下今天的内容。"
)
SAMPLE_SUMMARY = (
    "## 核心摘要\n\n"
    "本期视频讨论了人工智能的最新进展，涵盖大语言模型突破和多模态发展。\n\n"
    "## 要点列表\n\n"
    "- 大语言模型在 2024 年取得重大突破\n"
    "- 多模态模型快速发展\n"
    "> AI 技术正以前所未有的速度进步\n\n"
    "## 总结与建议\n\n"
    "关注大语言模型和多模态领域的发展趋势。"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transcript_md(tmp_path: Path) -> Path:
    """Create a minimal raw transcript markdown file."""
    md = f"# {SAMPLE_TITLE}\n\n**UP主**: {SAMPLE_AUTHOR}\n\n---\n\n{SAMPLE_BODY}"
    path = tmp_path / "测试视频.md"
    path.write_text(md, encoding="utf-8")
    return path


@pytest.fixture
def sample_corrected_md(tmp_path: Path) -> Path:
    """Create a corrected transcript markdown file."""
    md = f"# {SAMPLE_TITLE}\n\n**UP主**: {SAMPLE_AUTHOR}\n\n---\n\n{SAMPLE_BODY}"
    path = tmp_path / "测试视频.corrected.md"
    path.write_text(md, encoding="utf-8")
    return path


@pytest.fixture
def sample_final_md(tmp_path: Path) -> Path:
    """Create a final markdown file (summary + body) via build_final_markdown."""
    content = build_final_markdown(
        title=SAMPLE_TITLE,
        author=SAMPLE_AUTHOR,
        summary=SAMPLE_SUMMARY,
        corrected_text=SAMPLE_BODY,
    )
    path = tmp_path / "测试视频.final.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def tmp_state_manager(tmp_path: Path) -> StateManager:
    """Create a StateManager backed by a temporary file."""
    return StateManager(tmp_path / "pipeline_state.json")


@pytest.fixture
def sample_video_info() -> VideoInfo:
    """Create a sample VideoInfo instance."""
    return VideoInfo(
        bvid="BV1test12345",
        title=SAMPLE_TITLE,
        url="https://www.bilibili.com/video/BV1test12345",
        duration=600,
        up_name=SAMPLE_AUTHOR,
        pubdate=1700000000,
    )


@pytest.fixture
def sample_queue_item() -> QueueItem:
    """Create a sample QueueItem instance."""
    return QueueItem(
        bvid="BV1test12345",
        title=SAMPLE_TITLE,
        url="https://www.bilibili.com/video/BV1test12345",
        duration=600,
        up_name=SAMPLE_AUTHOR,
        pubdate=1700000000,
    )


@pytest.fixture
def sample_filter_config() -> dict:
    """Sample filter configuration."""
    return {
        "min_seconds": 60,
        "up_deny_contains": ["广告号", "spam"],
        "title_deny_regex": [r"(?i)游戏直播", r"(?i)抽奖"],
        "keep_when_uncertain": True,
    }
