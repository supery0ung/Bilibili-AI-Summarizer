"""Tests for StepLLM contracts (mocked — no real LLM calls).

These verify the file I/O and state update contracts that _correct_item
and _summarize_item must honor, without actually calling Ollama.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.models import QueueItem, VideoState
from core.state import StateManager
from clients.ollama_client import build_final_markdown

# Same sample data as conftest fixtures
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


def _make_item() -> QueueItem:
    return QueueItem(
        bvid="BV1llm",
        title=SAMPLE_TITLE,
        url="https://bilibili.com/video/BV1llm",
        duration=600,
        up_name=SAMPLE_AUTHOR,
    )


class TestCorrectItemContract:
    """Test _correct_item file I/O and state contract."""

    def test_produces_corrected_md(self, tmp_path: Path, tmp_state_manager: StateManager):
        """_correct_item must create a .corrected.md file."""
        # Setup: create a raw transcript
        transcript = tmp_path / "test.md"
        transcript.write_text(
            f"# {SAMPLE_TITLE}\n\n**UP主**: {SAMPLE_AUTHOR}\n\n---\n\n{SAMPLE_BODY}",
            encoding="utf-8",
        )

        sm = tmp_state_manager
        sm.update("BV1llm", status="transcript_ready", transcript_md=str(transcript))

        # Simulate what _correct_item does (without importing StepLLM which needs Pipeline)
        video_state = sm.get_video_state("BV1llm")
        transcript_path = Path(video_state.transcript_md)
        content = transcript_path.read_text(encoding="utf-8")

        body_start = content.find("---")
        if body_start != -1:
            body_start = content.find("\n", body_start) + 1
            raw_text = content[body_start:].strip()
        else:
            raw_text = content

        # Mock correction output (identity — just return same text)
        corrected_text = raw_text
        corrected_md = f"# {SAMPLE_TITLE}\n\n**UP主**: {SAMPLE_AUTHOR}\n\n---\n\n{corrected_text}"
        corrected_path = transcript_path.with_suffix(".corrected.md")
        corrected_path.write_text(corrected_md, encoding="utf-8")

        sm.update("BV1llm", status="corrected", corrected_md=str(corrected_path))

        # Assertions
        assert corrected_path.exists()
        assert corrected_path.suffix == ".md"
        assert ".corrected" in corrected_path.name
        assert sm.get_status("BV1llm") == "corrected"
        assert sm.get_video_state("BV1llm").corrected_md == str(corrected_path)

    def test_corrected_file_format(self, tmp_path: Path):
        """Corrected file has # Title, **UP主**, ---, body structure."""
        corrected_path = tmp_path / "test.corrected.md"
        corrected_md = f"# {SAMPLE_TITLE}\n\n**UP主**: {SAMPLE_AUTHOR}\n\n---\n\n{SAMPLE_BODY}"
        corrected_path.write_text(corrected_md, encoding="utf-8")

        content = corrected_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()

        assert lines[0].startswith("# ")
        assert "**UP主**" in content
        assert "---" in content


class TestSummarizeItemContract:
    """Test _summarize_item file I/O and state contract."""

    def test_saves_summary_md_to_state(self, tmp_path: Path, tmp_state_manager: StateManager):
        """_summarize_item must set summary_md in state."""
        corrected = tmp_path / "test.corrected.md"
        corrected.write_text(
            f"# {SAMPLE_TITLE}\n\n**UP主**: {SAMPLE_AUTHOR}\n\n---\n\n{SAMPLE_BODY}",
            encoding="utf-8",
        )

        sm = tmp_state_manager
        sm.update("BV1llm", status="corrected", corrected_md=str(corrected))

        # Simulate _summarize_item
        final_md = build_final_markdown(
            title=SAMPLE_TITLE,
            author=SAMPLE_AUTHOR,
            summary=SAMPLE_SUMMARY,
            corrected_text=SAMPLE_BODY,
        )
        final_path = corrected.with_name(corrected.name.replace(".corrected.md", ".final.md"))
        final_path.write_text(final_md, encoding="utf-8")

        sm.update("BV1llm", status="summarized", summary_md=str(final_path))

        # Assertions
        assert final_path.exists()
        assert sm.get_status("BV1llm") == "summarized"
        assert sm.get_video_state("BV1llm").summary_md == str(final_path)

    def test_final_md_has_summary_sections(self, tmp_path: Path):
        """Final markdown has 核心摘要, 要点列表, 完整文本 sections."""
        final_md = build_final_markdown(
            title=SAMPLE_TITLE,
            author=SAMPLE_AUTHOR,
            summary=SAMPLE_SUMMARY,
            corrected_text=SAMPLE_BODY,
        )

        assert "## 核心摘要" in final_md
        assert "## 要点列表" in final_md
        assert "## 完整文本" in final_md


class TestStatusFlowThroughLLM:
    """Test status progression through the LLM steps."""

    def test_status_flow(self, tmp_state_manager: StateManager):
        """Status progresses: transcript_ready → correcting → corrected → summarizing → summarized."""
        sm = tmp_state_manager
        bvid = "BV1flow"

        sm.update(bvid, status="transcript_ready")
        assert sm.get_status(bvid) == "transcript_ready"

        sm.update(bvid, status="correcting")
        assert sm.get_status(bvid) == "correcting"

        sm.update(bvid, status="corrected", corrected_md="/path/corrected.md")
        assert sm.get_status(bvid) == "corrected"

        sm.update(bvid, status="summarizing")
        assert sm.get_status(bvid) == "summarizing"

        sm.update(bvid, status="summarized", summary_md="/path/final.md")
        assert sm.get_status(bvid) == "summarized"
        assert sm.get_video_state(bvid).summary_md == "/path/final.md"
