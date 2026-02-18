"""Tests for build_final_markdown structure."""

from clients.ollama_client import build_final_markdown


TITLE = "测试视频：技术分析"
AUTHOR = "测试UP主"
SUMMARY = (
    "## 核心摘要\n\n一段摘要文字。\n\n"
    "## 要点列表\n\n- 要点一\n- 要点二\n\n"
    "## 总结与建议\n\n总结建议。"
)
BODY = "大家好，今天来聊聊技术。\n\n第一段内容。\n\n第二段内容。"


class TestBuildFinalMarkdown:
    def test_structure(self):
        """Output has # Title, **UP主**, ---, summary, ---, ## 完整文本, body."""
        md = build_final_markdown(title=TITLE, author=AUTHOR, summary=SUMMARY, corrected_text=BODY)

        assert md.startswith(f"# {TITLE}")
        assert f"**UP主**: {AUTHOR}" in md
        assert "## 完整文本" in md
        assert "---" in md

    def test_summary_before_body(self):
        """Summary text appears before ## 完整文本 section."""
        md = build_final_markdown(title=TITLE, author=AUTHOR, summary=SUMMARY, corrected_text=BODY)

        summary_pos = md.find("## 核心摘要")
        body_pos = md.find("## 完整文本")
        assert summary_pos < body_pos, "Summary must appear before body"

    def test_chinese_content(self):
        """Chinese title/author/text preserved correctly."""
        md = build_final_markdown(title=TITLE, author=AUTHOR, summary=SUMMARY, corrected_text=BODY)

        assert TITLE in md
        assert AUTHOR in md
        assert "大家好" in md
        assert "核心摘要" in md
