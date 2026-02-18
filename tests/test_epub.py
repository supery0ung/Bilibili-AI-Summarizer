"""Tests for EPUB generation — the most critical regression tests.

These validate that:
1. Generated EPUBs contain AI summaries when available
2. summary_md is prioritized over transcript_md
3. EPUB structure is valid
"""

import zipfile
from pathlib import Path

from clients.ollama_client import build_final_markdown
from core.state import StateManager
from utils.md_to_epub import convert_md_to_epub

# Re-use the same sample data as conftest fixtures
SAMPLE_TITLE = "测试视频：AI 技术分析"
SAMPLE_AUTHOR = "测试UP主"


class TestEpubFromFinalMd:
    """Test EPUB generated from .final.md (summary + body)."""

    def test_epub_contains_summary_sections(self, sample_final_md: Path, tmp_path: Path):
        """EPUB from .final.md contains 核心摘要, 要点列表, 完整文本."""
        epub_path = tmp_path / "output.epub"
        convert_md_to_epub(sample_final_md, epub_path, SAMPLE_TITLE)

        with zipfile.ZipFile(epub_path, "r") as z:
            xhtml = z.read("OEBPS/content.xhtml").decode("utf-8")

        assert "核心摘要" in xhtml, "EPUB must contain '核心摘要' section"
        assert "要点列表" in xhtml, "EPUB must contain '要点列表' section"
        assert "完整文本" in xhtml, "EPUB must contain '完整文本' section"

    def test_epub_contains_body_text(self, sample_final_md: Path, tmp_path: Path):
        """EPUB contains the actual transcript body text."""
        epub_path = tmp_path / "output.epub"
        convert_md_to_epub(sample_final_md, epub_path, SAMPLE_TITLE)

        with zipfile.ZipFile(epub_path, "r") as z:
            xhtml = z.read("OEBPS/content.xhtml").decode("utf-8")

        assert "人工智能" in xhtml


class TestEpubFromTranscriptMd:
    """Test EPUB generated from raw transcript (no summary)."""

    def test_epub_from_transcript_still_works(self, sample_transcript_md: Path, tmp_path: Path):
        """EPUB generated from raw transcript is valid."""
        epub_path = tmp_path / "output.epub"
        convert_md_to_epub(sample_transcript_md, epub_path, SAMPLE_TITLE)

        assert epub_path.exists()
        assert epub_path.stat().st_size > 0

        with zipfile.ZipFile(epub_path, "r") as z:
            xhtml = z.read("OEBPS/content.xhtml").decode("utf-8")
        assert "人工智能" in xhtml


class TestEpubPrioritization:
    """Test that summary_md is prioritized over transcript_md."""

    def test_prioritizes_summary_md(
        self,
        sample_final_md: Path,
        sample_transcript_md: Path,
        tmp_state_manager: StateManager,
        tmp_path: Path,
    ):
        """When both summary_md and transcript_md exist, summary_md is used."""
        sm = tmp_state_manager
        bvid = "BV1prio"
        sm.update(
            bvid,
            status="summarized",
            transcript_md=str(sample_transcript_md),
            summary_md=str(sample_final_md),
        )

        vs = sm.get_video_state(bvid)
        # This is the critical line that was previously broken
        md_path = vs.summary_md or vs.transcript_md
        assert md_path == str(sample_final_md), "summary_md must be prioritized"

        # Generate EPUB from the selected file
        epub_path = tmp_path / "priority.epub"
        convert_md_to_epub(Path(md_path), epub_path, SAMPLE_TITLE)

        with zipfile.ZipFile(epub_path, "r") as z:
            xhtml = z.read("OEBPS/content.xhtml").decode("utf-8")

        assert "核心摘要" in xhtml, "EPUB from prioritized summary_md must contain summary"

    def test_falls_back_to_transcript_md(
        self,
        sample_transcript_md: Path,
        tmp_state_manager: StateManager,
    ):
        """When summary_md is None, transcript_md is used."""
        sm = tmp_state_manager
        bvid = "BV1fall"
        sm.update(
            bvid,
            status="transcript_ready",
            transcript_md=str(sample_transcript_md),
        )

        vs = sm.get_video_state(bvid)
        md_path = vs.summary_md or vs.transcript_md
        assert md_path == str(sample_transcript_md)


class TestEpubStructure:
    """Test EPUB ZIP structure and metadata."""

    def test_valid_zip(self, sample_final_md: Path, tmp_path: Path):
        """EPUB is a valid ZIP with required entries."""
        epub_path = tmp_path / "structure.epub"
        convert_md_to_epub(sample_final_md, epub_path, SAMPLE_TITLE)

        with zipfile.ZipFile(epub_path, "r") as z:
            names = z.namelist()

        assert "mimetype" in names
        assert "META-INF/container.xml" in names
        assert "OEBPS/content.opf" in names
        assert "OEBPS/toc.ncx" in names
        assert "OEBPS/content.xhtml" in names

    def test_title_in_metadata(self, sample_final_md: Path, tmp_path: Path):
        """EPUB content.opf contains the correct <dc:title>."""
        epub_path = tmp_path / "meta.epub"
        convert_md_to_epub(sample_final_md, epub_path, SAMPLE_TITLE)

        with zipfile.ZipFile(epub_path, "r") as z:
            opf = z.read("OEBPS/content.opf").decode("utf-8")

        assert SAMPLE_TITLE in opf

    def test_xhtml_well_formed(self, sample_final_md: Path, tmp_path: Path):
        """content.xhtml has proper XML declaration and html/body tags."""
        epub_path = tmp_path / "xhtml.epub"
        convert_md_to_epub(sample_final_md, epub_path, SAMPLE_TITLE)

        with zipfile.ZipFile(epub_path, "r") as z:
            xhtml = z.read("OEBPS/content.xhtml").decode("utf-8")

        assert xhtml.startswith("<?xml")
        assert "<html" in xhtml
        assert "<body>" in xhtml
        assert "</body>" in xhtml
        assert "</html>" in xhtml
