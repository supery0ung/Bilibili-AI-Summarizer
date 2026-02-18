"""Tests for md_to_html conversion and safe_filename utility."""

from utils.md_to_epub import md_to_html, safe_filename


class TestMdToHtml:
    def test_headings(self):
        """# H1, ## H2 → <h1>, <h2>."""
        result = md_to_html("# Heading 1\n## Heading 2\n### Heading 3")
        assert "<h1>" in result
        assert "<h2>" in result
        assert "<h3>" in result

    def test_bold_italic(self):
        """**bold**, *italic* → <b>, <i>."""
        result = md_to_html("This is **bold** and *italic* text.")
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result

    def test_lists(self):
        """- item → <ul><li>."""
        result = md_to_html("- Item A\n- Item B\n- Item C")
        assert "<ul>" in result
        assert "<li>" in result
        assert result.count("<li>") == 3

    def test_blockquote(self):
        """> text → <blockquote>."""
        result = md_to_html("> This is a quote")
        assert "<blockquote>" in result

    def test_horizontal_rule(self):
        """--- → <hr/>."""
        result = md_to_html("Some text\n\n---\n\nMore text")
        assert "<hr/>" in result

    def test_inline_code(self):
        """`code` → <code>."""
        result = md_to_html("Use `pip install` to install.")
        assert "<code" in result

    def test_paragraph(self):
        """Normal text becomes <p>."""
        result = md_to_html("Hello world")
        assert "<p>" in result


class TestSafeFilename:
    def test_special_chars_removed(self):
        """Unsafe characters are replaced."""
        result = safe_filename('file<>:"/\\|?*name')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "\\" not in result
        assert "?" not in result
        assert "*" not in result

    def test_length_capped(self):
        """Filename is truncated to max_length."""
        long_title = "A" * 200
        result = safe_filename(long_title, max_length=50)
        assert len(result) <= 50

    def test_chinese_preserved(self):
        """Chinese characters are preserved."""
        result = safe_filename("人工智能技术分析")
        assert "人工智能" in result

    def test_empty_returns_untitled(self):
        """Empty input returns 'untitled'."""
        result = safe_filename("")
        assert result == "untitled"
