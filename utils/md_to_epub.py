"""Convert Markdown to EPUB.

This is a pure Python implementation with no external dependencies.
Adapted from the original bilibili_summarizer project.
"""

from __future__ import annotations

import html
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def md_to_html(md: str) -> str:
    """Convert markdown to XHTML."""
    lines = md.splitlines()
    out: list[str] = []
    in_ul = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def process_inline(text: str) -> str:
        """Process inline formatting like bold, italic, code."""
        if not text:
            return ""
            
        # Escape HTML first
        text = html.escape(text)
        
        # 1. Inline code: `text`
        text = re.sub(r'`([^`]+)`', r'<code style="background:#f4f4f4;padding:0 2px"> \1 </code>', text)
        
        # 2. Extract links to protect them from style formatting
        links = []
        def save_link(m):
            links.append(m.group(0))
            return f"__LINK_PLACEHOLDER_{len(links)-1}__"
        
        text = re.sub(r'\[(.*?)\]\((.*?)\)', save_link, text)
        
        # 3. Bold: **text** or __text__
        text = re.sub(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1', r'<b>\2</b>', text)
        
        # 4. Italic: *text* or _text_
        text = re.sub(r'(?<!\*)(\*|_)(?=\S)(.+?)(?<=\S)\1(?!\*)', r'<i>\2</i>', text)
        
        # 5. Restore and format links
        def restore_link(m):
            placeholder_text = m.group(0)
            idx = int(re.search(r'\d+', placeholder_text).group())
            link_raw = links[idx]
            # Format the actual link
            ml = re.match(r'\[(.*?)\]\((.*?)\)', link_raw)
            if ml:
                return f'<a href="{ml.group(2)}">{ml.group(1)}</a>'
            return link_raw
            
        text = re.sub(r'__LINK_PLACEHOLDER_\d+__', restore_link, text)
        
        return text

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            close_ul()
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_ul()
            level = len(m.group(1))
            text = process_inline(m.group(2).strip())
            out.append(f"<h{level}>{text}</h{level}>")
            continue

        # horizontal rule (---, ***, ___)
        if re.match(r"^\s*([-*_])\1{2,}\s*$", line):
            close_ul()
            out.append("<hr/>")
            continue

        # blockquote
        if line.lstrip().startswith("> "):
            close_ul()
            text = process_inline(line.lstrip()[2:].strip())
            out.append(f"<blockquote>{text}</blockquote>")
            continue

        # bullet list (support both - and *)
        strip_line = line.lstrip()
        if strip_line.startswith("- ") or strip_line.startswith("* "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            item_text = strip_line[2:].strip()
            out.append(f"<li>{process_inline(item_text)}</li>")
            continue

        close_ul()
        # Regular paragraph
        out.append(f"<p>{process_inline(line.strip())}</p>")

    close_ul()
    body = "\n".join(out)
    return (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<!DOCTYPE html>\n"
        "<html xmlns='http://www.w3.org/1999/xhtml' xml:lang='zh-CN' lang='zh-CN'>\n"
        "<head>\n"
        "  <meta charset='utf-8'/>\n"
        "  <title>Content</title>\n"
        "  <style>\n"
        "    body{font-family:serif;line-height:1.45;font-size:0.95em}\n"
        "    h1,h2,h3{margin:0.6em 0 0.35em; padding:0}\n"
        "    p{margin:0.35em 0}\n"
        "    ul{margin:0.25em 0 0.35em 1.1em; padding:0}\n"
        "    li{margin:0.15em 0}\n"
        "    blockquote{margin:0.5em 0; padding-left:1em; border-left:3px solid #ccc; color:#666; font-style:italic}\n"
        "    hr{border:0; border-top:1px solid #ddd; margin:1em 0}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def convert_md_to_epub(
    md_path: Path,
    epub_path: Path,
    title: str,
    author: str = "",
) -> Path:
    """Convert a markdown file to EPUB.
    
    Args:
        md_path: Path to input markdown file.
        epub_path: Path for output EPUB file.
        title: Book title.
        author: Book author (optional).
        
    Returns:
        Path to the created EPUB file.
    """
    epub_path.parent.mkdir(parents=True, exist_ok=True)

    md = md_path.read_text(encoding="utf-8")
    xhtml = md_to_html(md)

    book_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Files inside epub
    mimetype = "application/epub+zip"
    container_xml = """<?xml version='1.0' encoding='utf-8'?>
<container version='1.0' xmlns='urn:oasis:names:tc:opendocument:xmlns:container'>
  <rootfiles>
    <rootfile full-path='OEBPS/content.opf' media-type='application/oebps-package+xml'/>
  </rootfiles>
</container>
"""

    content_opf = f"""<?xml version='1.0' encoding='utf-8'?>
<package xmlns='http://www.idpf.org/2007/opf' unique-identifier='BookId' version='2.0'>
  <metadata xmlns:dc='http://purl.org/dc/elements/1.1/' xmlns:opf='http://www.idpf.org/2007/opf'>
    <dc:title>{html.escape(title)}</dc:title>
    <dc:language>zh-CN</dc:language>
    <dc:identifier id='BookId'>{book_id}</dc:identifier>
    <dc:creator>{html.escape(author)}</dc:creator>
    <dc:date>{now}</dc:date>
  </metadata>
  <manifest>
    <item id='ncx' href='toc.ncx' media-type='application/x-dtbncx+xml'/>
    <item id='content' href='content.xhtml' media-type='application/xhtml+xml'/>
  </manifest>
  <spine toc='ncx'>
    <itemref idref='content'/>
  </spine>
</package>
"""

    toc_ncx = f"""<?xml version='1.0' encoding='utf-8'?>
<ncx xmlns='http://www.daisy.org/z3986/2005/ncx/' version='2005-1'>
  <head>
    <meta name='dtb:uid' content='{book_id}'/>
    <meta name='dtb:depth' content='1'/>
    <meta name='dtb:totalPageCount' content='0'/>
    <meta name='dtb:maxPageNumber' content='0'/>
  </head>
  <docTitle><text>{html.escape(title)}</text></docTitle>
  <navMap>
    <navPoint id='navPoint-1' playOrder='1'>
      <navLabel><text>{html.escape(title)}</text></navLabel>
      <content src='content.xhtml'/>
    </navPoint>
  </navMap>
</ncx>
"""

    # Write epub (zip) with stored mimetype first
    with zipfile.ZipFile(epub_path, "w") as z:
        z.writestr("mimetype", mimetype, compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        z.writestr("OEBPS/toc.ncx", toc_ncx)
        z.writestr("OEBPS/content.xhtml", xhtml)

    return epub_path


def safe_filename(title: str, max_length: int = 100) -> str:
    """Convert title to a safe filename.
    
    Args:
        title: Original title.
        max_length: Maximum filename length.
        
    Returns:
        Safe filename string.
    """
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe.strip('._')
    
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip('_')
    
    return safe or "untitled"
