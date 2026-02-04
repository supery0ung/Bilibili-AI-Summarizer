#!/usr/bin/env python3
"""Test WeChat Reading upload with existing EPUB."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Direct import to avoid circular import
import importlib.util
spec = importlib.util.spec_from_file_location(
    "weread_browser", 
    Path(__file__).parent / "clients" / "weread_browser.py"
)
weread_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weread_module)
WeReadBrowserClient = weread_module.WeReadBrowserClient

# Find EPUB
epub_dir = Path("E:/bilibili_summarizer_v3/output/epub")
epubs = list(epub_dir.glob("*.epub"))

if not epubs:
    print("No EPUB files found!")
    sys.exit(1)

epub_path = epubs[0]
print(f"Testing upload with: {epub_path.name}")
print(f"File size: {epub_path.stat().st_size / 1024:.1f} KB")

# Upload (non-headless to see the browser)
weread = WeReadBrowserClient(headless=False)
try:
    print("\nStarting upload... (browser window will open)")
    print("Please scan QR code if prompted to login.")
    success = weread.upload_epub(str(epub_path))
    
    if success:
        print("\n✓ Upload successful!")
    else:
        print("\n✗ Upload failed")
finally:
    try:
        input("\nPress Enter to close browser (or wait 5s auto-close)...")
    except EOFError:
        pass
    import time
    time.sleep(5)  # Give user a moment to see result
    weread.close()
    print("Browser closed.")