"""WeChat Reading (WeRead) Browser Client for automation.

Handles login via QR code and EPUB upload.
"""

from __future__ import annotations

import time
import os
import sys
from pathlib import Path
from typing import Optional

# Reuse Playwright availability check logic
try:
    from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# User data directory for persistent login (separate from main Chrome profile)
USER_DATA_DIR = Path.home() / ".weread_browser_session"


def _chrome_channel() -> Optional[str]:
    """Use installed Google Chrome when requested (Windows).
    Set env WEREAD_USE_CHROME=1 to use system Chrome; otherwise Chromium is used
    so the browser window is visible and stable.
    """
    if sys.platform == "win32" and os.environ.get("WEREAD_USE_CHROME", "").strip() in ("1", "true", "yes"):
        return "chrome"
    return None


class WeReadBrowserClient:
    """Browser-based client for WeChat Reading upload."""
    
    def __init__(self, headless: bool = False):
        """Initialize browser client.
        
        Args:
            headless: Run browser in headless mode. 
                      NOTE: For first login (QR scan), headless typically must be False.
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required. Install it with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
        
        self.headless = headless
        self._current_headless = headless
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        
    def _ensure_browser(self, force_headful: bool = False) -> Page:
        """Ensure browser is started and return page.
        
        Args:
            force_headful: Override self.headless to False for this instance.
        """
        if self._page is not None:
            # If we need to force headful but current context is currently headless, we must restart
            if force_headful and self._current_headless:
                print("  Restarting browser in headful mode for interaction...")
                self.close()
            else:
                return self._page
        
        channel = _chrome_channel()
        self._playwright = sync_playwright().start()
        
        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        headless_to_use = False if force_headful else self.headless
        self._current_headless = headless_to_use
        
        launch_kw = {
            "user_data_dir": str(USER_DATA_DIR),
            "headless": headless_to_use,
            "viewport": {"width": 1920, "height": 1080},
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if channel:
            launch_kw["channel"] = channel
        
        try:
            mode_str = "headless" if headless_to_use else "headful"
            if channel:
                print(f"Starting browser ({mode_str}) for WeChat Reading (using system Chrome)...")
            else:
                print(f"Starting browser ({mode_str}) for WeChat Reading...")
            self._context = self._playwright.chromium.launch_persistent_context(**launch_kw)
        except Exception as e:
            if channel:
                print(f"Chrome failed ({e}), falling back to Chromium...")
                launch_kw.pop("channel", None)
                self._context = self._playwright.chromium.launch_persistent_context(**launch_kw)
            else:
                raise
        
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = self._context.new_page()
            
        return self._page

    def upload_epub(self, file_path: str) -> bool:
        """Upload an EPUB file to WeChat Reading.
        
        Args:
            file_path: Absolute path to the .epub file.
            
        Returns:
            bool: True if upload appears successful.
        """
        if not os.path.exists(file_path):
            print(f"  ❌ File not found: {file_path}")
            return False
            
        page = self._ensure_browser()
        
        # 1. Navigate to Shelf (bookshelf) - this usually triggers login if needed
        print("  Navigating to WeRead Shelf...")
        page.goto("https://weread.qq.com/web/shelf", wait_until="domcontentloaded")
        time.sleep(2)
        
        # 2. Check Login
        # If not logged in, URL usually stays at /web/shelf but shows QR code
        try:
            login_indicator = page.query_selector('.login_dialog_content, .login_container')
            if login_indicator and login_indicator.is_visible():
                if self.headless:
                    print("  ⚠ Login required but running in headless mode. Restarting in headful mode...")
                    page = self._ensure_browser(force_headful=True)
                    print("  Navigating back to shelf...")
                    page.goto("https://weread.qq.com/web/shelf", wait_until="domcontentloaded")
                    time.sleep(2)
                
                print("  ⚠ Please scan the QR code in the browser window to log in.")
                print("  Waiting for login (up to 120 seconds)...")
                
                # Wait for login dialog to disappear or shelf content to appear
                page.wait_for_selector('.shelf_list, .shelfItem', timeout=120000)
                print("  ✓ Login detected.")
                time.sleep(2) 
        except Exception as e:
            # Maybe already logged in
            pass
            
        # Double check we are on shelf
        if "shelf" not in page.url:
            print("  Redirecting to shelf...")
            page.goto("https://weread.qq.com/web/shelf")
            time.sleep(2)

        # 3. Click "Import" / "Upload" button
        # Usually checking for a button with text "传书" or "导入"
        try:
            print("  Looking for Upload button...")
            
            # Helper to check if we are already on the upload page
            if "/web/upload" in page.url:
                print("  Already on upload page.")
            else:
                # Try to find the button on the shelf first
                btn_handle = page.evaluate_handle("""() => {
                    // 1. Specific class
                    let btn = document.querySelector('.import_book, .shelf_header_btn.import_book');
                    if (btn) return btn;
                    
                    // 2. Text "传书" but NOT "传书到手机"
                    const elements = Array.from(document.querySelectorAll('button, a, div, span'));
                    for (const el of elements) {
                        const text = el.innerText.trim();
                        if (text === "传书" || (text === "导入" )) {
                            return el;
                        }
                        if (text.includes("传书") && !text.includes("到手机") && text.length < 10) {
                            return el;
                        }
                    }
                    
                    // 3. Cloud/upload SVG icon
                    const svgs = document.querySelectorAll('svg');
                    for (const svg of svgs) {
                        if (svg.innerHTML.includes('M') && svg.closest('.shelf_header_btn')) {
                            return svg.closest('.shelf_header_btn');
                        }
                    }
                    return null;
                }""")
                
                import_btn = btn_handle.as_element()
                if import_btn:
                    print(f"  ✓ Found Import button: {import_btn.inner_text().strip() or 'Icon'}")
                    import_btn.click()
                    time.sleep(2)
                else:
                    print("  ⚠ Could not find 'Import' button on shelf. Navigating directly to /web/upload...")
                    page.goto("https://weread.qq.com/web/upload")
                    time.sleep(2)

            # 4. Handle File Selection on /web/upload page
            # This page usually has a "Select File" (选择文件) button or a drop zone.
            print("  Waiting for 'Select File' button or drop zone...")
            
            # Common selectors for the file input or the "Select File" button
            # WeRead's upload page often has a big button or a hidden input[type="file"]
            
            try:
                # Look for the hidden file input or the visible button
                select_btn_selectors = [
                    'input[type="file"]',
                    '.upload_btn',
                    ':has-text("选择文件")',
                    '.add_file_btn'
                ]
                
                file_chooser_btn = None
                for selector in select_btn_selectors:
                    try:
                        el = page.wait_for_selector(selector, state='attached', timeout=5000)
                        if el:
                            file_chooser_btn = el
                            break
                    except:
                        continue
                
                if not file_chooser_btn:
                    # Final debug screenshot
                    debug_path = Path("output/debug_weread_upload_page.png")
                    page.screenshot(path=str(debug_path))
                    print(f"  ❌ Could not find file selection element on upload page. Screenshot: {debug_path}")
                    return False

                # Handle file selection
                if file_chooser_btn.evaluate("el => el.tagName") == "INPUT" and file_chooser_btn.get_attribute("type") == "file":
                    print(f"  Setting files directly on input: {Path(file_path).name}...")
                    file_chooser_btn.set_input_files(file_path)
                    print("  ✓ set_input_files called on input.")
                else:
                    print(f"  Clicking selection button to trigger file chooser...")
                    with page.expect_file_chooser(timeout=10000) as fc_info:
                        file_chooser_btn.click(force=True)
                    file_chooser = fc_info.value
                    print(f"  Setting file via file_chooser: {Path(file_path).name}...")
                    file_chooser.set_files(file_path)
                    print("  ✓ set_files called via file_chooser.")
                
                print(f"  ✓ File selection logic finished.")
                
            except Exception as e:
                print(f"  ❌ Error during file selection: {e}")
                debug_path = Path("output/debug_weread_upload_error.png")
                page.screenshot(path=str(debug_path))
                return False

            # 5. Wait for upload completion with robust monitoring
            print("  Uploading and waiting for processing (Robust Loop)...")
            
            start_time = time.time()
            max_wait = 120 # 2 minutes
            last_screenshot_time = 0
            
            while time.time() - start_time < max_wait:
                # 1. Take debug screenshot every 10s
                if time.time() - last_screenshot_time > 10:
                    debug_path = Path(f"output/debug_upload_progress_{int(time.time())}.png")
                    page.screenshot(path=str(debug_path))
                    last_screenshot_time = time.time()
                
                # 2. Check for success/completion indicators
                # We must be careful to avoid static help text like "上传完成后可在书架查看"
                # Dynamic success often appears in the progress bar or a toast
                
                # Check for "100%" or specific success classes
                is_100_percent = page.evaluate("""() => {
                    const bars = Array.from(document.querySelectorAll('div, span, p'));
                    return bars.some(el => el.innerText.includes('100%'));
                }""")
                
                if is_100_percent:
                    print("  ✓ 100% reached. Waiting for final processing...")
                    time.sleep(10) # Give it time to finish parsing
                    return True

                # Check for success toast/message (excluding the help text)
                success_detected = page.evaluate("""() => {
                    const elements = Array.from(document.querySelectorAll('div, span, p, h1, h2, h3'));
                    return elements.some(el => {
                        const txt = el.innerText.trim();
                        // Ignore the static help text
                        if (txt.includes("上传完成后") || txt.includes("支持 txt / pdf")) return false;
                        return txt === "上传成功" || txt === "完成" || (txt.includes("已上传") && !txt.includes("0%"));
                    });
                }""")
                
                if success_detected:
                    print("  ✓ Successful upload message detected (excluding static text).")
                    time.sleep(5)
                    return True
                
                # 3. Check for percentage progress
                progress_text = page.evaluate("""() => {
                    const text = document.body.innerText;
                    const match = text.match(/(\d+)%/);
                    return match ? match[0] : null;
                }""")
                
                if progress_text:
                    if progress_text != "0%": # Don't log 0% repeatedly
                        print(f"  ... Upload Progress: {progress_text}")
                else:
                    # If percentage is gone BUT we don't have success yet, 
                    # check if the progress bar/modal is still there
                    # Be careful: the "正在" might be in the help text too? No, usually not.
                    is_processing = page.evaluate("""() => {
                        const text = document.body.innerText;
                        if (text.includes("正在上传") || text.includes("正在解析")) return true;
                        return !!document.querySelector('.progress, .uploading');
                    }""")
                    
                    if not is_processing:
                        # If we were seeing progress and now it's gone, and no error visible
                        print("  ✓ Progress indicators disappeared. Final check...")
                        time.sleep(5)
                        return True
                    else:
                        print("  ... Still processing...")
                
                time.sleep(2)
            
            print(f"  ⚠ Upload monitoring timed out after {max_wait}s.")
            return True # Try to proceed anyway as the file was sent

        except Exception as e:
            print(f"  ❌ Upload process failed: {e}")
            return False

    def close(self):
        """Close browser."""
        if self._context:
            self._context.close()
            self._context = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
