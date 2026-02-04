"""Gemini Browser-based client for video summarization.

Uses Playwright to automate Chrome browser interaction with gemini.google.com.
This avoids API costs by using the free web interface.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from core.models import VideoInfo

# Playwright is optional - only required for browser mode
try:
    from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# Same sentinel as API client for consistency
FAIL_SENTINEL = "无法提取视频信息：未找到可验证的同内容 YouTube 视频"

# User data directory for persistent login
USER_DATA_DIR = Path.home() / ".gemini_browser_session"


class GeminiBrowserClient:
    """Browser-based client for Gemini web interface."""
    
    def __init__(self, headless: bool = False):
        """Initialize browser client.
        
        Args:
            headless: Run browser in headless mode (default: False for login visibility).
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for browser mode. Install it with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )
        
        self.headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        
    def _ensure_browser(self) -> Page:
        """Ensure browser is started and return page."""
        if self._page is not None:
            return self._page
        
        print("Starting browser...")
        self._playwright = sync_playwright().start()
        
        # Use persistent context to save login session
        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=self.headless,
            viewport={"width": 1280, "height": 900},
            permissions=["clipboard-read", "clipboard-write"],  # Grant clipboard permissions
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )
        
        # Get the first page or create one
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = self._context.new_page()
            
        return self._page
    
    def render_prompt(self, template: str, video: VideoInfo) -> str:
        """Render prompt template with video info.
        
        Same interface as GeminiClient for compatibility.
        """
        return (
            template
            .replace("{{BILIBILI_TITLE}}", video.title or "")
            .replace("{{UP_NAME}}", video.up_name or "")
            .replace("{{BILIBILI_URL}}", video.url or "")
            .replace("{{BVID}}", video.bvid or "")
            .replace("{{DURATION_SECONDS}}", str(video.duration or ""))
        )
    
    def summarize_video(
        self,
        video: VideoInfo,
        prompt_template: str,
    ) -> tuple[str, bool]:
        """Generate summary for a video using Gemini web interface.
        
        Args:
            video: Video information.
            prompt_template: The prompt template to use.
            
        Returns:
            Tuple of (summary_text, is_success).
        """
        prompt = self.render_prompt(prompt_template, video)
        
        # Debug: Save prompt to file
        try:
            with open("last_browser_prompt.md", "w", encoding="utf-8") as f:
                f.write(prompt)
            print("  (Debug: Rendered prompt saved to 'last_browser_prompt.md')")
        except Exception as e:
            print(f"  (Debug: Failed to save prompt: {e})")
        
        try:
            page = self._ensure_browser()
            
            # Navigate to Gemini
            current_url = page.url
            if "gemini.google.com" not in current_url:
                print("  Navigating to Gemini...")
                try:
                    page.goto("https://gemini.google.com/app", timeout=60000, wait_until="domcontentloaded")
                except Exception as e:
                    print(f"  ⚠ Navigation warning: {e}")
                
                # Wait for page to settle
                time.sleep(3)
            
            # Check if logged in by looking for the prompt input
            # Wait for either login button or chat input
            try:
                # Try to find the prompt textarea
                prompt_area = page.wait_for_selector(
                    'div[contenteditable="true"], textarea[placeholder*="Enter"], .ql-editor',
                    timeout=30000
                )
                if not prompt_area:
                    # If selector not found but no error raised (unlikely), check login
                    pass
            except Exception:
                # May need to login
                print("  ⚠ Please log in to Google in the browser window...")
                print("  Waiting for login (up to 120 seconds)...")
                page.wait_for_selector(
                    'div[contenteditable="true"], textarea[placeholder*="Enter"], .ql-editor',
                    timeout=120000
                )
            
            # Click "New chat" button if exists to start fresh
            try:
                new_chat_btn = page.query_selector('button[aria-label*="New chat"], a[href="/app"], span[data-test-id="new-chat-button"]')
                if new_chat_btn:
                    new_chat_btn.click()
                    time.sleep(1)
            except Exception:
                pass

            # Model Selection: "Thinking" (JS Automated & Verified)
            try:
                # Wait for body just in case
                try: page.wait_for_selector("body", timeout=5000)
                except: pass

                # 1. Open Menu
                js_click_picker = """
                () => {
                    const keywords = ['Fast', 'Gemini 2.0 Flash', 'Gemini Advanced', 'Gemini'];
                    const isVisible = (elem) => !!(elem.offsetWidth || elem.offsetHeight || elem.getClientRects().length);
                    const buttons = Array.from(document.querySelectorAll('button, [role="button"], .mat-mdc-button-base'));
                    
                    // Priority: Exact "Fast" (or standard Gemini)
                    for (const btn of buttons) {
                        if (isVisible(btn)) {
                            const txt = btn.innerText.trim();
                            if (txt === 'Fast' || txt === 'Gemini') {
                                btn.click();
                                return 'Clicked ' + txt;
                            }
                        }
                    }
                    // Fuzzy match
                    for (const btn of buttons) {
                        if (!isVisible(btn)) continue;
                        const text = btn.innerText || '';
                        if (text.includes('Enter a prompt')) continue;
                        if (keywords.some(k => text.includes(k))) {
                            btn.click();
                            return 'Clicked ' + text.substring(0, 20);
                        }
                    }
                    return null;
                }
                """
                
                # Retry logic for clicking the picker
                picker_clicked = False
                for _ in range(2):
                    res = page.evaluate(js_click_picker)
                    if res:
                        picker_clicked = True
                        break
                    time.sleep(1)
                
                if picker_clicked:
                    time.sleep(2) # Wait for menu to fully render

                    # 2. Click "Thinking" and Verify
                    js_click_thinking = """
                    () => {
                        const candidates = Array.from(document.querySelectorAll('li, [role="menuitem"], span, div'));
                        const isVisible = (e) => !!(e.offsetWidth || e.offsetHeight);
                        
                        // Filter for "Thinking", preventing container clicks
                        const targets = candidates.filter(e => 
                            isVisible(e) && 
                            (e.innerText || '').includes('Thinking') &&
                            !e.classList.contains('cdk-overlay-container') && 
                            !e.classList.contains('cdk-global-overlay-wrapper') &&
                            !e.classList.contains('mat-mdc-menu-panel')
                        );
                        
                        if (targets.length > 0) {
                            // Sort by text length
                            targets.sort((a, b) => a.innerText.length - b.innerText.length);
                            
                            let best = targets[0];
                            const menuItems = targets.filter(t => t.getAttribute('role') === 'menuitem');
                            if (menuItems.length > 0 && menuItems[0].innerText.length < 50) {
                                 best = menuItems[0];
                            }
                            
                            best.click();
                            return "Clicked: " + best.tagName;
                        }
                        return null;
                    }
                    """
                    
                    res = page.evaluate(js_click_thinking)
                    if res:
                        time.sleep(3) # Wait for UI update
                        
                        # 3. VERIFY
                        js_check_status = """
                        () => {
                            const buttons = Array.from(document.querySelectorAll('button, [role="button"]'));
                            const isVisible = (e) => !!(e.offsetWidth || e.offsetHeight);
                            for (const btn of buttons) {
                                 if (!isVisible(btn)) continue;
                                 if (btn.innerText.includes('Thinking')) return true;
                            }
                            return false;
                        }
                        """
                        if page.evaluate(js_check_status):
                            pass # Verification success
                        else:
                            print("  ⚠ Verification warning: Active model might not be 'Thinking'.")

            except Exception as e:
                print(f"  ⚠ Auto-selection warning: {e}")
            
            time.sleep(1)
            input_selectors = [
                'div.ql-editor[contenteditable="true"]',
                'div[contenteditable="true"][aria-label*="prompt"]',
                'div[contenteditable="true"]',
                'rich-textarea div[contenteditable="true"]',
            ]
            
            input_element = None
            for selector in input_selectors:
                try:
                    input_element = page.wait_for_selector(selector, timeout=5000)
                    if input_element:
                        break
                except Exception:
                    continue
            
            if not input_element:
                return "Error: Could not find prompt input element", False
            
            # Verify length
            # Strategy: Prefer Clipboard Paste for large texts
            print(f"  Preparing to input prompt ({len(prompt)} chars)...")
            
            success = False
            
            # Application 1: Clipboard Paste (Primary)
            try:
                print("  Attempting clipboard paste...")
                # Clear again to be safe
                input_element.click()
                for _ in range(2):
                    page.keyboard.press("Control+a")
                    page.keyboard.press("Meta+a")
                    page.keyboard.press("Backspace")
                time.sleep(0.5)

                escaped_prompt = prompt.replace('`', '\\`').replace('$', '\\$')
                page.evaluate(f"navigator.clipboard.writeText(`{escaped_prompt}`)")
                time.sleep(0.5)
                
                input_element.focus()
                page.keyboard.press("Meta+v") # Paste
                time.sleep(1.5)
                
                # Verify
                current_text = input_element.inner_text()
                # Allow small difference due to whitespace formatting (tabs vs spaces etc)
                if len(current_text) > len(prompt) * 0.95:
                    print(f"  ✓ Paste successful (Input length: {len(current_text)})")
                    success = True
                else:
                    print(f"  ⚠ Paste verification failed! Expected ~{len(prompt)}, got {len(current_text)}")
            except Exception as e:
                print(f"  ⚠ Clipboard paste failed: {e}")
            
            # Strategy 2: Fill (Fallback)
            if not success:
                print("  Falling back to direct fill...")
                try:
                    input_element.click()
                    # Clear again
                    page.keyboard.press("Control+a")
                    page.keyboard.press("Meta+a")
                    page.keyboard.press("Backspace")
                    time.sleep(0.5)
                    
                    input_element.fill(prompt)
                    time.sleep(1)
                    
                    current_text = input_element.inner_text()
                    print(f"  Fill completed. Input length: {len(current_text)}")
                    if len(current_text) > len(prompt) * 0.9:
                        success = True
                    else:
                        print("  ⚠ Fill also resulted in truncated text.")
                except Exception as e:
                    print(f"  ⚠ Fill failed: {e}")

            time.sleep(1)
            
            if not success:
                print("  ⚠ Proceeding despite potential prompt truncation...")
            
            # Find and click send button
            send_selectors = [
                'button[aria-label*="Send"]',
                'button[aria-label*="发送"]',
                'button.send-button',
                'button[data-test-id="send-button"]',
            ]
            
            send_btn = None
            for selector in send_selectors:
                try:
                    send_btn = page.query_selector(selector)
                    if send_btn and send_btn.is_visible():
                        break
                except Exception:
                    continue
            
            if not send_btn:
                # Try pressing Enter instead
                page.keyboard.press("Enter")
            else:
                send_btn.click()
            
            print("  Waiting for Gemini response...")
            
            # Wait for response to complete
            # Look for the response container and wait for it to stop updating
            time.sleep(3)  # Initial wait for response to start
            
            # Wait for response to finish (stop button disappears or content stabilizes)
            max_wait = 180  # 3 minutes max
            start_time = time.time()
            last_content = ""
            stable_count = 0
            
            while time.time() - start_time < max_wait:
                # Try to get response content
                response_selectors = [
                    '.model-response-text',
                    '.response-content',
                    'message-content.model-response-text',
                    '.markdown-main-panel',
                    'div[data-content-type="response"]',
                ]
                
                current_content = ""
                for selector in response_selectors:
                    try:
                        elements = page.query_selector_all(selector)
                        if elements:
                            # Get the last response (most recent)
                            current_content = elements[-1].inner_text()
                            if current_content:
                                break
                    except Exception:
                        continue
                
                # Check if content is stable (stopped generating)
                if current_content and current_content == last_content:
                    stable_count += 1
                    if stable_count >= 3:  # Stable for 3 checks = done
                        break
                else:
                    stable_count = 0
                    last_content = current_content
                
                time.sleep(2)
            
            # Extract final response
            response_text = ""
            for selector in ['.model-response-text', '.response-content', '.markdown-main-panel']:
                try:
                    elements = page.query_selector_all(selector)
                    if elements:
                        response_text = elements[-1].inner_text()
                        if response_text:
                            break
                except Exception:
                    continue
            
            if not response_text:
                return "Error: Could not extract response from Gemini", False
            
            # Check for failure sentinel
            if FAIL_SENTINEL in response_text:
                return FAIL_SENTINEL, False
            
            return response_text.strip(), True
            
        except Exception as e:
            return f"Error: {str(e)}", False
    
    def close(self):
        """Clean up browser resources."""
        if self._context:
            self._context.close()
            self._context = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._page = None
        print("Browser closed.")
