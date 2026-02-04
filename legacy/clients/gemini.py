"""Gemini API client for video summarization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import google.generativeai as genai

from core.models import VideoInfo


# Sentinel for failed YouTube search
FAIL_SENTINEL = "无法提取视频信息：未找到可验证的同内容 YouTube 视频"


class GeminiClient:
    """Client for Google Gemini API."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
    ):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def render_prompt(
        self,
        template: str,
        video: VideoInfo,
    ) -> str:
        """Render prompt template with video info.
        
        Args:
            template: Prompt template with placeholders.
            video: Video information.
            
        Returns:
            Rendered prompt string.
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
        """Generate summary for a video.
        
        Args:
            video: Video information.
            prompt_template: The prompt template to use.
            
        Returns:
            Tuple of (summary_text, is_success).
            If no YouTube video found, returns (FAIL_SENTINEL, False).
        """
        prompt = self.render_prompt(prompt_template, video)
        
        try:
            response = self.model.generate_content(prompt)
            
            if not response.text:
                return "Error: Empty response from Gemini", False
            
            text = response.text.strip()
            
            # Check if Gemini couldn't find YouTube video
            if FAIL_SENTINEL in text:
                return FAIL_SENTINEL, False
            
            return text, True
            
        except Exception as e:
            return f"Error: {str(e)}", False
    
    def summarize_with_youtube_url(
        self,
        video: VideoInfo,
        youtube_url: str,
        prompt_template: str,
    ) -> tuple[str, bool]:
        """Generate summary using a known YouTube URL.
        
        Gemini can directly analyze YouTube videos when given the URL.
        
        Args:
            video: Bilibili video information.
            youtube_url: The matching YouTube video URL.
            prompt_template: The prompt template to use.
            
        Returns:
            Tuple of (summary_text, is_success).
        """
        prompt = self.render_prompt(prompt_template, video)
        
        # Add YouTube URL context
        prompt += f"\n\n已找到的 YouTube 视频链接：{youtube_url}"
        
        try:
            response = self.model.generate_content(prompt)
            
            if not response.text:
                return "Error: Empty response from Gemini", False
            
            return response.text.strip(), True
            
        except Exception as e:
            return f"Error: {str(e)}", False
