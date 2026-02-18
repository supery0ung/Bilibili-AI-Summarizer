"""Tests for bilingual correction logic in OllamaClient."""

import pytest
from unittest.mock import MagicMock, patch
from clients.ollama_client import OllamaClient
from pathlib import Path

@pytest.fixture
def mock_ollama_client(tmp_path):
    # Mock prompt loading
    with patch("clients.ollama_client.OllamaClient._load_prompt") as mock_load:
        # Minimal prompt for testing
        mock_load.return_value = "Title: {title}\nExtra: {extra_instructions}\nText: {text}"
        client = OllamaClient(base_url="http://localhost:11434")
        return client

def test_bilingual_hint_triggered_for_english(mock_ollama_client):
    client = mock_ollama_client
    
    # Mock the generate call to see what prompt it receives
    with patch.object(client, "generate", return_value="English text\n中文翻译") as mock_generate:
        result = client.correct_text_batched(
            "English text", # Input text to match expected mocked behavior (identity keep)
            title="Test",
            author="User",
            language="en"
        )
        
        # Check if the prompt contained the bilingual requirement
        args, kwargs = mock_generate.call_args
        prompt = args[0]
        assert "强制性双语要求" in prompt
        assert "Original Paragraph" in prompt
        # The result should contain both if LLM returned both
        assert "English text" in result
        assert "中文翻译" in result

def test_no_bilingual_hint_for_chinese(mock_ollama_client):
    client = mock_ollama_client
    
    with patch.object(client, "generate", return_value="修正后的中文") as mock_generate:
        client.correct_text_batched(
            "你好世界",
            title="测试",
            author="用户",
            language="zh"
        )
        
        args, kwargs = mock_generate.call_args
        prompt = args[0]
        assert "强制性双语要求" not in prompt

def test_language_heuristic_in_qwen_client():
    from clients.qwen_asr_client import Qwen3ASRClient
    
    # Use object.__new__ to bypass __init__ which calls _verify_connection
    client = object.__new__(Qwen3ASRClient)
    
    # English text
    assert client._detect_language_heuristic("This is clearly english text without any CJK.") == "en"
    
    # Chinese text
    assert client._detect_language_heuristic("这是明显的中文文本。") == "zh"
    
    # Japanese text
    assert client._detect_language_heuristic("これは日本語のテキストです。") == "ja"
    
    # Empty or missing
    assert client._detect_language_heuristic("") == "zh"
