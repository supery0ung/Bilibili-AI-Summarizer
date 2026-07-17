"""Tests for the ASR sizing env-override helper.

The scheduled runner (scripts/run_and_hibernate.py) shrinks the ASR forward-pass
size on a retry by setting ASR_MAX_CHUNK_SECONDS / ASR_BATCH_SIZE /
ASR_BATCH_MAX_TOTAL_SECONDS in the child's environment. These guard that the
override helper honours valid values and ignores junk (so a bad env var can never
silently zero out the chunk size). See memory: asr-hang-rootcause.
"""
import pytest

from clients.qwen_asr_client import _env_int


def test_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("ASR_X", raising=False)
    assert _env_int("ASR_X", 180) == 180


def test_reads_valid_override(monkeypatch):
    monkeypatch.setenv("ASR_X", "120")
    assert _env_int("ASR_X", 180) == 120


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "", "12.5"])
def test_falls_back_on_invalid(monkeypatch, bad):
    monkeypatch.setenv("ASR_X", bad)
    assert _env_int("ASR_X", 180) == 180
