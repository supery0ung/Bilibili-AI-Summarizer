"""Unit tests for the transcribe-then-label speaker assignment logic.

These exercise the pure alignment helpers (no GPU / no model load): given an
original transcription, aligned per-char timestamps, and diarization turns, the
original text must be split at speaker-change boundaries with punctuation kept.
"""
from dataclasses import dataclass

from clients.qwen_asr_client import Qwen3ASRClient, SpeakerTurn


@dataclass
class FakeItem:
    text: str
    start_time: float
    end_time: float


def make_client() -> Qwen3ASRClient:
    # __init__ is lazy — no model is loaded just by constructing the client.
    return Qwen3ASRClient(diarize=True, convert_to_simplified=False)


def test_speaker_at_covering_and_nearest():
    c = make_client()
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.0),
        SpeakerTurn(speaker="SPEAKER_01", start=2.0, end=3.0),
    ]
    assert c._speaker_at(0.5, turns) == "SPEAKER_00"   # inside A
    assert c._speaker_at(2.5, turns) == "SPEAKER_01"   # inside B
    assert c._speaker_at(1.1, turns) == "SPEAKER_00"   # gap, nearer A
    assert c._speaker_at(1.9, turns) == "SPEAKER_01"   # gap, nearer B


def test_assign_splits_text_at_speaker_change_keeping_punctuation():
    c = make_client()
    text = "甲说话。乙说话。"
    items = [
        FakeItem("甲", 0.1, 0.2),
        FakeItem("说", 0.3, 0.4),
        FakeItem("话", 0.5, 0.6),
        FakeItem("乙", 2.1, 2.2),
        FakeItem("说", 2.3, 2.4),
        FakeItem("话", 2.5, 2.6),
    ]
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=1.0),
        SpeakerTurn(speaker="SPEAKER_01", start=2.0, end=3.0),
    ]
    segs = c._assign_speakers_to_text(text, items, turns)
    assert len(segs) == 2
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[0].text == "甲说话。"   # punctuation stays with preceding speaker
    assert segs[1].speaker == "SPEAKER_01"
    assert segs[1].text == "乙说话。"


def test_assign_single_speaker_one_segment():
    c = make_client()
    text = "全程一个人讲话。没有切换。"
    items = [FakeItem(ch, i * 0.2, i * 0.2 + 0.1) for i, ch in enumerate("全程一个人讲话没有切换")]
    turns = [SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=10.0)]
    segs = c._assign_speakers_to_text(text, items, turns)
    assert len(segs) == 1
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[0].text == text


def test_assign_preserves_full_text_concatenation():
    """No characters should be dropped across the split."""
    c = make_client()
    text = "你好世界，今天天气不错呀！我们出去走走吧。"
    items = [FakeItem(ch, i * 0.5, i * 0.5 + 0.3) for i, ch in enumerate(
        [ch for ch in text if ch not in "，！。"]
    )]
    # speaker switches halfway through (by time)
    half = items[len(items) // 2].start_time
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=half),
        SpeakerTurn(speaker="SPEAKER_01", start=half, end=items[-1].end_time + 1),
    ]
    segs = c._assign_speakers_to_text(text, items, turns)
    # concatenating segment texts (stripped) must reproduce the original text
    assert "".join(s.text for s in segs) == text


def test_assign_empty_items_returns_empty():
    c = make_client()
    assert c._assign_speakers_to_text("anything", [], []) == []
