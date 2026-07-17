"""Regression test: a single unreadable audio file must not abort the whole
batch.

A truncated/corrupt m4a (e.g. a partial download) used to raise inside the
audio-loading loop of `transcribe_files_batch`, propagate all the way up, and
cause `step_asr` to mark the entire group of 8 videos as `error` — including
every healthy file. The loader now isolates per-file failures: the bad file
yields an empty result while the good ones transcribe normally.
"""
from clients.qwen_asr_client import Qwen3ASRClient


class _FakeResult:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    """Returns one result per chunk passed in, with deterministic text."""

    def transcribe(self, audio=None, language=None, context=None):
        return [_FakeResult("ok") for _ in audio]


def _make_client():
    # __init__ is lazy — constructing the client loads no model.
    return Qwen3ASRClient(diarize=False, convert_to_simplified=False)


def test_one_bad_file_does_not_kill_the_batch(tmp_path, monkeypatch):
    good = tmp_path / "good.m4a"
    bad = tmp_path / "bad.m4a"
    good.write_bytes(b"x")
    bad.write_bytes(b"x")

    import numpy as np

    def fake_load(self, audio_path):
        if audio_path.name == "bad.m4a":
            raise RuntimeError()  # empty-message exception, like librosa on corrupt audio
        return np.zeros(16000, dtype=np.float32), 16000

    client = _make_client()
    client._model = _FakeModel()  # bypass the lazy-loading `model` property
    monkeypatch.setattr(Qwen3ASRClient, "_load_audio_mono_16k", fake_load)

    texts = client.transcribe_files_batch([good, bad])

    assert len(texts) == 2
    assert texts[0] == "ok"   # healthy file still transcribed
    assert texts[1] == ""     # bad file isolated to an empty result, no crash


def test_call_with_timeout_returns_value():
    assert Qwen3ASRClient._call_with_timeout(5, lambda: 42) == 42


def test_call_with_timeout_propagates_error():
    import pytest

    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        Qwen3ASRClient._call_with_timeout(5, boom)


def test_call_with_timeout_raises_on_hang():
    import time
    import pytest

    # A model load that never returns must surface as TimeoutError rather than
    # freezing the pipeline forever (the Step C hang this guards against).
    with pytest.raises(TimeoutError):
        Qwen3ASRClient._call_with_timeout(0.2, lambda: time.sleep(30))


def test_long_audio_transcription_is_chunked_with_offset_timestamps(monkeypatch):
    """A long multi-speaker audio must be transcribed in chunks (not one giant
    forward pass that thrashes VRAM and stalls), with each chunk's timestamps
    offset to absolute time so the diarization overlay still lines up.
    """
    import clients.qwen_asr_client as qa
    import numpy as np

    monkeypatch.setattr(qa, "MAX_CHUNK_SECONDS", 1)  # 1s chunks for the test

    class _Item:
        def __init__(self, text, s, e):
            self.text, self.start_time, self.end_time = text, s, e

    class _Res:
        def __init__(self, text, items):
            self.text, self.time_stamps = text, items

    calls = {"n": 0}

    class _Model:
        def transcribe(
            self,
            audio=None,
            language=None,
            context=None,
            return_time_stamps=False,
        ):
            calls["n"] += 1
            i = calls["n"]
            # one local item per chunk at 0.1-0.2s into the chunk
            return [_Res(f"x{i}", [_Item(f"x{i}", 0.1, 0.2)])]

    client = _make_client()
    client._model = _Model()
    sr = 16000
    y = np.zeros(sr * 3, dtype=np.float32)  # 3s of audio → 3 chunks of 1s

    text, items = client._transcribe_with_timestamps_chunked(y, sr, 3.0)

    assert calls["n"] == 3                 # chunked, not one giant pass
    assert text == "x1x2x3"                # texts concatenated in order
    assert [round(it.start_time, 1) for it in items] == [0.1, 1.1, 2.1]  # offset to absolute time
    assert [round(it.end_time, 1) for it in items] == [0.2, 1.2, 2.2]


def test_chunk_checkpoints_resume_without_retranscribing(tmp_path, monkeypatch):
    import clients.qwen_asr_client as qa
    import numpy as np

    monkeypatch.setattr(qa, "MAX_CHUNK_SECONDS", 1)

    class _Item:
        def __init__(self, text, s, e):
            self.text, self.start_time, self.end_time = text, s, e

    class _Res:
        def __init__(self, text):
            self.text = text
            self.time_stamps = [_Item(text, 0.1, 0.2)]

    calls = {"n": 0}

    class _Model:
        def transcribe(self, **kwargs):
            calls["n"] += 1
            return [_Res(f"x{calls['n']}")]

    client = _make_client()
    client._model = _Model()
    y = np.zeros(16000 * 3, dtype=np.float32)
    checkpoint_dir = tmp_path / "checkpoint"

    first = client._transcribe_with_timestamps_chunked(
        y, 16000, 3.0, checkpoint_dir=checkpoint_dir
    )
    assert calls["n"] == 3

    class _MustNotRun:
        def transcribe(self, **kwargs):
            raise AssertionError("completed chunks should be loaded from checkpoint")

    client._model = _MustNotRun()
    second = client._transcribe_with_timestamps_chunked(
        y, 16000, 3.0, checkpoint_dir=checkpoint_dir
    )

    assert second[0] == first[0]
    assert [(i.text, i.start_time, i.end_time) for i in second[1]] == [
        (i.text, i.start_time, i.end_time) for i in first[1]
    ]


def test_empty_long_batch_chunk_marks_file_incomplete(tmp_path, monkeypatch):
    import clients.qwen_asr_client as qa
    import numpy as np

    monkeypatch.setattr(qa, "MAX_CHUNK_SECONDS", 1)
    monkeypatch.setattr(qa, "MIN_NONEMPTY_CHUNK_SECONDS", 1)

    class _Res:
        def __init__(self, text):
            self.text = text

    class _Model:
        def transcribe(self, audio=None, **kwargs):
            return [_Res("ok" if i == 0 else "") for i, _ in enumerate(audio)]

    def fake_load(self, audio_path):
        return np.zeros(16000 * 40, dtype=np.float32), 16000

    audio = tmp_path / "long.m4a"
    audio.write_bytes(b"x")

    client = _make_client()
    client._model = _Model()
    monkeypatch.setattr(Qwen3ASRClient, "_load_audio_mono_16k", fake_load)

    assert client.transcribe_files_batch([audio]) == [""]


def test_empty_timestamp_chunk_refuses_partial_transcript(monkeypatch):
    import clients.qwen_asr_client as qa
    import numpy as np
    import pytest

    monkeypatch.setattr(qa, "MAX_CHUNK_SECONDS", 1)
    monkeypatch.setattr(qa, "MIN_NONEMPTY_CHUNK_SECONDS", 1)

    class _Res:
        text = ""
        time_stamps = []

    class _Model:
        def transcribe(self, **kwargs):
            return [_Res()]

    client = _make_client()
    client._model = _Model()
    y = np.zeros(16000 * 40, dtype=np.float32)

    with pytest.raises(RuntimeError, match="refusing partial transcript"):
        client._transcribe_with_timestamps_chunked(y, 16000, 40.0)


def test_asr_context_and_overlap_helpers():
    context = Qwen3ASRClient.build_context("Qwen3-ASR 深度解析", "测试UP主")
    assert "Qwen3-ASR" in context
    assert "测试UP主" in context
    assert Qwen3ASRClient._merge_chunk_text(
        "前文在这里继续讨论专有名词",
        "讨论专有名词以及后续内容",
    ) == "前文在这里继续讨论专有名词以及后续内容"
