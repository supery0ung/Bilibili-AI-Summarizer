import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from clients.bilibili_api_client import BilibiliClient
from clients.downloader import VideoDownloader
from clients.ollama_client import OllamaClient
from clients.weread_browser import WeReadBrowserClient
from core.models import QueueItem
from core.pipeline import Pipeline
from core.step_asr import LONG_VIDEO_SECONDS, StepASR
from core.step_llm import StepLLM


def _item(bvid: str, duration: int = 300) -> QueueItem:
    return QueueItem(
        bvid=bvid,
        title=bvid,
        url=f"https://www.bilibili.com/video/{bvid}",
        duration=duration,
        up_name="author",
    )


def test_run_all_persists_only_the_fixed_current_batch(tmp_path):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = {"pipeline": {"max_items_per_run": 2}}
    pipeline.logger = logging.getLogger("test.manifest")
    pipeline.queue_file = tmp_path / "queue.json"
    pipeline._ollama = SimpleNamespace(unload_model=lambda: None)
    pipeline._asr_client = SimpleNamespace(unload_model=lambda: None)
    pipeline.state = SimpleNamespace(
        get_status=lambda bvid: "uploaded",
    )
    source = [_item("BV1"), _item("BV2"), _item("BV3")]
    pipeline.run_step_a = lambda manual_urls=None: source
    pipeline.run_step_ba_ai_filter = lambda max_items=None: {}
    pipeline.run_step_b_download = lambda max_items=None: ({}, [])
    pipeline.run_step_c_transcribe = lambda max_items=None: ({}, [])
    pipeline.run_step_d_correct = lambda max_items=None, **kwargs: ({}, [])
    pipeline.run_step_e_summarize = lambda max_items=None, **kwargs: ({}, [])
    captured = {}
    pipeline.run_step_f_epub = lambda target_bvids=None: (
        captured.update({"epub_targets": target_bvids}) or {},
        [],
    )
    pipeline.cleanup_debug_files = lambda: None

    result = Pipeline.run_all(pipeline, max_items=2, upload=False)
    manifest = json.loads(pipeline.queue_file.read_text(encoding="utf-8"))

    assert result["queued"] == 2
    assert manifest["run_id"] == result["run_id"]
    assert [row["bvid"] for row in manifest["queue"]] == ["BV1", "BV2"]
    assert "epub_targets" not in captured


def test_run_all_feeds_transcript_backlog_to_downstream_steps(tmp_path):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = {"pipeline": {"max_items_per_run": 10}}
    pipeline.logger = logging.getLogger("test.downstream-backlog")
    pipeline.queue_file = tmp_path / "queue.json"
    pipeline._ollama = SimpleNamespace(unload_model=lambda: None)
    pipeline._asr_client = SimpleNamespace(unload_model=lambda: None)

    state_data = {
        "videos": {
            "BVready": {
                "status": "transcript_ready",
                "title": "ready",
                "up_name": "author",
                "last_attempt": "2026-06-01T00:00:00Z",
            },
            "BVcorrected": {
                "status": "corrected",
                "title": "corrected",
                "up_name": "author",
                "last_attempt": "2026-06-02T00:00:00Z",
            },
        }
    }

    class _State:
        _state = state_data

        def get_status(self, bvid):
            return self._state["videos"].get(bvid, {"status": "uploaded"})["status"]

        def set_status(self, bvid, status):
            self._state["videos"][bvid]["status"] = status

    pipeline.state = _State()
    pipeline.run_step_a = lambda manual_urls=None: [_item("BVnew1"), _item("BVnew2")]
    pipeline.run_step_ba_ai_filter = lambda max_items=None: {}
    pipeline.run_step_b_download = lambda max_items=None: ({}, [])
    pipeline.run_step_c_transcribe = lambda max_items=None: ({}, [])
    captured = {}

    def fake_correct(max_items=None, **kwargs):
        queue = json.loads(pipeline.queue_file.read_text(encoding="utf-8"))["queue"]
        captured["correct_queue"] = [row["bvid"] for row in queue]
        pipeline.state.set_status("BVready", "corrected")
        return {}, ["BVready"]

    def fake_summarize(max_items=None, **kwargs):
        queue = json.loads(pipeline.queue_file.read_text(encoding="utf-8"))["queue"]
        captured["summarize_queue"] = [row["bvid"] for row in queue]
        pipeline.state.set_status("BVready", "summarized")
        pipeline.state.set_status("BVcorrected", "summarized")
        return {}, ["BVready", "BVcorrected"]

    pipeline.run_step_d_correct = fake_correct
    pipeline.run_step_e_summarize = fake_summarize
    pipeline.run_step_f_epub = lambda target_bvids=None: (
        captured.update({"epub_targets": target_bvids}) or {},
        ["BVready", "BVcorrected"],
    )
    pipeline.run_step_g_upload = lambda **kwargs: (
        captured.update({"upload_only": kwargs["only_bvids"]}) or {}
    )
    pipeline.cleanup_debug_files = lambda: None

    Pipeline.run_all(pipeline, max_items=10, upload=True)

    assert captured["correct_queue"] == ["BVready"]
    assert captured["summarize_queue"] == ["BVcorrected", "BVready"]
    assert set(captured["epub_targets"]) == {"BVcorrected", "BVready"}
    assert set(captured["upload_only"]) == {"BVcorrected", "BVready"}


def test_long_videos_are_isolated_after_short_groups(tmp_path):
    pending = [
        (_item("long", LONG_VIDEO_SECONDS), tmp_path / "long.m4a"),
        (_item("short1"), tmp_path / "short1.m4a"),
        (_item("short2"), tmp_path / "short2.m4a"),
    ]

    groups = StepASR._build_groups(pending)

    assert [item.bvid for item, _ in groups[0]] == ["short1", "short2"]
    assert [item.bvid for item, _ in groups[1]] == ["long"]


def test_bilibili_requests_use_bounded_timeout(monkeypatch):
    client = BilibiliClient("s", "j", "u", "b")
    seen = {}

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"code": 0, "data": {"isLogin": True}}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return _Response()

    monkeypatch.setattr(client.session, "get", fake_get)
    assert client.check_auth()
    assert seen["timeout"] == (10, 30)


def test_runtime_ytdlp_update_is_disabled(monkeypatch):
    downloader = VideoDownloader.__new__(VideoDownloader)
    monkeypatch.setattr(
        "clients.downloader.subprocess.run",
        lambda *args, **kwargs: pytest.fail("runtime pip update must not run"),
    )
    assert downloader._update_ytdlp() is False


def test_bilibili_download_cmd_adds_412_workaround_headers():
    downloader = VideoDownloader.__new__(VideoDownloader)
    downloader.ytdlp_path = "yt-dlp"
    downloader.audio_only = True
    downloader.cookies_browser = None
    downloader.cookies_file = None
    downloader.ffmpeg_location = None

    cmd = downloader._build_cmd(
        "https://www.bilibili.com/video/BV1MUjX6uEE8",
        "out.%(ext)s",
    )

    assert "-f" in cmd
    assert "16/bestaudio/best[height<=360]/worst" in cmd
    assert "Origin:https://www.bilibili.com" in cmd
    assert "Referer:https://www.bilibili.com/" in cmd


def test_non_bilibili_download_cmd_keeps_default_format_selection():
    downloader = VideoDownloader.__new__(VideoDownloader)
    downloader.ytdlp_path = "yt-dlp"
    downloader.audio_only = True
    downloader.cookies_browser = None
    downloader.cookies_file = None
    downloader.ffmpeg_location = None

    cmd = downloader._build_cmd("https://example.com/video", "out.%(ext)s")

    assert "16/bestaudio/best[height<=360]/worst" not in cmd
    assert "Origin:https://www.bilibili.com" not in cmd


def test_ollama_summary_timeout_raises(monkeypatch):
    client = OllamaClient.__new__(OllamaClient)
    client.model = "test"
    client.base_url = "http://localhost"
    client.prompts_dir = Path(__file__).parent.parent / "prompts"
    monkeypatch.setattr(
        client,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout()),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        client.summarize("long enough source text", title="title", author="author")


def test_ollama_generate_sends_context_and_keep_alive(monkeypatch):
    client = OllamaClient.__new__(OllamaClient)
    client.model = "test"
    client.base_url = "http://localhost"
    client.keep_alive = "30m"
    captured = {}

    class _Response:
        content = json.dumps({"response": "ok"}).encode("utf-8")

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, **kwargs):
        captured.update(json)
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    assert client.generate("prompt", num_ctx=12288) == "ok"
    assert captured["keep_alive"] == "30m"
    assert captured["options"]["num_ctx"] == 12288


def test_long_summary_uses_hierarchical_synthesis(monkeypatch):
    client = OllamaClient.__new__(OllamaClient)
    client.prompts_dir = Path(__file__).parent.parent / "prompts"
    client.hierarchical_summary_chars = 20
    client.summary_chunk_chars = 12
    client.correction_num_ctx = 12288
    client.summary_num_ctx = 32768
    prompts = []

    def fake_generate(prompt, **kwargs):
        prompts.append((prompt, kwargs))
        return "这是足够长的结构化总结内容。" * 8

    monkeypatch.setattr(client, "generate", fake_generate)
    result = client.summarize(
        "第一部分内容较长。\n\n第二部分内容也很长。\n\n第三部分继续讨论。",
        title="测试视频",
        author="作者",
    )

    assert len(prompts) >= 3
    assert "第 1/" in prompts[0][0]
    assert "第 1 部分提炼" in prompts[-1][0]
    assert prompts[-1][1]["num_ctx"] == 32768
    assert len(result) >= 80


def test_summarize_mode_uses_corrected_file_without_raw_transcript(tmp_path):
    corrected = tmp_path / "video.corrected.md"
    corrected.write_text("# title\n\n---\n\ncorrected body", encoding="utf-8")
    item = _item("BVsummary")

    class _State:
        def get_status(self, bvid):
            return "corrected"

        def get_video_state(self, bvid):
            return SimpleNamespace(
                transcript_md=None,
                corrected_md=str(corrected),
                language="zh",
            )

        def update(self, *args, **kwargs):
            pass

    unload_calls = []
    fake_pipeline = SimpleNamespace(
        config={"pipeline": {"max_items_per_run": 10}},
        state=_State(),
        logger=logging.getLogger("test.summary-mode"),
        root=tmp_path,
        queue_file=tmp_path / "queue.json",
        ollama=SimpleNamespace(unload_model=lambda: unload_calls.append(True)),
    )
    fake_pipeline.queue_file.write_text(
        json.dumps({"queue": [item.to_dict()]}), encoding="utf-8"
    )
    step = StepLLM(fake_pipeline)
    step._summarize_item = lambda item, state: tmp_path / "video.final.md"

    stats, processed = step.run(mode="summarize", unload_after=False)

    assert stats["summarized"] == 1
    assert processed == ["BVsummary"]
    assert unload_calls == []


class _FakeLocator:
    def __init__(self, text):
        self.text = text

    def inner_text(self, timeout=0):
        return self.text


class _FakePage:
    def __init__(self, text):
        self.text = text
        self.visited = []

    def goto(self, url, **kwargs):
        self.visited.append(url)

    def locator(self, selector):
        return _FakeLocator(self.text)


def test_upload_verification_requires_title_on_shelf():
    client = WeReadBrowserClient.__new__(WeReadBrowserClient)
    page = _FakePage("书架里有 测试视频标题")
    assert client._verify_uploaded_on_shelf(
        page, r"C:\books\测试视频标题.epub", timeout_seconds=0
    ) is False

    page = _FakePage("书架里有 测试视频标题")
    assert client._verify_uploaded_on_shelf(
        page, r"C:\books\测试视频标题.epub", timeout_seconds=1
    ) is True


def test_upload_completion_detects_weread_done_text_with_percent_title():
    text = "觉得有好点子就想创业？这三关淘汰99%的人.epub 导入完成 · 立即阅读"

    assert WeReadBrowserClient._upload_completion_text_detected(text) is True
