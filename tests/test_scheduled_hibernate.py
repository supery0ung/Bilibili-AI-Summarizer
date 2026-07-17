import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_runner(monkeypatch):
    monkeypatch.setattr(sys, "platform", "test")
    path = Path(__file__).parents[1] / "scripts" / "run_and_hibernate.py"
    spec = importlib.util.spec_from_file_location("scheduled_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hibernate_is_scheduled_in_detached_process(monkeypatch):
    runner = _load_runner(monkeypatch)
    calls = []

    monkeypatch.setattr(runner, "get_idle_seconds", lambda: 601)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    runner.maybe_hibernate()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][0] == "powershell.exe"
    assert "Start-Sleep -Seconds 15; shutdown.exe /h" in args[0]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["close_fds"] is True


def test_active_user_skips_hibernate(monkeypatch):
    runner = _load_runner(monkeypatch)

    monkeypatch.setattr(runner, "get_idle_seconds", lambda: 30)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected hibernate")),
    )

    runner.maybe_hibernate()
