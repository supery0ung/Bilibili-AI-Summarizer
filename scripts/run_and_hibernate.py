"""Scheduled entry point: run pipeline, then hibernate if no user activity."""
from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Env vars required by main.py (must be set before pipeline import)
MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "BILIBILI_MODEL_CACHE",
        Path.home() / ".cache" / "bilibili_summarizer" / "ai_models",
    )
)
TEMP_ROOT = Path(
    os.environ.get(
        "BILIBILI_TEMP",
        Path.home() / ".cache" / "bilibili_summarizer" / "temp",
    )
)
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_CACHE_ROOT))
os.environ.setdefault("WHISPER_CACHE", str(MODEL_CACHE_ROOT / "whisper"))
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_ROOT / "huggingface" / "hub"))
os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

# UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"scheduled_{datetime.now():%Y-%m-%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Hibernate if no keyboard/mouse activity for this long after pipeline finishes
IDLE_THRESHOLD_MINUTES = 10

# Give the scheduled-task wrapper enough time to record the exit code and exit
# before a detached helper hibernates the machine. Otherwise shutdown /h blocks
# until the next wake and Task Scheduler's IgnoreNew policy skips that run.
HIBERNATE_DELAY_SECONDS = int(os.environ.get("HIBERNATE_DELAY_SECONDS", "15"))

# Per-attempt hard cap: a backstop to guarantee the machine eventually hibernates.
# The real guard against hangs is the heartbeat watchdog below. The ACTUAL root
# cause of the hours-long morning stalls was a DUPLICATE scheduled task (see
# memory: asr-hang-rootcause): two pipelines ran at 10:00 and fought over the
# 12 GB GPU. A 100-item run with long multi-speaker videos can legitimately
# exceed four hours, so leave the heartbeat watchdog to detect real stalls and
# keep this as a generous final backstop.
PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", 12 * 3600))

# Heartbeat watchdog. The pipeline writes continuously to the log file; if it
# goes silent for this long while still running, we treat it as a stall and kill
# it. 20 min is comfortably longer than any legitimate quiet step (a single
# Ollama correction/summary or an upload) so it won't false-trigger, yet it
# catches a genuine GPU stall fast (vs. blocking for the full hard cap).
STALL_SECONDS = int(os.environ.get("PIPELINE_STALL_SECONDS", 20 * 60))
# How often the watchdog checks the heartbeat.
HEARTBEAT_POLL_SECONDS = 30
# Total attempts. With the duplicate task removed, attempt 1 should just work.
# The single fallback retry runs in a fresh process with smaller ASR chunks —
# harmless defence-in-depth for a one-off transient stall.
MAX_ATTEMPTS = 2
# Per-attempt ASR sizing overrides (read by clients/qwen_asr_client.py via env).
RETRY_ASR_ENV = [
    {},  # attempt 1: normal (180s chunks)
    {"ASR_MAX_CHUNK_SECONDS": "90", "ASR_BATCH_MAX_TOTAL_SECONDS": "90", "ASR_BATCH_SIZE": "1"},
]


def get_idle_seconds() -> float:
    """Return seconds since last keyboard or mouse event (Windows only)."""

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(lii)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    # Both values are 32-bit tick counts; mask handles wrap-around
    now_tick = ctypes.windll.kernel32.GetTickCount()
    elapsed_ms = (now_tick - lii.dwTime) & 0xFFFF_FFFF
    return elapsed_ms / 1000.0


LOCK_FILE = PROJECT_ROOT / "output" / "pipeline.lock"


def _is_pid_running(pid: int) -> bool:
    PROCESS_QUERY_INFORMATION = 0x0400
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            if _is_pid_running(pid):
                logger.warning(f"Pipeline already running (PID {pid}), skipping scheduled run")
                return False
        except Exception:
            pass  # stale lock, overwrite
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    LOCK_FILE.unlink(missing_ok=True)


# Grace period to confirm a killed process actually died. A process stalled
# INSIDE an uninterruptible GPU driver call resists `taskkill /F` — proven on
# 2026-06-08 when the 10:00 run's PID was still alive 7h after the 14:00 kill.
# If it won't die in this window, a retry can't acquire the GPU, so we must NOT
# spawn a competing process.
KILL_VERIFY_SECONDS = 120


def _kill_tree_verify(proc: subprocess.Popen) -> bool:
    """taskkill the process tree and confirm the direct child actually died.

    Returns True if the child is gone, False if it survived the kill (stuck in
    an uninterruptible GPU driver call — only a GPU/driver reset, i.e. a
    hibernate/wake cycle or reboot, can clear that).
    """
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
        check=False,
        capture_output=True,
    )
    try:
        proc.wait(timeout=KILL_VERIFY_SECONDS)
        return True
    except subprocess.TimeoutExpired:
        return False


def _run_pipeline_attempt(asr_env: dict[str, str]) -> tuple[int, str]:
    """Run one `main.py run` and watch its heartbeat.

    Returns (returncode, status) where status is:
      - "finished":   process exited on its own (success or normal failure)
      - "stalled":    went silent / hit the hard cap and was killed cleanly (retry is safe)
      - "unkillable": stalled AND survived taskkill (GPU-stuck; a retry would fight it)
    """
    env = os.environ.copy()
    env.update(asr_env)

    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "main.py"), "run"],
            cwd=PROJECT_ROOT,
            stdout=lf,
            stderr=lf,
            env=env,
        )
        start = time.time()
        while True:
            try:
                return proc.wait(timeout=HEARTBEAT_POLL_SECONDS), "finished"
            except subprocess.TimeoutExpired:
                pass  # still running — check the heartbeat below

            now = time.time()
            try:
                last_activity = log_file.stat().st_mtime
            except OSError:
                last_activity = start
            silent_for = now - last_activity

            reason = None
            if silent_for >= STALL_SECONDS:
                reason = (
                    f"No log output for {silent_for:.0f}s (>= {STALL_SECONDS}s) — "
                    f"pipeline stalled (likely degraded-GPU ASR forward)"
                )
            elif now - start >= PIPELINE_TIMEOUT_SECONDS:
                reason = f"Pipeline exceeded hard cap {PIPELINE_TIMEOUT_SECONDS}s"

            if reason:
                logger.error(f"{reason} → killing process tree")
                if _kill_tree_verify(proc):
                    return -1, "stalled"
                logger.error(
                    "Killed process did NOT die after taskkill — it is stuck in an "
                    "uninterruptible GPU driver call. A retry cannot acquire the GPU, "
                    "so aborting retries; the GPU needs a reset (the hibernate/wake "
                    "below, or a reboot)."
                )
                return -1, "unkillable"


def run_pipeline() -> int:
    """Run the pipeline, retrying in a fresh process if it stalls.

    A stall is almost always the post-resume degraded-GPU ASR hang. Killing the
    process frees the CUDA context and the retry starts clean (and with smaller
    ASR chunks), which gets the scheduled run to actually finish instead of
    hanging for hours. If the stalled process can't be killed at all (stuck in
    the GPU driver), retrying is pointless — we stop and let the hibernate/wake
    cycle reset the GPU.
    """
    logger.info("=== Scheduled pipeline run starting ===")
    returncode = -1
    for attempt in range(1, MAX_ATTEMPTS + 1):
        asr_env = RETRY_ASR_ENV[min(attempt - 1, len(RETRY_ASR_ENV) - 1)]
        if attempt > 1:
            logger.info(
                f"--- Retry attempt {attempt}/{MAX_ATTEMPTS} "
                f"(fresh process; ASR overrides: {asr_env or 'defaults'}) ---"
            )
        returncode, status = _run_pipeline_attempt(asr_env)
        if status == "finished":
            logger.info(f"Pipeline finished, exit code={returncode}")
            return returncode
        if status == "unkillable":
            logger.error("Stalled process is unkillable — aborting retries, proceeding to hibernate.")
            return returncode
        logger.warning(f"Attempt {attempt}/{MAX_ATTEMPTS} stalled and was killed cleanly.")

    logger.error(f"All {MAX_ATTEMPTS} attempts stalled — giving up, will hibernate.")
    return returncode


def _schedule_hibernate() -> None:
    """Launch hibernation out of process so this scheduled task can exit first."""
    command = (
        f"Start-Sleep -Seconds {HIBERNATE_DELAY_SECONDS}; "
        "shutdown.exe /h"
    )
    creationflags = (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-Command",
            command,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )


def maybe_hibernate() -> None:
    idle = get_idle_seconds()
    threshold = IDLE_THRESHOLD_MINUTES * 60
    logger.info(f"Idle time: {idle:.0f}s | threshold: {threshold}s ({IDLE_THRESHOLD_MINUTES} min)")
    if idle >= threshold:
        logger.info(
            "No user activity detected → scheduling hibernate in "
            f"{HIBERNATE_DELAY_SECONDS}s"
        )
        _schedule_hibernate()
    else:
        logger.info(f"User was active {idle:.0f}s ago → skipping hibernate")


if __name__ == "__main__":
    if not acquire_lock():
        sys.exit(0)
    exit_code = 1
    try:
        exit_code = run_pipeline()
    except Exception:
        # Never let a crash in the pipeline path skip hibernation.
        logger.exception("Pipeline run raised an exception")
    finally:
        release_lock()
        # maybe_hibernate is in the finally so the machine sleeps even if the
        # pipeline crashed or was killed by the timeout above.
        maybe_hibernate()
    sys.exit(exit_code)
