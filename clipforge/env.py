from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys

from . import media


def _first_line(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        return out.splitlines()[0] if out.strip() else "ok"
    except Exception:
        return None


def check_env() -> list[tuple[str, bool, str, bool]]:
    """(name, ok, detail, critical)"""
    rows: list[tuple[str, bool, str, bool]] = []

    def add(name, ok, detail="", critical=False):
        rows.append((name, bool(ok), detail, critical))

    add("OS", True, f"{platform.system()} {platform.release()}")
    add("Python", sys.version_info >= (3, 11), platform.python_version(), True)
    for t in ("ffmpeg", "ffprobe"):
        v = _first_line([t, "-version"]) if shutil.which(t) else None
        add(t, v, v or "MISSING (install ffmpeg)", True)
    yt = _first_line([sys.executable, "-m", "yt_dlp", "--version"])
    add("yt-dlp", yt, yt or "MISSING (pip install yt-dlp)", True)
    if shutil.which("ffmpeg"):
        add("ffmpeg libass (captions)", media.has_filter("ass"), "" if media.has_filter("ass") else "captions disabled")
        n = media.nvenc_works()
        add("NVENC encoder", n, "GPU encode available" if n else "not usable: will use libx264 (CPU)")
    vram = media.vram_mib()
    add("NVIDIA GPU", vram is not None, f"{vram} MiB VRAM" if vram else "not detected")
    cuda = media.cuda_available()
    add("CUDA for Whisper", cuda, "yes" if cuda else "no: CPU int8 'small' fallback")
    claude = shutil.which("claude")
    add("claude CLI (ranker)", claude, claude or "not found: install Claude Code or use ranker: gemini", False)
    add("GEMINI_API_KEY (fallback ranker)", bool(os.environ.get("GEMINI_API_KEY")), "set" if os.environ.get("GEMINI_API_KEY") else "not set (optional)")
    if os.environ.get("ANTHROPIC_API_KEY"):
        add("ANTHROPIC_API_KEY", False, "is set: ClipForge ignores it and strips it from the claude subprocess")
    for mod, pip, crit in (("faster_whisper", "faster-whisper", True), ("scenedetect", "scenedetect", False),
                           ("cv2", "opencv-python-headless", True), ("mediapipe", "mediapipe", False),
                           ("numpy", "numpy", True), ("yaml", "pyyaml", True), ("typer", "typer", True),
                           ("rich", "rich", True), ("jinja2", "jinja2", True)):
        ok = importlib.util.find_spec(mod) is not None
        add(pip, ok, "" if ok else f"MISSING (pip install {pip})" + ("" if crit else " [optional: Haar fallback]"), crit)
    return rows


def print_env() -> bool:
    rows = check_env()
    for name, ok, detail, _ in rows:
        print(f"[{'ok' if ok else '!!'}] {name:34s} {detail}")
    return all(ok for _, ok, _, crit in rows if crit)
