from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys

from . import media


def _ver(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        return out.splitlines()[0] if out else "ok"
    except Exception:
        return None


def check_env() -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    rows.append(("OS", True, f"{platform.system()} {platform.release()}"))
    rows.append(("Python", sys.version_info >= (3, 10), platform.python_version()))
    for t in ("ffmpeg", "ffprobe"):
        v = _ver([t, "-version"]) if shutil.which(t) else None
        rows.append((t, v is not None, v or "MISSING (install ffmpeg)"))
    yt = _ver([sys.executable, "-m", "yt_dlp", "--version"])
    rows.append(("yt-dlp", yt is not None, yt or "MISSING (pip install yt-dlp)"))
    if shutil.which("ffmpeg"):
        rows.append(("ffmpeg libass (captions)", media.has_filter("ass"), ""))
        n = media.nvenc_works()
        rows.append(("NVENC encoder", n, "GPU encode" if n else "will use libx264 (CPU)"))
    smi = _ver(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) if shutil.which("nvidia-smi") else None
    rows.append(("NVIDIA GPU", smi is not None, smi or "not detected"))
    cuda = media.cuda_available()
    rows.append(("CUDA for Whisper", cuda, "yes" if cuda else "no (CPU fallback, slower)"))
    for mod, pipname in (("faster_whisper", "faster-whisper"), ("cv2", "opencv-python-headless"),
                         ("numpy", "numpy")):
        ok = importlib.util.find_spec(mod) is not None
        rows.append((pipname, ok, "" if ok else f"MISSING (pip install {pipname})"))
    has_llm = importlib.util.find_spec("anthropic") is not None
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    rows.append(("LLM rerank (optional)", has_llm and key,
                 "enabled" if has_llm and key else "off: needs `pip install anthropic` + ANTHROPIC_API_KEY"))
    return rows


def print_env() -> bool:
    rows = check_env()
    critical = {"ffmpeg", "ffprobe", "yt-dlp", "faster-whisper", "opencv-python-headless", "numpy"}
    for name, ok, detail in rows:
        print(f"[{'ok' if ok else '!!'}] {name:26s} {detail}")
    return all(ok for name, ok, _ in rows if name in critical)
