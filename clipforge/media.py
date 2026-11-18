from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class MediaInfo:
    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool


def probe(path: Path) -> MediaInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, _, den = v.get("avg_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    dur = float(d["format"].get("duration") or v.get("duration") or 0)
    return MediaInfo(int(v["width"]), int(v["height"]), dur, fps or 30.0,
                     any(s["codec_type"] == "audio" for s in d["streams"]))


@lru_cache(maxsize=None)
def has_filter(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return any(len(p) > 2 and p[1] == name for p in (line.split() for line in out.splitlines()))


@lru_cache(maxsize=None)
def nvenc_works() -> bool:
    """`ffmpeg -encoders` lists h264_nvenc AND a real test encode succeeds."""
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_nvenc" not in enc:
        return False
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
         "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    return r.returncode == 0


def resolve_encoder(pref: str) -> str:
    if pref == "x264":
        return "x264"
    if nvenc_works():
        return "nvenc"
    if pref == "nvenc":
        from .config import log
        log("encoder: nvenc requested but not usable here, falling back to x264", "yellow")
    return "x264"


def vram_mib() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        return max(int(x) for x in out.split())
    except Exception:
        return None


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
