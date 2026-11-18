from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import captions, layout, media
from .config import Config, log
from .cut import Clip
from .media import MediaInfo

TARGET_I, TARGET_TP, TARGET_LRA = -14.0, -1.5, 11.0


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()[:n] or "clip"


def encoder_args(kind: str, cfg: Config) -> list[str]:
    e = cfg.encode
    if kind == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
                "-cq", str(e.get("nvenc_cq", 19)), "-b:v", "0", "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", str(e.get("preset", "slow")), "-crf", str(e.get("crf", 18)),
            "-profile:v", "high"]


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {r.stderr[-1200:]}")
    return r


def measure_loudness(src: Path, start: float, dur: float) -> dict | None:
    """loudnorm pass 1 on exactly the clip's audio."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
         "-vn", "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return None
    d = json.loads(m.group(0))
    try:
        if float(d["input_i"]) < -70:  # silence: nothing to normalise
            return None
        return {k: float(d[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (KeyError, ValueError):
        return None


def audio_filter(m: dict | None) -> str:
    base = f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
    if m:
        base += (f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
                 f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    return base + ",aresample=48000"


def render_clip(src: Path, info: MediaInfo, clip: Clip, plan: layout.Plan, out: Path, build: Path,
                cfg: Config, encoder: str) -> dict:
    """Frame-accurate re-encode of one clip to 1080x1920. Returns render stats."""
    out.parent.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    fps = 60 if info.fps >= 50 else 30  # "30 or 60 fps matching the source"
    graph = layout.video_graph(plan, info, fps)

    family = captions.ensure_font(build / "fonts")
    rel = [{**w, "start": w["start"] - clip.start, "end": w["end"] - clip.start} for w in clip.words]
    ass_name = f"{out.stem}.ass"
    (build / ass_name).write_text(
        captions.build_ass(rel, clip.r.hook_text, plan.caption_y, family, cfg.profanity), encoding="utf-8")
    if media.has_filter("ass"):  # relative names + cwd=build: no path escaping on any OS
        graph += f";[v0]ass={ass_name}:fontsdir=fonts[v1];[v1]format=yuv420p[vout]"
    else:
        log("  ffmpeg has no libass: captions/hook not burned in", "yellow")
        graph += ";[v0]format=yuv420p[vout]"

    loud = measure_loudness(src, clip.start, clip.duration)
    base = ["ffmpeg", "-y", "-v", "error"]
    inp = ["-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", str(src.resolve())]
    tail = ["-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0?", *encoder_args(encoder, cfg),
            "-c:a", "aac", "-b:a", "192k", "-af", audio_filter(loud),
            "-r", str(fps), "-movflags", "+faststart", str(out.resolve())]
    try:
        # NVENC runs with GPU decoding too: frames come back to system memory for the CPU filters
        _run(base + (["-hwaccel", "cuda"] if encoder == "nvenc" else []) + inp + tail, cwd=build)
    except RuntimeError:
        if encoder != "nvenc":
            raise
        log("  GPU decode failed; retrying with CPU decode + NVENC", "yellow")
        _run(base + inp + tail, cwd=build)
    thumb = out.with_suffix(".jpg")
    _run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(out.resolve()), "-frames:v", "1",
          "-q:v", "3", str(thumb.resolve())])
    return {"loudness_in": loud["input_i"] if loud else None, "fps": fps}
