from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import captions, media, reframe
from .config import Config
from .media import MediaInfo
from .select import Candidate

AUDIO_FILTER = "loudnorm=I=-14:TP=-1.5:LRA=11"  # YouTube's reference loudness


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()[:n] or "clip"


def render_clip(src: Path, info: MediaInfo, cand: Candidate, words: list[dict],
                layout: reframe.Layout, out: Path, cfg: Config) -> Path:
    """Render one clip. Runs ffmpeg with cwd=out.parent so the ass filename needs no escaping."""
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = reframe.video_graph(layout, info)
    use_caps = cfg.captions and bool(words) and media.has_filter("ass")
    if use_caps:
        rel = [{**w, "start": w["start"] - cand.start, "end": w["end"] - cand.start} for w in words]
        ass = out.with_suffix(".ass")
        ass.write_text(captions.build_ass(rel, reframe.caption_y(layout)), encoding="utf-8")
        graph += f";[v0]ass={ass.name}[v1];[v1]format=yuv420p[vout]"
    else:
        graph += ";[v0]format=yuv420p[vout]"

    cmd = ["ffmpeg", "-y", "-v", "error", "-stats",
           "-ss", f"{cand.start:.3f}", "-t", f"{cand.duration:.3f}", "-i", str(src.resolve()),
           "-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0?",
           *media.video_encoder_args(cfg.encoder),
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-af", AUDIO_FILTER,
           "-r", "30", "-movflags", "+faststart", out.name]
    subprocess.run(cmd, check=True, cwd=out.parent)
    thumb = out.with_suffix(".jpg")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", out.name, "-frames:v", "1",
                    "-q:v", "3", thumb.name], check=True, cwd=out.parent)
    return out
