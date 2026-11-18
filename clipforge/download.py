from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, log


@dataclass
class Source:
    id: str
    title: str
    path: Path
    workdir: Path
    chat_path: Path | None = None


def _ytdlp(*args: str) -> list[str]:
    return [sys.executable, "-m", "yt_dlp", *args]


def _slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()[:n] or "video"


def acquire(src: str, cfg: Config) -> Source:
    """Local file or URL -> Source in <out>/<id>/. Re-uses an existing download."""
    p = Path(src).expanduser()
    if p.exists():
        wd = cfg.out_dir / _slug(p.stem)
        wd.mkdir(parents=True, exist_ok=True)
        return Source(_slug(p.stem), p.stem, p.resolve(), wd)

    cookies = ["--cookies-from-browser", cfg.cookies_from_browser] if cfg.cookies_from_browser else []
    meta = subprocess.run(
        _ytdlp("--no-playlist", "--skip-download", "--print", "%(id)s\t%(title)s", *cookies, src),
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[-1]
    vid, _, title = meta.partition("\t")
    wd = cfg.out_dir / vid
    wd.mkdir(parents=True, exist_ok=True)

    existing = [f for f in wd.glob("source.*") if f.suffix in (".mp4", ".mkv", ".webm")]
    if not existing:
        h = cfg.max_height
        fmt = (f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]")
        log(f"downloading {vid} ({title})")
        subprocess.run(
            _ytdlp("--no-playlist", "-f", fmt, "--merge-output-format", "mp4",
                   "--concurrent-fragments", "4", "--retries", "10",
                   "-o", str(wd / "source.%(ext)s"), *cookies, src),
            check=True,
        )
        existing = [f for f in wd.glob("source.*") if f.suffix in (".mp4", ".mkv", ".webm")]
    if not existing:
        raise RuntimeError("yt-dlp produced no video file")

    chat = next(iter(wd.glob("source*.live_chat.json")), None)
    if chat is None:  # chat replay is best-effort: many VODs have none
        subprocess.run(
            _ytdlp("--no-playlist", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
                   "-o", str(wd / "source.%(ext)s"), *cookies, src),
            capture_output=True,
        )
        chat = next(iter(wd.glob("source*.live_chat.json")), None)
    return Source(vid, title, existing[0], wd, chat)
