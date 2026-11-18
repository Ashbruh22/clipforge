from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="clipforge", description="Stream/VOD -> YouTube Shorts")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check ffmpeg, yt-dlp, GPU and Python dependencies")

    r = sub.add_parser("run", help="process a URL or local video file")
    r.add_argument("source", help="YouTube stream/VOD URL or a local video path")
    r.add_argument("-o", "--out", default="clipforge_out")
    r.add_argument("-n", "--clips", type=int, default=8)
    r.add_argument("--min-dur", type=float, default=20)
    r.add_argument("--max-dur", type=float, default=55)
    r.add_argument("--whisper-model", default="auto", help="auto, large-v3, medium, small ...")
    r.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    r.add_argument("--language", default=None, help="e.g. en; default auto-detect")
    r.add_argument("--layout", choices=["auto", "face", "split", "fit"], default="auto")
    r.add_argument("--no-captions", action="store_true")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--llm-model", default="claude-sonnet-5-5")
    r.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto")
    r.add_argument("--max-height", type=int, default=1080)
    r.add_argument("--cookies-from-browser", default=None, metavar="BROWSER")
    r.add_argument("--force", action="store_true", help="ignore cached transcript/audio")

    v = sub.add_parser("review", help="rebuild review.html for a finished run directory")
    v.add_argument("workdir")

    a = p.parse_args(argv)
    if a.cmd == "doctor":
        from .env import print_env

        sys.exit(0 if print_env() else 1)
    if a.cmd == "review":
        from .review import write_review

        wd = Path(a.workdir)
        print(write_review(wd, wd.name, json.loads((wd / "clips.json").read_text())))
        return

    cfg = Config(out_dir=Path(a.out), clips=a.clips, min_dur=a.min_dur, max_dur=a.max_dur,
                 max_height=a.max_height, whisper_model=a.whisper_model, device=a.device,
                 language=a.language, layout=a.layout, captions=not a.no_captions,
                 use_llm=not a.no_llm, llm_model=a.llm_model, encoder=a.encoder,
                 cookies_from_browser=a.cookies_from_browser, force=a.force)
    from .pipeline import run

    run(a.source, cfg)
