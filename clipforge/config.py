from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Config:
    out_dir: Path = Path("clipforge_out")
    clips: int = 8
    min_dur: float = 20.0
    max_dur: float = 55.0
    max_height: int = 1080
    whisper_model: str = "auto"  # auto: large-v3 on CUDA, small on CPU
    device: str = "auto"  # auto | cuda | cpu
    language: str | None = None
    layout: str = "auto"  # auto | face | split | fit
    captions: bool = True
    use_llm: bool = True
    llm_model: str = "claude-sonnet-5-5"
    encoder: str = "auto"  # auto | nvenc | x264
    cookies_from_browser: str | None = None
    force: bool = False
    min_gap: float = 15.0  # min seconds between selected clips
