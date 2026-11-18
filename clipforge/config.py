from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

console = Console(stderr=True)


def log(msg: str, style: str | None = None) -> None:
    console.print(msg, markup=False, highlight=False, style=style)


class ConfigError(Exception):
    pass


LAYOUTS = ("split", "center", "blur_fill")
RANKERS = ("claude_cli", "gemini")
ENCODERS = ("auto", "nvenc", "x264")


def norm_channel(s: str) -> str:
    """Normalise a handle, channel id, name or channel URL for comparison."""
    s = (s or "").strip().rstrip("/")
    if "youtube.com" in s or "youtu.be" in s:
        s = s.split("/")[-1]
    return s.lstrip("@").lower()


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Config:
    allowed_channels: list[str] = field(default_factory=list)
    streamers: dict[str, dict] = field(default_factory=dict)
    layout: str = "split"
    ranker: str = "claude_cli"
    min_score: float = 7.0
    max_clips: int = 10
    signals: dict[str, float] = field(
        default_factory=lambda: {"chat": 0.45, "audio": 0.35, "speech": 0.15, "scene": 0.05})
    chat_lag_seconds: float = 8.0
    chat_window_s: float = 10.0
    hype_bonus: float = 3.0
    hype_tokens: list[str] = field(default_factory=lambda: [
        "lmao", "lol", "lmfao", "w", "omegalul", "kekw", "pog", "poggers", "clip it", "clip that",
        "😂", "🤣", "💀", "🔥", "no way", "insane"])
    candidates: int = 40
    window_s: float = 90.0
    peak_min_gap_s: float = 60.0
    rank_batch_size: int = 5
    rank_retries: int = 3
    claude_model: str | None = None
    claude_timeout: int = 300
    gemini_model: str = "gemini-2.5-flash"
    whisper_model: str = "auto"
    language: str | None = "en"
    encoder: str = "auto"
    encode: dict[str, Any] = field(default_factory=lambda: {"crf": 18, "preset": "slow", "nvenc_cq": 19})
    profanity: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "words": [], "mask": "*"})
    watch: dict[str, Any] = field(default_factory=lambda: {"interval_minutes": 30, "backfill": 1})
    max_height: int = 1080
    cookies_from_browser: str | None = None
    work_dir: str = "work"
    output_dir: str = "output"
    # quality gate (section 8)
    gate_min_s: float = 15.0
    gate_max_s: float = 59.0
    gate_min_wps: float = 1.5
    gate_wps_exempt_score: float = 9.0

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        defaults = cls()
        known = {f.name for f in fields(cls)}
        kw: dict[str, Any] = {}
        for k, v in (raw or {}).items():
            if k not in known:
                log(f"config: ignoring unknown key '{k}'", "yellow")
                continue
            cur = getattr(defaults, k)
            kw[k] = {**cur, **(v or {})} if isinstance(cur, dict) and k != "streamers" else v
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p} (create one from the example in the README)")
        return cls.from_dict(yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    def validate(self) -> None:
        if self.ranker not in RANKERS:
            raise ConfigError(f"ranker must be one of {RANKERS}")
        if self.encoder not in ENCODERS:
            raise ConfigError(f"encoder must be one of {ENCODERS}")
        if self.layout not in LAYOUTS:
            raise ConfigError(f"layout must be one of {LAYOUTS}")
        if any(v < 0 for v in self.signals.values()) or not any(v > 0 for v in self.signals.values()):
            raise ConfigError("signals weights must be >= 0 with at least one > 0")
        for name, s in self.streamers.items():
            if s.get("layout", self.layout) not in LAYOUTS:
                raise ConfigError(f"streamer {name}: layout must be one of {LAYOUTS}")
        self.allowed_channels = [str(c) for c in self.allowed_channels]

    # --- helpers -------------------------------------------------------------------------
    def streamer(self, handle: str | None) -> dict:
        """Streamer settings merged over global defaults; handle matched leniently."""
        base = {"layout": self.layout}
        if handle:
            for k, v in self.streamers.items():
                if norm_channel(k) == norm_channel(handle):
                    return {**base, **(v or {}), "_key": k}
        return {**base, "_key": handle}

    @property
    def work(self) -> Path:
        return Path(self.work_dir)

    @property
    def output(self) -> Path:
        return Path(self.output_dir)
