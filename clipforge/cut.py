from __future__ import annotations

import gc
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import media
from .config import Config, log
from .rank import Ranked

PAD = 2.0  # seconds of audio transcribed either side of the ranked range
BREATH = 0.25
TERMINAL = re.compile(r"[.?!]['\")\]]*$")


# ---------------------------------------------------------------- whisper -----------------
def pick_model(cfg: Config) -> tuple[str, str, str]:
    """(model, device, compute_type). Largest model that fits the GPU's VRAM; CPU int8 small otherwise."""
    cuda = media.cuda_available()
    if not cuda:
        return (cfg.whisper_model if cfg.whisper_model != "auto" else "small"), "cpu", "int8"
    if cfg.whisper_model != "auto":
        return cfg.whisper_model, "cuda", "float16"
    vram = media.vram_mib()
    if vram is None or vram >= 6000:
        name = "large-v3"
    elif vram >= 4000:
        name = "medium"
    else:
        name = "small"
    return name, "cuda", "float16"


def load_audio(video: Path, start: float, end: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{end - start:.3f}",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class WhisperRunner:
    """Transcribes only the requested ranges (never the whole VOD), with a per-range cache.
    The model loads lazily and is released by close() before rendering starts, so only one
    GPU job runs at a time."""

    def __init__(self, video: Path, workdir: Path, cfg: Config, duration: float):
        self.video, self.cfg, self.duration = video, cfg, duration
        self.cache_dir = workdir / "words"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            name, device, ct = pick_model(self.cfg)
            log(f"whisper: {name} on {device} ({ct})")
            self._model = WhisperModel(name, device=device, compute_type=ct)
        return self._model

    def transcribe(self, start: float, end: float) -> list[dict]:
        start, end = max(0.0, start), min(self.duration, end)
        f = self.cache_dir / f"{start:.1f}_{end:.1f}.json"
        if f.exists():
            return json.loads(f.read_text())
        audio = load_audio(self.video, start, end)
        segs, _ = self._load().transcribe(
            audio, language=self.cfg.language, word_timestamps=True, beam_size=5,
            condition_on_previous_text=False, vad_filter=False)
        words = [{"start": round(start + w.start, 3), "end": round(start + w.end, 3), "word": w.word.strip()}
                 for s in segs for w in (s.words or []) if w.word.strip()]
        f.write_text(json.dumps(words))
        return words

    def close(self) -> None:
        self._model = None
        gc.collect()


# ---------------------------------------------------------------- snapping ----------------
def _sentence_starts(ws: list[dict]) -> list[int]:
    return [i for i in range(len(ws)) if i == 0 or TERMINAL.search(ws[i - 1]["word"])
            or ws[i]["start"] - ws[i - 1]["end"] > 0.7]


def _sentence_ends(ws: list[dict]) -> list[int]:
    return [i for i in range(len(ws)) if i == len(ws) - 1 or TERMINAL.search(ws[i]["word"])
            or ws[i + 1]["start"] - ws[i]["end"] > 0.7]


def snap(ws: list[dict], req_start: float, req_end: float, tol: float = 3.0) -> tuple[float, float] | None:
    """Move the requested cut to sentence (else word) boundaries and add 0.25 s of breathing room,
    never reaching into a neighbouring word. Returns None when there are no words."""
    if not ws:
        return None
    ss, se = _sentence_starts(ws), _sentence_ends(ws)
    near_s = [i for i in ss if abs(ws[i]["start"] - req_start) <= tol]
    i0 = min(near_s, key=lambda i: abs(ws[i]["start"] - req_start)) if near_s else \
        min(range(len(ws)), key=lambda i: abs(ws[i]["start"] - req_start))
    near_e = [i for i in se if abs(ws[i]["end"] - req_end) <= tol and i >= i0]
    i1 = min(near_e, key=lambda i: abs(ws[i]["end"] - req_end)) if near_e else \
        min((i for i in range(len(ws)) if i >= i0), key=lambda i: abs(ws[i]["end"] - req_end))
    a = ws[i0]["start"] - BREATH
    if i0 > 0:
        a = max(a, (ws[i0 - 1]["end"] + ws[i0]["start"]) / 2)
    b = ws[i1]["end"] + BREATH
    if i1 < len(ws) - 1:
        b = min(b, (ws[i1]["end"] + ws[i1 + 1]["start"]) / 2)
    return max(0.0, a), b


def fit_length(ws: list[dict], a: float, b: float, lo: float, hi: float) -> tuple[float, float]:
    """If snapping left the clip outside [lo, hi], move the end to another sentence end."""
    se = [ws[i]["end"] + BREATH for i in _sentence_ends(ws)]
    if b - a > hi:
        ok = [e for e in se if lo <= e - a <= hi]
        if ok:
            b = max(ok)
    elif b - a < lo:
        ok = [e for e in se if lo <= e - a <= hi]
        if ok:
            b = min(ok)
    return a, b


# ---------------------------------------------------------------- clips + gate ------------
@dataclass
class Clip:
    r: Ranked
    start: float
    end: float
    words: list[dict]  # absolute times
    wps: float = 0.0
    index: int = 0
    drop: str | None = None
    layout: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_clip(r: Ranked, runner: WhisperRunner, cfg: Config) -> Clip:
    words = runner.transcribe(r.start - PAD, r.end + PAD)
    snapped = snap(words, r.start, r.end)
    if snapped is None:  # no speech found: keep Claude's times; the gate decides
        return Clip(r, r.start, r.end, [])
    a, b = fit_length(words, *snapped, cfg.gate_min_s + 5, cfg.gate_max_s - 4)
    inside = [w for w in words if w["start"] >= a - 0.01 and w["end"] <= b + 0.01]
    d = max(0.01, b - a)
    return Clip(r, a, b, inside, wps=len(inside) / d)


def quality_gate(clips: list[Clip], cfg: Config) -> tuple[list[Clip], list[Clip]]:
    """Section 8: drop clips outside 15-59 s, too little speech (unless score >= 9), or overlapping
    a clip already kept from the same VOD (higher score wins)."""
    kept: list[Clip] = []
    dropped: list[Clip] = []
    for c in sorted(clips, key=lambda c: c.r.score, reverse=True):
        if not (cfg.gate_min_s <= c.duration <= cfg.gate_max_s):
            c.drop = f"length {c.duration:.1f}s outside {cfg.gate_min_s:.0f}-{cfg.gate_max_s:.0f}s"
        elif c.wps < cfg.gate_min_wps and c.r.score < cfg.gate_wps_exempt_score:
            c.drop = f"only {c.wps:.2f} words/s (< {cfg.gate_min_wps})"
        elif any(c.start < k.end and k.start < c.end for k in kept):
            c.drop = "overlaps a higher-scoring clip from the same VOD"
        if c.drop:
            dropped.append(c)
        else:
            kept.append(c)
    kept.sort(key=lambda c: c.r.score, reverse=True)
    for i, c in enumerate(kept, 1):
        c.index = i
    return kept, dropped
