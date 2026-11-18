from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config, log
from .ingest import Ingested

SR = 16000


# ---------------------------------------------------------------- raw signals --------------
def audio_db(video: Path, duration: float) -> np.ndarray:
    """RMS level in dB per second. ffmpeg pipes 16 kHz mono PCM; processed in chunks so a 3 h VOD
    stays at a few MB of memory and no temp WAV is written."""
    n_sec = int(duration) + 1
    out = np.full(n_sec, -90.0, dtype=np.float32)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        stdout=subprocess.PIPE)
    sec = 0
    carry = b""
    chunk = SR * 2 * 300
    while True:
        raw = p.stdout.read(chunk)
        if not raw:
            break
        raw = carry + raw
        whole = len(raw) // (SR * 2) * (SR * 2)
        carry = raw[whole:]
        if whole:
            x = np.frombuffer(raw[:whole], dtype=np.int16).astype(np.float32) / 32768.0
            x = x.reshape(-1, SR)
            rms = np.sqrt((x ** 2).mean(axis=1) + 1e-10)
            n = min(len(rms), n_sec - sec)
            out[sec: sec + n] = 20 * np.log10(rms[:n])
            sec += len(rms)
    p.wait()
    return out


def rolling_z(x: np.ndarray, win: int = 300) -> np.ndarray:
    """Robust z-score against a rolling baseline (median/MAD per half-window block, interpolated):
    a spike matters relative to the surrounding five minutes, not in absolute terms."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < 8:
        return np.zeros(n, dtype=np.float32)
    centers, med, mad = [], [], []
    for s in range(0, n, win // 2):
        blk = x[max(0, s - win // 4): s + win // 2 + win // 4]
        m = np.median(blk)
        centers.append(s + win // 4)
        med.append(m)
        mad.append(np.median(np.abs(blk - m)) * 1.4826 + 1e-3)
    idx = np.arange(n)
    return ((x - np.interp(idx, centers, med)) / np.interp(idx, centers, mad)).astype(np.float32)


def scene_cuts(video: Path, fps: float) -> np.ndarray:
    """Scene-cut times in seconds (PySceneDetect, sampled at ~3 fps: this is only a light signal)."""
    from scenedetect import ContentDetector, SceneManager, open_video

    v = open_video(str(video))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0))
    try:
        sm.downscale = 4
    except Exception:
        pass
    sm.detect_scenes(v, frame_skip=max(0, int(round(fps / 3)) - 1), show_progress=False)
    return np.array([s[0].get_seconds() for s in sm.get_scene_list()[1:]], dtype=np.float32)


def _tokens_match(text: str, hype: list[str]) -> bool:
    low = text.lower()
    words = set(re.findall(r"[\w']+", low))
    for h in hype:
        h = h.lower().strip()
        if not h:
            continue
        if re.fullmatch(r"[\w']+", h):  # single plain word: whole-token match ("w" must not hit "wow")
            if h in words:
                return True
        elif h in low:  # phrases, emoji
            return True
    return False


def chat_series(chat: list[tuple[float, str]], n_sec: int, cfg: Config) -> np.ndarray | None:
    """Chat velocity (msgs per window) + hype-token bonus, shifted back by chat_lag_seconds."""
    if len(chat) < 50:
        return None
    msgs = np.zeros(n_sec, dtype=np.float32)
    hype = np.zeros(n_sec, dtype=np.float32)
    for t, text in chat:
        i = int(t)
        if 0 <= i < n_sec:
            msgs[i] += 1
            if _tokens_match(text, cfg.hype_tokens):
                hype[i] += 1
    k = np.ones(max(1, int(cfg.chat_window_s)), dtype=np.float32)
    vel = np.convolve(msgs + cfg.hype_bonus * hype, k, mode="same")
    lag = int(round(cfg.chat_lag_seconds))
    # the spike shows up `lag` s after the moment, so the moment's score is the later chat value
    return np.concatenate([vel[lag:], np.zeros(lag, dtype=np.float32)]) if lag > 0 else vel


def speech_series(cues: list[dict], n_sec: int, win: int = 10) -> np.ndarray | None:
    """Words per second from the captions, smoothed."""
    if not cues:
        return None
    wps = np.zeros(n_sec, dtype=np.float32)
    for c in cues:
        a, b = int(c["start"]), max(int(c["start"]) + 1, int(np.ceil(c["end"])))
        words = len(c["text"].split())
        span = min(b, n_sec) - a
        if 0 <= a < n_sec and span > 0:
            wps[a: a + span] += words / (b - a)
    return np.convolve(wps, np.ones(win, dtype=np.float32) / win, mode="same")


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 50), np.percentile(x, 99.5)
    return np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- analysis ----------------
@dataclass
class Signals:
    n_sec: int
    series: dict[str, np.ndarray]  # normalised 0..1, per second, only the signals that exist
    combined: np.ndarray
    weights_used: dict[str, float] = field(default_factory=dict)


def analyse(ing: Ingested, duration: float, fps: float, cfg: Config, chat=None) -> Signals:
    from .ingest import parse_chat

    n_sec = int(duration) + 1
    st = ing.video.stat()
    key = f"{st.st_size}-{int(st.st_mtime)}-{n_sec}"
    cache = ing.workdir / "signals.npz"
    audio = scenes = None
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            audio = z["audio"]
            scenes = z["scenes"] if "scenes" in z.files else None
    if audio is None:
        log("analysing audio loudness")
        audio = rolling_z(audio_db(ing.video, duration))
    if scenes is None and cfg.signals.get("scene", 0) > 0:
        log("detecting scene changes")
        try:
            scenes = scene_cuts(ing.video, fps)
        except Exception as e:  # secondary signal: never fatal
            log(f"scene detection skipped: {e}", "yellow")
            scenes = np.zeros(0, dtype=np.float32)
    np.savez(cache, key=key, audio=audio,
             **({"scenes": scenes} if scenes is not None else {}))

    chat = parse_chat(ing.chat_path) if chat is None else chat
    raw: dict[str, np.ndarray | None] = {
        "chat": chat_series(chat, n_sec, cfg),
        "audio": np.clip(audio, 0, None),
        "speech": speech_series(ing.cues, n_sec),
        "scene": None,
    }
    if scenes is not None and len(scenes):
        c = np.zeros(n_sec, dtype=np.float32)
        for t in scenes:
            if 0 <= int(t) < n_sec:
                c[int(t)] += 1
        raw["scene"] = np.convolve(c, np.ones(10, dtype=np.float32), mode="same")

    series, used = {}, {}
    for k, v in raw.items():
        w = float(cfg.signals.get(k, 0))
        if v is not None and w > 0 and float(np.max(v)) > 0:
            series[k] = _norm(v)
            used[k] = w
    if not series:
        raise RuntimeError("no usable signals (no audio energy?)")
    total = sum(used.values())
    combined = sum(used[k] * series[k] for k in series) / total
    combined = np.convolve(combined, np.ones(5, dtype=np.float32) / 5, mode="same")
    log("signals used: " + ", ".join(f"{k}={w / total:.2f}" for k, w in used.items())
        + ("" if "chat" in used else "  (no chat replay: weights re-balanced)"))
    return Signals(n_sec, series, combined.astype(np.float32), {k: w / total for k, w in used.items()})


# ---------------------------------------------------------------- peak picking -------------
@dataclass
class Window:
    id: int
    peak: float
    start: float
    end: float
    scores: dict[str, float]


def pick_windows(sig: Signals, duration: float, cfg: Config) -> list[Window]:
    """Top-N peaks, windows of window_s centred on the peak, never overlapping.

    The spec asks for 'no overlap' and 'at least 60 s apart'; 90 s windows only avoid overlap when
    peaks are >= 90 s apart, so the effective separation is max(peak_min_gap_s, window_s).
    """
    win = min(cfg.window_s, duration)
    sep = max(cfg.peak_min_gap_s, win)
    order = np.argsort(sig.combined)[::-1]
    peaks: list[int] = []
    for t in order:
        if sig.combined[t] <= 0:
            break
        if all(abs(int(t) - p) >= sep for p in peaks):
            peaks.append(int(t))
            if len(peaks) >= cfg.candidates:
                break
    out = []
    for p in peaks:
        start = float(min(max(0.0, p - win / 2), max(0.0, duration - win)))
        end = start + win
        a, b = int(start), int(end)
        scores = {k: float(v[a:b].mean()) for k, v in sig.series.items()}
        scores["combined"] = float(sig.combined[p])
        out.append(Window(0, float(p), start, end, scores))
    out.sort(key=lambda w: w.start)
    for i, w in enumerate(out):
        w.id = i
    return out
