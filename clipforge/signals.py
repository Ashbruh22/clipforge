from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

HOP = 0.5  # seconds per energy frame


def audio_energy(wav: Path) -> np.ndarray:
    """RMS level in dB per HOP seconds, streamed so a 3 h VOD stays in low memory."""
    out: list[np.ndarray] = []
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        hop = int(sr * HOP)
        chunk = hop * 600  # 5 minutes
        while True:
            raw = w.readframes(chunk)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            n = len(x) // hop
            if n == 0:
                break
            rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10)
            out.append(20 * np.log10(rms))
    return np.concatenate(out) if out else np.zeros(1, dtype=np.float32)


def local_zscore(x: np.ndarray, win: int = 600) -> np.ndarray:
    """Robust z-score against a rolling baseline (median/MAD per block, interpolated).

    A loud streamer is not 'exciting' all the time; we want spikes relative to the
    surrounding five minutes.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < 4:
        return np.zeros(n, dtype=np.float32)
    starts = np.arange(0, n, win // 2)
    centers, med, mad = [], [], []
    for s in starts:
        blk = x[max(0, s - win // 4): s + win // 2 + win // 4]
        m = np.median(blk)
        centers.append(s + win // 4)
        med.append(m)
        mad.append(np.median(np.abs(blk - m)) * 1.4826 + 1e-3)
    idx = np.arange(n)
    return ((x - np.interp(idx, centers, med)) / np.interp(idx, centers, mad)).astype(np.float32)


def chat_rate(chat_json: Path | None, duration: float) -> np.ndarray | None:
    """Messages per HOP from a yt-dlp live_chat replay file, or None."""
    if not chat_json or not chat_json.exists():
        return None
    n = int(duration / HOP) + 2
    counts = np.zeros(n, dtype=np.float32)
    seen = 0
    with open(chat_json, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
                off = d["replayChatItemAction"]["videoOffsetTimeMsec"]
                i = int(int(off) / 1000 / HOP)
            except (ValueError, KeyError, TypeError):
                continue
            if 0 <= i < n:
                counts[i] += 1
                seen += 1
    if seen < 50:
        return None
    k = np.ones(6, dtype=np.float32) / 6  # smooth over 3 s
    return np.convolve(counts, k, mode="same")
