from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .signals import HOP, local_zscore

HOOK_RE = re.compile(
    r"\b(oh my god|omg|what the|no way|wait|holy|insane|crazy|unbelievable|secret|never|worst|best|"
    r"biggest|literally|can't believe|clutch|let's go|are you kidding|shut up|incredible|actually)\b",
    re.I,
)
WEAK_START = re.compile(r"^(and|but|so|because|which|then|or|also)\b", re.I)
TERMINAL = re.compile(r"[.?!]['\")\]]*$")


@dataclass
class Candidate:
    id: int
    start: float
    end: float
    heur: float
    text: str
    features: dict = field(default_factory=dict)
    score: float = 0.0
    llm_score: float | None = None
    title: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _split_long(ws: list[dict], max_len: float = 12.0) -> list[list[dict]]:
    if len(ws) < 4 or ws[-1]["end"] - ws[0]["start"] <= max_len:
        return [ws]
    lo, hi = len(ws) // 4, max(len(ws) // 4 + 1, 3 * len(ws) // 4)
    k = max(range(lo, hi), key=lambda i: ws[i]["start"] - ws[i - 1]["end"])
    return _split_long(ws[:k], max_len) + _split_long(ws[k:], max_len)


def sentences(words: list[dict], gap: float = 0.9) -> list[dict]:
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > gap:
            groups.append(cur)
            cur = []
        cur.append(w)
        if TERMINAL.search(w["word"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        for part in _split_long(g):
            out.append({"start": part[0]["start"], "end": part[-1]["end"],
                        "text": " ".join(w["word"] for w in part), "n": len(part)})
    return out


def hook_score(text: str) -> float:
    s = 0.0
    if "?" in text:
        s += 0.35
    if HOOK_RE.search(text):
        s += 0.4
    if "!" in text:
        s += 0.2
    if len(text.split()) <= 16:
        s += 0.15
    return min(1.0, s)


def build_candidates(words: list[dict], energy: np.ndarray, chat: np.ndarray | None,
                     duration: float, cfg: Config) -> list[Candidate]:
    sents = sentences(words)
    if not sents:
        return []
    ez = local_zscore(energy)
    cz = local_zscore(chat) if chat is not None else None
    wcfg = ({"energy": .26, "peak": .10, "chat": .16, "density": .14, "hook": .14, "speech": .12, "kw": .08}
            if cz is not None else
            {"energy": .30, "peak": .12, "chat": 0, "density": .16, "hook": .16, "speech": .16, "kw": .10})
    cands: list[Candidate] = []
    n = len(sents)
    for i in range(n):
        for j in range(i, n):
            span = sents[j]["end"] - sents[i]["start"]
            if span > cfg.max_dur - 0.65:
                break
            if span < cfg.min_dur - 0.65:
                continue
            start = max(0.0, sents[i]["start"] - 0.25)
            end = min(duration, sents[j]["end"] + 0.4)
            a, b = int(start / HOP), max(int(end / HOP), int(start / HOP) + 2)
            e = np.sort(ez[a:b])
            if len(e) == 0:
                continue
            top = float(e[-max(1, len(e) // 5):].mean())
            seg = sents[i: j + 1]
            text = " ".join(s["text"] for s in seg)
            dur = end - start
            speech_ratio = min(1.0, sum(s["end"] - s["start"] for s in seg) / dur)
            density = sum(s["n"] for s in seg) / dur
            kw = len(HOOK_RE.findall(text)) / dur * 30
            f = {
                "energy": float(np.clip(top / 3, 0, 1.2)),
                "peak": float(np.clip(e[-1] / 5, 0, 1)),
                "chat": float(np.clip(cz[a:b].mean() / 2, 0, 1)) if cz is not None else 0.0,
                "density": float(np.clip(density / 3.5, 0, 1)),
                "hook": hook_score(seg[0]["text"]),
                "speech": float(speech_ratio),
                "kw": float(np.clip(kw / 3, 0, 1)),
            }
            score = sum(wcfg[k] * f[k] for k in wcfg)
            if WEAK_START.match(seg[0]["text"]):
                score -= 0.08
            if not TERMINAL.search(seg[-1]["text"]):
                score -= 0.06
            if speech_ratio < 0.4:
                score -= 0.1
            score += 0.03 * (1 - abs(dur - 40) / 40)  # mild preference for ~40 s
            cands.append(Candidate(0, start, end, float(score), text, f))
    return cands


def pick(cands: list[Candidate], k: int, min_gap: float) -> list[Candidate]:
    """Greedy non-maximum suppression: best first, no overlaps."""
    chosen: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.heur, reverse=True):
        if all(c.end + min_gap <= o.start or c.start >= o.end + min_gap for o in chosen):
            chosen.append(c)
            if len(chosen) >= k:
                break
    for i, c in enumerate(chosen):
        c.id = i
        c.score = c.heur
        c.title = c.title or _default_title(c.text)
    return chosen


def _default_title(text: str) -> str:
    first = re.split(r"(?<=[.?!])\s", text.strip())[0]
    return (first[:67] + "...") if len(first) > 70 else first
