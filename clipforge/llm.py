from __future__ import annotations

import importlib.util
import json
import os

from .config import Config, log
from .select import Candidate

PROMPT = """You are an editor picking the best moments of a livestream to post as YouTube Shorts.
Stream title: {title}

Below are candidate clips with transcripts. For each, judge: does it open with a hook, is it a
self-contained moment (funny, surprising, insightful, high-stakes), does it end cleanly?
Return ONLY a JSON array, one object per candidate:
[{{"id": <int>, "score": <0-10 float>, "title": "<punchy title, max 70 chars, no clickbait lies>",
   "description": "<1-2 sentences>", "hashtags": ["#tag", ...max 4], "reason": "<short why>"}}]

Candidates:
{items}
"""


def available() -> bool:
    return importlib.util.find_spec("anthropic") is not None and bool(os.environ.get("ANTHROPIC_API_KEY"))


def rerank(cands: list[Candidate], video_title: str, cfg: Config) -> None:
    """Adds LLM score/title/description; blends with heuristic score. Failures are non-fatal."""
    if not cands:
        return
    import anthropic

    client = anthropic.Anthropic()
    for i in range(0, len(cands), 8):
        batch = cands[i: i + 8]
        items = "\n\n".join(
            f"[id={c.id}] ({c.duration:.0f}s)\n{c.text[:1800]}" for c in batch)
        try:
            resp = client.messages.create(
                model=cfg.llm_model, max_tokens=4000,
                messages=[{"role": "user", "content": PROMPT.format(title=video_title, items=items)}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            data = json.loads(text[text.index("["): text.rindex("]") + 1])
        except Exception as e:  # network, parse, quota: keep heuristic result
            log(f"LLM rerank failed for batch {i // 8}: {e}")
            continue
        by_id = {c.id: c for c in batch}
        for d in data:
            c = by_id.get(d.get("id"))
            if not c:
                continue
            c.llm_score = float(d.get("score", 0))
            c.title = str(d.get("title") or c.title)[:100]
            c.description = str(d.get("description", ""))
            c.hashtags = [str(h) for h in d.get("hashtags", [])][:4]
            c.reason = str(d.get("reason", ""))
    hs = [c.heur for c in cands]
    lo, hi = min(hs), max(hs)
    for c in cands:
        h = (c.heur - lo) / (hi - lo) if hi > lo else 0.5
        c.score = 0.35 * h + 0.65 * (c.llm_score / 10) if c.llm_score is not None else 0.35 * h
