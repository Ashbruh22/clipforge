from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config, log
from .signals import Window

PROMPT_VERSION = "v1"

PROMPT = """You are an expert short-form video editor clipping a livestream VOD into YouTube Shorts.
Stream: "{title}" by {channel}.

Below are candidate windows. Each has auto-generated captions (may contain errors) with absolute
timestamps in seconds from the start of the VOD, plus signal scores from 0 to 1 (chat reaction,
audio loudness spike, speech density, scene changes) measuring how lively that window was.

For EVERY candidate decide whether it holds a great, self-contained Short and, if so, where to cut.

Rules:
- "start" and "end" are absolute VOD seconds and must lie inside the candidate's window.
- The clip must be 20-55 seconds. Start right before the setup, end right after the payoff or punchline.
- "hook_text": at most 6 words, an on-screen title for the first 2 seconds.
- "title": a Shorts title of at most 60 characters. No lies, no spoilers.
- "description": 1-2 sentences. "hashtags": 3-5 hashtags, each starting with #.
- "reason": one sentence on why it scores what it does.
- "score": 0-10. Be a tough judge; most windows are not Short-worthy and should score below 7.
- If the candidate should be skipped set "reject_reason" to a short string, otherwise null. Reject when it:
  is a cutscene or story spoiler; has no clear payoff; is dead air or a menu screen; or depends on
  context the viewer won't have.

Reply with ONLY a JSON array (no prose, no code fences), one object per candidate:
[{{"id": <int>, "score": <number>, "start": <number>, "end": <number>, "hook_text": "<str>",
   "title": "<str>", "description": "<str>", "hashtags": ["#a", "#b"], "reason": "<str>",
   "reject_reason": <string or null>}}]

Candidates:
{items}
"""


@dataclass
class Ranked:
    id: int
    window: tuple[float, float]
    score: float
    start: float
    end: float
    hook_text: str = ""
    title: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    reason: str = ""
    reject_reason: str | None = None
    signals: dict = field(default_factory=dict)


# ---------------------------------------------------------------- backends ----------------
class RankerError(Exception):
    pass


def claude_env() -> dict[str, str]:
    """Environment for the claude subprocess: the API key must never be used (subscription only)."""
    env = dict(os.environ)
    if env.pop("ANTHROPIC_API_KEY", None) is not None:
        log("warning: ANTHROPIC_API_KEY is set; ignoring it for the claude CLI so your "
            "subscription is used instead of the API", "yellow")
    return env


def call_claude(prompt: str, cfg: Config) -> str:
    exe = shutil.which("claude")
    if not exe:
        raise RankerError("`claude` CLI not found on PATH (install Claude Code and log in), "
                          "or set ranker: gemini")
    # prompt goes via stdin (same as `claude -p "<prompt>"`) so long batches never hit OS argv limits
    cmd = [exe, "-p", "--output-format", "json"]
    if cfg.claude_model:
        cmd += ["--model", cfg.claude_model]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                       env=claude_env(), timeout=cfg.claude_timeout)
    if r.returncode != 0:
        raise RankerError(f"claude exited {r.returncode}: {(r.stderr or r.stdout)[:300]}")
    try:
        outer = json.loads(r.stdout)
    except json.JSONDecodeError:
        return r.stdout  # plain text fallback
    if isinstance(outer, dict):
        if outer.get("is_error"):
            raise RankerError(f"claude reported an error: {str(outer.get('result'))[:300]}")
        return str(outer.get("result", ""))
    return r.stdout


def call_gemini(prompt: str, cfg: Config) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RankerError("GEMINI_API_KEY not set (put it in .env)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.gemini_model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json"}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            d = json.loads(resp.read())
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RankerError(f"gemini request failed: {e}") from e


BACKENDS = {"claude_cli": call_claude, "gemini": call_gemini}


# ---------------------------------------------------------------- parsing -----------------
def extract_json_array(text: str) -> list:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j < i:
        raise ValueError("no JSON array in reply")
    data = json.loads(t[i: j + 1])
    if not isinstance(data, list):
        raise ValueError("reply is not a JSON array")
    return data


def validate_item(d: dict, w: Window) -> Ranked:
    """Coerce one model answer into a Ranked; raises ValueError when it is unusable."""
    score = float(d["score"])
    start, end = float(d["start"]), float(d["end"])
    if not (0 <= score <= 10):
        raise ValueError("score out of range")
    start, end = max(start, w.start), min(end, w.end)  # keep inside the window
    if end <= start:
        raise ValueError("end <= start")
    hook = " ".join(str(d.get("hook_text") or "").split()[:6])
    title = str(d.get("title") or "").strip()[:60]
    tags = [("#" + str(h).lstrip("#").replace(" ", "")) for h in (d.get("hashtags") or []) if str(h).strip()]
    rr = d.get("reject_reason")
    return Ranked(
        id=w.id, window=(w.start, w.end), score=score, start=start, end=end, hook_text=hook,
        title=title, description=str(d.get("description") or "").strip(), hashtags=tags[:6],
        reason=str(d.get("reason") or "").strip(), reject_reason=str(rr).strip() if rr else None,
        signals=w.scores)


def cand_key(w: Window, text: str, cfg: Config) -> str:
    h = hashlib.sha1(f"{PROMPT_VERSION}|{cfg.ranker}|{w.start:.1f}|{w.end:.1f}|{text}".encode())
    return h.hexdigest()[:16]


def format_candidate(w: Window, lines: list[tuple[float, str]]) -> str:
    sc = " ".join(f"{k}={v:.2f}" for k, v in w.scores.items() if k != "combined")
    body = "\n".join(f"[{t:.1f}] {x}" for t, x in lines) or "(no captions)"
    return (f"### Candidate id={w.id}  window {w.start:.0f}s-{w.end:.0f}s  peak {w.peak:.0f}s\n"
            f"signals: {sc}\n{body}")


# ---------------------------------------------------------------- driver ------------------
def rank_windows(windows: list[Window], lines_for: dict[int, list[tuple[float, str]]], title: str,
                 channel: str, cfg: Config, cache_path: Path) -> list[Ranked]:
    """Judge every window (cached per candidate), then apply min_score / reject / max_clips."""
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    keys = {w.id: cand_key(w, json.dumps(lines_for[w.id]), cfg) for w in windows}
    todo = [w for w in windows if keys[w.id] not in cache]
    if windows and not todo:
        log("ranking: using cached results")
    backend = BACKENDS[cfg.ranker]

    batches: list[list[Window]] = []
    cur: list[Window] = []
    size = 0
    for w in todo:  # batch by count and by prompt size (argv limits)
        n = len(format_candidate(w, lines_for[w.id]))
        if cur and (len(cur) >= cfg.rank_batch_size or size + n > 60_000):
            batches.append(cur)
            cur, size = [], 0
        cur.append(w)
        size += n
    if cur:
        batches.append(cur)

    for bi, batch in enumerate(batches, 1):
        log(f"ranking batch {bi}/{len(batches)} ({len(batch)} candidates) via {cfg.ranker}")
        pending = {w.id: w for w in batch}
        for attempt in range(1, cfg.rank_retries + 1):
            if not pending:
                break
            prompt = PROMPT.format(title=title, channel=channel, items="\n\n".join(
                format_candidate(w, lines_for[w.id]) for w in pending.values()))
            try:
                items = extract_json_array(backend(prompt, cfg))
            except RankerError:
                raise
            except Exception as e:  # malformed JSON etc.: retry
                log(f"  attempt {attempt}: unusable reply ({e})", "yellow")
                continue
            for d in items:
                try:
                    w = pending[int(d["id"])]
                    r = validate_item(d, w)
                except (KeyError, ValueError, TypeError):
                    continue
                cache[keys[w.id]] = asdict(r)
                del pending[w.id]
            if pending:
                log(f"  attempt {attempt}: {len(pending)} candidate(s) missing or invalid", "yellow")
        for w in pending.values():  # give up on these but remember the failure for this run only
            log(f"  candidate {w.id}: no valid answer, skipped", "yellow")
        cache_path.write_text(json.dumps(cache, indent=1))

    ranked = []
    for w in windows:
        d = cache.get(keys[w.id])
        if d:
            d["window"] = tuple(d["window"])
            ranked.append(Ranked(**d))
    return select_ranked(ranked, cfg)


def select_ranked(ranked: list[Ranked], cfg: Config) -> list[Ranked]:
    keep = [r for r in ranked if r.reject_reason is None and r.score >= cfg.min_score
            and cfg.gate_min_s <= r.end - r.start <= cfg.gate_max_s]
    keep.sort(key=lambda r: r.score, reverse=True)
    chosen: list[Ranked] = []
    for r in keep:  # windows don't overlap, but trimmed clips must not either
        if all(r.end <= c.start or r.start >= c.end for c in chosen):
            chosen.append(r)
        if len(chosen) >= cfg.max_clips:
            break
    return chosen
