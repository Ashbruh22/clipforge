from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from . import llm, media, reframe, render, review, select, signals, transcribe
from .config import Config, log
from .download import acquire
from .select import Candidate


def run(source: str, cfg: Config) -> None:
    src = acquire(source, cfg)
    wd = src.workdir
    info = media.probe(src.path)
    log(f"{src.title}: {info.width}x{info.height}, {info.duration / 60:.1f} min")

    wav = wd / "audio.wav"
    if not wav.exists() or cfg.force:
        log("extracting audio")
        media.extract_audio(src.path, wav)

    words = transcribe.transcribe(wav, wd / "transcript.json", cfg, info.duration)
    log(f"{len(words)} words transcribed")

    log("analysing audio energy and chat")
    energy = signals.audio_energy(wav)
    chat = signals.chat_rate(src.chat_path, info.duration)
    log("  chat signal: " + ("yes" if chat is not None else "none"))

    all_c = select.build_candidates(words, energy, chat, info.duration, cfg)
    if not all_c:
        raise SystemExit("no candidate windows found (is there speech in this video?)")
    llm_on = cfg.use_llm and llm.available()
    if cfg.use_llm and not llm_on:
        log("LLM rerank off (needs `pip install anthropic` and ANTHROPIC_API_KEY); using heuristics")
    pool = select.pick(all_c, cfg.clips * 3 if llm_on else cfg.clips, cfg.min_gap)
    if llm_on:
        log(f"LLM reranking {len(pool)} candidates")
        llm.rerank(pool, src.title, cfg)
    final = sorted(pool, key=lambda c: c.score, reverse=True)[: cfg.clips]
    (wd / "candidates.json").write_text(json.dumps([asdict(c) for c in final], indent=2))

    items = []
    for rank, c in enumerate(final, 1):
        name = f"{rank:02d}_{render.slug(c.title)}.mp4"
        out = wd / "clips" / name
        cw = [w for w in words if w["start"] >= c.start - 0.05 and w["end"] <= c.end + 0.05]
        log(f"[{rank}/{len(final)}] {c.start:.0f}s-{c.end:.0f}s score {c.score:.2f}: {c.title}")
        layout = reframe.plan_layout(src.path, c.start, c.duration, info, cfg.layout)
        log(f"  layout: {layout.kind} (face detected in {layout.detect_rate:.0%} of samples)")
        render.render_clip(src.path, info, c, cw, layout, out, cfg)
        items.append({"file": f"clips/{name}", "thumb": f"clips/{name[:-4]}.jpg", "title": c.title,
                      "description": c.description, "hashtags": c.hashtags, "reason": c.reason,
                      "score": float(c.score), "start": c.start, "end": c.end,
                      "duration": c.duration, "layout": layout.kind})
    (wd / "clips.json").write_text(json.dumps(items, indent=2))
    page = review.write_review(wd, src.title, items)
    log(f"\nDone. {len(items)} clips in {wd / 'clips'}\nReview page: {page}")
