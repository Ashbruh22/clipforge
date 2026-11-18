from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rich.table import Table

from . import cut, ingest, layout, media, rank, render, review, signals
from .config import Config, console, log
from .ingest import Ingested, Refused
from .rank import Ranked


def _lines_for_window(w: signals.Window, ing: Ingested, runner: cut.WhisperRunner) -> list[tuple[float, str]]:
    """Caption lines for a window. Uses the YouTube captions; if the VOD has none, falls back to
    Whisper on just this window (never on the whole VOD)."""
    if ing.cues:
        return [(c["start"], c["text"]) for c in ing.cues if c["end"] >= w.start and c["start"] <= w.end]
    words = runner.transcribe(w.start, w.end)
    lines, cur = [], []
    for x in words:
        cur.append(x)
        if len(cur) >= 10 or x["word"].endswith((".", "?", "!")):
            lines.append((cur[0]["start"], " ".join(y["word"] for y in cur)))
            cur = []
    if cur:
        lines.append((cur[0]["start"], " ".join(y["word"] for y in cur)))
    return lines


def _print_ranked(ranked: list[Ranked], n_windows: int) -> None:
    t = Table(title=f"Selected {len(ranked)} of {n_windows} candidate windows")
    for c in ("#", "score", "start", "end", "len", "hook", "title"):
        t.add_column(c)
    for i, r in enumerate(ranked, 1):
        t.add_row(str(i), f"{r.score:.1f}", review.hms(r.start), review.hms(r.end), f"{r.end - r.start:.0f}s",
                  r.hook_text, r.title)
    console.print(t)


def _load_ranking(wd: Path) -> list[Ranked]:
    out = []
    for d in json.loads((wd / "ranking.json").read_text()).values():
        d["window"] = tuple(d["window"])
        out.append(Ranked(**d))
    return sorted(out, key=lambda r: r.start)


def run(source: str, cfg: Config, *, layout_override: str | None = None, dry_run: bool = False,
        streamer: str | None = None) -> Path | None:
    p = Path(source).expanduser()
    ing = ingest.ingest_local(p, streamer, cfg) if p.exists() else ingest.ingest_url(source, cfg)
    return process(ing, cfg, layout_override=layout_override, dry_run=dry_run)


def rerender(video_id: str, cfg: Config, layout_override: str | None = None) -> Path | None:
    ing = ingest.load_ingested(video_id, cfg)
    if not (ing.workdir / "ranking.json").exists():
        raise FileNotFoundError(f"{video_id} has never been ranked; run `clipforge run` first")
    return process(ing, cfg, layout_override=layout_override, dry_run=False, reuse_ranking=True)


def process(ing: Ingested, cfg: Config, *, layout_override: str | None, dry_run: bool,
            reuse_ranking: bool = False) -> Path | None:
    info = media.probe(ing.video)
    wd, meta = ing.workdir, ing.meta
    log(f"{meta.title}: {info.width}x{info.height} @ {info.fps:.0f}fps, {info.duration / 60:.1f} min")
    runner = cut.WhisperRunner(ing.video, wd, cfg, info.duration)

    # ---- 1. rank (cached) -------------------------------------------------------------
    if reuse_ranking:
        all_ranked = _load_ranking(wd)
        ranked = rank.select_ranked(all_ranked, cfg)
        log(f"reusing cached ranking: {len(ranked)} clip(s) pass min_score {cfg.min_score}")
    else:
        sig = signals.analyse(ing, info.duration, info.fps, cfg)
        windows = signals.pick_windows(sig, info.duration, cfg)
        log(f"{len(windows)} candidate windows")
        lines_for = {w.id: _lines_for_window(w, ing, runner) for w in windows}
        ranked = rank.rank_windows(windows, lines_for, meta.title, meta.channel, cfg, wd / "ranking.json")
        _print_ranked(ranked, len(windows))
    if dry_run:
        runner.close()
        log("--dry-run: stopping after ranking (times shown are the ranker's, before word snapping)")
        return None
    if not ranked:
        runner.close()
        log("No candidate cleared min_score. Lower min_score or check the signals.", "yellow")
        return None

    # ---- 2. cut: whisper on the chosen ranges only, snap to words, gate ---------------
    log(f"cutting {len(ranked)} clip(s): transcribing only those ranges")
    clips = [cut.build_clip(r, runner, cfg) for r in ranked]
    runner.close()  # free the GPU before rendering: one GPU job at a time
    kept, dropped = cut.quality_gate(clips, cfg)
    for c in dropped:
        log(f"  dropped '{c.r.title}': {c.drop}", "yellow")
    if not kept:
        log("Every clip failed the quality gate.", "red")
        return None

    # ---- 3. layout + captions + render --------------------------------------------------
    sc = cfg.streamer(meta.streamer_key)
    kind = layout_override or sc["layout"]
    encoder = media.resolve_encoder(cfg.encoder)
    out_dir = cfg.output / meta.id
    for f in list(out_dir.glob("[0-9][0-9]_*")):  # stale clips from an earlier run of this VOD
        f.unlink()
    log(f"rendering {len(kept)} clip(s): layout={kind}, encoder={encoder}")
    detector = []

    def det_factory():
        if not detector:
            detector.append(layout.FaceDetector())
        return detector[0]

    ranges = [(c.start, c.end) for c in kept]
    items = []
    for c in kept:
        name = f"{c.index:02d}_{render.slug(c.r.title)}.mp4"
        log(f"[{c.index}/{len(kept)}] {review.hms(c.start)} +{c.duration:.0f}s score {c.r.score:.1f}: {c.r.title}")
        plan = layout.plan_layout(kind, ing.video, c.start, c.duration, info, sc, det_factory,
                                  wd / "facecam.json", ranges)
        stats = render.render_clip(ing.video, info, c, plan, out_dir / name, wd / "build", cfg, encoder)
        caption = review.build_caption(c.r.description, c.r.hashtags, sc.get("required_caption", ""))
        items.append({
            "index": c.index, "file": name, "thumb": name[:-4] + ".jpg", "title": c.r.title,
            "hook_text": c.r.hook_text, "description": c.r.description, "hashtags": c.r.hashtags,
            "caption": caption, "score": c.r.score, "reason": c.r.reason, "layout": plan.kind,
            "source_start": round(c.start, 2), "source_end": round(c.end, 2),
            "source_ts": review.hms(c.start), "duration": round(c.duration, 2),
            "source_link": review.source_link(meta.id, c.start, meta.url), "source_url": meta.url,
            "words_per_second": round(c.wps, 2), "signals": c.r.signals, **stats})
    dropped_meta = [{"title": c.r.title, "score": c.r.score, "reason": c.drop,
                     "source_start": round(c.start, 2), "source_end": round(c.end, 2)} for c in dropped]
    page = review.write_outputs(out_dir, {"id": meta.id, "title": meta.title, "url": meta.url,
                                          "channel": meta.channel}, items, dropped_meta)
    log(f"\nDone: {len(items)} clip(s) in {out_dir}\nReview page: {page.resolve()}")
    return page


# ---------------------------------------------------------------- calibrate -----------------
def calibrate(url: str, cfg: Config, at: float | None = None) -> Path:
    import cv2

    meta = ingest.fetch_meta(url, cfg)
    ingest.require_allowed(meta, cfg)
    wd = cfg.work / meta.id
    wd.mkdir(parents=True, exist_ok=True)
    t = at if at is not None else max(5.0, meta.duration * 0.25)
    frame = wd / "calibrate_frame.png"
    video = next(iter(wd.glob("source.mp4")), None)
    if video:
        src = str(video)
    else:  # no download needed: read one frame straight from the stream
        cookies = ["--cookies-from-browser", cfg.cookies_from_browser] if cfg.cookies_from_browser else []
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--no-playlist", "-g", "-f", "bv*[height<=1080]/b", *cookies, meta.url],
            capture_output=True, text=True, check=True)
        src = r.stdout.split()[0]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.1f}", "-i", src, "-frames:v", "1", str(frame)],
                   check=True)
    img = cv2.imread(str(frame))
    h, w = img.shape[:2]
    step = 100 if w >= 1280 else 50
    for x in range(0, w, step):
        big = x % (step * 5) == 0
        cv2.line(img, (x, 0), (x, h), (0, 255, 255) if big else (255, 255, 255), 2 if big else 1)
        cv2.putText(img, str(x), (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        big = y % (step * 5) == 0
        cv2.line(img, (0, y), (w, y), (0, 255, 255) if big else (255, 255, 255), 2 if big else 1)
        cv2.putText(img, str(y), (3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    fc = cfg.streamer(meta.handle or meta.allowed_as).get("facecam")
    if fc:
        s = w / float(fc.get("ref_width", 1920))
        x, y, fw, fh = (int(float(fc[k]) * s) for k in ("x", "y", "w", "h"))
        cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 0, 255), 3)
    out = wd / "calibrate.png"
    cv2.imwrite(str(out), img)
    log(f"Saved {out.resolve()}  ({w}x{h}, frame at {t:.0f}s)\n"
        f"Read the facecam's top-left (x, y) and size (w, h) off the grid, then set in config.yaml:\n"
        f"  facecam: {{x: ..., y: ..., w: ..., h: ...}}   # red box = currently configured facecam")
    return out


# ---------------------------------------------------------------- watch ---------------------
def watch(cfg: Config, once: bool = False) -> None:
    interval = float(cfg.watch.get("interval_minutes", 30))
    backfill = int(cfg.watch.get("backfill", 1))
    while True:
        st = ingest.load_state(cfg)
        for entry in cfg.allowed_channels:
            try:
                vods = ingest.list_stream_vods(entry)
            except Exception as e:
                log(f"watch: could not list {entry}: {e}", "yellow")
                continue
            new = [v for v in vods if v[0] not in st["seen"]]
            if entry not in st["initialised"]:  # first poll: don't process a whole backlog
                st["seen"] += [v[0] for v in new[backfill:]]
                st["initialised"].append(entry)
                new = new[:backfill]
            for vid, title in reversed(new):  # oldest first
                log(f"watch: new VOD {vid}: {title}")
                try:
                    run(f"https://www.youtube.com/watch?v={vid}", cfg)
                    st["seen"].append(vid)
                except Refused as e:
                    log(str(e), "yellow")
                    st["seen"].append(vid)
                except Exception as e:
                    n = st["attempts"].get(vid, 0) + 1
                    st["attempts"][vid] = n
                    log(f"watch: {vid} failed ({n}/3): {e}", "red")
                    if n >= 3:
                        st["seen"].append(vid)
                ingest.save_state(cfg, st)
            ingest.save_state(cfg, st)
        if once:
            return
        log(f"watch: sleeping {interval:.0f} min")
        time.sleep(interval * 60)
