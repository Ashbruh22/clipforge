from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config, log, norm_channel


class Refused(Exception):
    """Channel is not in allowed_channels (or can't be identified)."""


@dataclass
class VideoMeta:
    id: str
    title: str
    url: str
    channel: str = ""
    handle: str = ""
    channel_id: str = ""
    channel_url: str = ""
    duration: float = 0.0
    live_status: str = ""
    allowed_as: str = ""  # the allowed_channels entry that matched
    streamer_key: str = ""


@dataclass
class Ingested:
    meta: VideoMeta
    workdir: Path
    video: Path
    cues: list[dict] = field(default_factory=list)  # [{start,end,text}] from captions
    chat_path: Path | None = None


def _ytdlp(*args: str) -> list[str]:
    return [sys.executable, "-m", "yt_dlp", *args]


def slugify(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()[:n] or "video"


# ---------------------------------------------------------------- allowlist ---------------
def match_allowed(meta: VideoMeta, allowed: list[str]) -> str | None:
    ids = {norm_channel(x) for x in (meta.handle, meta.channel_id, meta.channel, meta.channel_url) if x}
    for a in allowed:
        if norm_channel(a) in ids:
            return a
    return None


def require_allowed(meta: VideoMeta, cfg: Config) -> None:
    hit = match_allowed(meta, cfg.allowed_channels)
    if not hit:
        who = meta.handle or meta.channel or meta.channel_id or "unknown channel"
        raise Refused(
            f"Refusing to process '{meta.title}': channel {who} is not in allowed_channels. "
            f"Add it to config.yaml if you are cleared to clip it.")
    meta.allowed_as = hit
    meta.streamer_key = cfg.streamer(hit)["_key"] or hit


def fetch_meta(url: str, cfg: Config) -> VideoMeta:
    cookies = ["--cookies-from-browser", cfg.cookies_from_browser] if cfg.cookies_from_browser else []
    out = subprocess.run(
        _ytdlp("--no-playlist", "--skip-download", "--dump-single-json", *cookies, url),
        check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    handle = d.get("uploader_id") or ""
    if not handle.startswith("@"):
        tail = (d.get("uploader_url") or d.get("channel_url") or "").rstrip("/").split("/")[-1]
        handle = tail if tail.startswith("@") else ""
    return VideoMeta(
        id=d["id"], title=d.get("title") or d["id"], url=d.get("webpage_url") or url,
        channel=d.get("channel") or d.get("uploader") or "", handle=handle,
        channel_id=d.get("channel_id") or "", channel_url=d.get("channel_url") or "",
        duration=float(d.get("duration") or 0), live_status=d.get("live_status") or "")


# ---------------------------------------------------------------- captions (vtt) ----------
_TS = re.compile(r"(?:(\d+):)?(\d+):(\d+)[.,](\d+)")
_TAG = re.compile(r"<[^>]+>")


def _sec(m: re.Match) -> float:
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def parse_vtt(text: str) -> list[dict]:
    """WebVTT -> [{start,end,text}], de-duplicating YouTube's rolling two-line auto captions."""
    cues: list[tuple[float, float, list[str]]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = block.strip().split("\n")
        i = next((k for k, l in enumerate(lines) if "-->" in l), None)
        if i is None:
            continue
        a, _, b = lines[i].partition("-->")
        ma, mb = _TS.search(a), _TS.search(b)
        if not ma or not mb:
            continue
        body = [html.unescape(_TAG.sub("", l)).strip() for l in lines[i + 1:]]
        cues.append((_sec(ma), _sec(mb), [l for l in body if l]))
    out: list[dict] = []
    prev_lines: set[str] = set()
    for s, e, lines in cues:
        fresh = [l for l in lines if l not in prev_lines]
        prev_lines = set(lines)
        if fresh:
            out.append({"start": s, "end": e, "text": " ".join(fresh)})
    for a, b in zip(out, out[1:]):  # a line stays on screen until the next one appears
        a["end"] = min(max(b["start"], a["start"] + 0.5), a["start"] + 8.0)
    return out


# ---------------------------------------------------------------- chat --------------------
def parse_chat(path: Path | None) -> list[tuple[float, str]]:
    """yt-dlp live_chat replay (JSON lines) -> [(seconds, text)]."""
    if not path or not path.exists():
        return []
    out: list[tuple[float, str]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                a = json.loads(line)["replayChatItemAction"]
                off = int(a["videoOffsetTimeMsec"]) / 1000
            except (ValueError, KeyError, TypeError):
                continue
            if off < 0:
                continue
            for act in a.get("actions", []):
                item = act.get("addChatItemAction", {}).get("item", {})
                r = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
                if not r:
                    continue
                parts = []
                for run in r.get("message", {}).get("runs", []):
                    if "text" in run:
                        parts.append(run["text"])
                    elif "emoji" in run:
                        e = run["emoji"]
                        eid = e.get("emojiId", "")
                        parts.append(eid if len(eid) <= 4 else (e.get("shortcuts") or [""])[0])
                out.append((off, "".join(parts)))
    return out


# ---------------------------------------------------------------- download ----------------
def _first(wd: Path, pattern: str) -> Path | None:
    return next(iter(sorted(wd.glob(pattern))), None)


def ingest_url(url: str, cfg: Config) -> Ingested:
    meta = fetch_meta(url, cfg)
    require_allowed(meta, cfg)  # BEFORE anything is downloaded
    if meta.live_status in ("is_live", "is_upcoming"):
        raise Refused(f"'{meta.title}' is still live/upcoming; wait for the VOD.")
    wd = cfg.work / meta.id
    wd.mkdir(parents=True, exist_ok=True)
    cookies = ["--cookies-from-browser", cfg.cookies_from_browser] if cfg.cookies_from_browser else []
    tmpl = str(wd / "source.%(ext)s")

    video = _first(wd, "source.mp4") or _first(wd, "source.mkv") or _first(wd, "source.webm")
    if not video:
        h = cfg.max_height
        log(f"downloading video {meta.id}: {meta.title}")
        subprocess.run(
            _ytdlp("--no-playlist", "-f", f"bv*[height<={h}]+ba/b[height<={h}]",
                   "--merge-output-format", "mp4", "--concurrent-fragments", "4", "--retries", "10",
                   "-o", tmpl, *cookies, meta.url), check=True)
        video = _first(wd, "source.mp4") or _first(wd, "source.mkv") or _first(wd, "source.webm")
        if not video:
            raise RuntimeError("yt-dlp produced no video file")

    # captions and chat are best-effort: a failure here must not abort the run
    if not _first(wd, "source.en*.vtt"):
        subprocess.run(_ytdlp("--no-playlist", "--skip-download", "--write-subs", "--write-auto-subs",
                              "--sub-langs", "en", "--sub-format", "vtt", "-o", tmpl, *cookies, meta.url),
                       capture_output=True)
    if not _first(wd, "source.live_chat.json"):
        subprocess.run(_ytdlp("--no-playlist", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
                              "-o", tmpl, *cookies, meta.url), capture_output=True)
    return _finish(meta, wd, video)


def ingest_local(path: str | Path, streamer: str | None, cfg: Config) -> Ingested:
    p = Path(path).expanduser().resolve()
    if not streamer:
        raise Refused("Local files need --streamer <allowed channel> so the allowlist can be checked.")
    meta = VideoMeta(id=slugify(p.stem), title=p.stem, url=p.as_uri(), channel=streamer, handle=streamer)
    require_allowed(meta, cfg)
    wd = cfg.work / meta.id
    wd.mkdir(parents=True, exist_ok=True)
    # optional sidecars next to the file: <stem>.en.vtt / <stem>.vtt and <stem>.live_chat.json
    for pat, dst in ((f"{p.stem}.en*.vtt", "source.en.vtt"), (f"{p.stem}.vtt", "source.en.vtt"),
                     (f"{p.stem}.live_chat.json", "source.live_chat.json")):
        src = next(iter(sorted(p.parent.glob(pat))), None)
        if src and not (wd / dst).exists():
            shutil.copy(src, wd / dst)
    return _finish(meta, wd, p)


def _finish(meta: VideoMeta, wd: Path, video: Path) -> Ingested:
    vtt = _first(wd, "source.en*.vtt")
    cues = parse_vtt(vtt.read_text(encoding="utf-8", errors="ignore")) if vtt else []
    chat = _first(wd, "source.live_chat.json")
    (wd / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    (wd / "video_path.txt").write_text(str(video), encoding="utf-8")
    log(f"captions: {len(cues)} lines | chat replay: {'yes' if chat else 'none'}")
    return Ingested(meta, wd, video, cues, chat)


def load_ingested(video_id: str, cfg: Config) -> Ingested:
    """Rebuild an Ingested from the work cache (rerender), without touching the network."""
    wd = cfg.work / video_id
    mp = wd / "meta.json"
    if not mp.exists():
        raise FileNotFoundError(f"no cached run for '{video_id}' under {cfg.work}/")
    meta = VideoMeta(**json.loads(mp.read_text()))
    vp = wd / "video_path.txt"
    video = Path(vp.read_text(encoding="utf-8").strip()) if vp.exists() else \
        (_first(wd, "source.mp4") or _first(wd, "source.mkv") or _first(wd, "source.webm"))
    if not video or not Path(video).exists():
        raise FileNotFoundError(f"source video for '{video_id}' not found in {wd}")
    return _finish(meta, wd, Path(video))


# ---------------------------------------------------------------- watch -------------------
def channel_streams_url(entry: str) -> str:
    e = entry.strip()
    if e.startswith("http"):
        return e.rstrip("/") + ("" if e.rstrip("/").endswith("/streams") else "/streams")
    if e.startswith("UC") and len(e) > 20:
        return f"https://www.youtube.com/channel/{e}/streams"
    return f"https://www.youtube.com/{e if e.startswith('@') else '@' + e}/streams"


def list_stream_vods(entry: str, limit: int = 10) -> list[tuple[str, str]]:
    """Newest-first (video_id, title) of finished streams on a channel's /streams tab."""
    r = subprocess.run(
        _ytdlp("--flat-playlist", "--playlist-end", str(limit),
               "--print", "%(id)s\t%(live_status)s\t%(title)s", channel_streams_url(entry)),
        capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        vid, _, rest = line.partition("\t")
        status, _, title = rest.partition("\t")
        if vid and status not in ("is_live", "is_upcoming"):
            out.append((vid, title))
    return out


def load_state(cfg: Config) -> dict:
    p = cfg.work / "watch_state.json"
    return json.loads(p.read_text()) if p.exists() else {"seen": [], "attempts": {}, "initialised": []}


def save_state(cfg: Config, st: dict) -> None:
    cfg.work.mkdir(parents=True, exist_ok=True)
    (cfg.work / "watch_state.json").write_text(json.dumps(st, indent=2))
