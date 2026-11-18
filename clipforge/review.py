from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, select_autoescape


def hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def build_caption(description: str, hashtags: list[str], required: str) -> str:
    parts = [p for p in (description.strip(), " ".join(hashtags), required.strip()) if p]
    return "\n".join(parts)


def source_link(video_id: str, start: float, url: str) -> str:
    if url.startswith("http") and "youtu" in url:
        return f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"
    return url


def captions_txt(clips: list[dict]) -> str:
    out = []
    for c in clips:
        out.append(f"=== {c['file']}  (score {c['score']:.1f}) ===\n"
                   f"TITLE: {c['title']}\n\n{c['caption']}\n")
    return "\n".join(out)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClipForge review: {{ meta.title }}</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#8b91a1;--ac:#ffd700;--btn:#2a2f3b}
@media (prefers-color-scheme: light){:root{--bg:#f4f5f8;--card:#fff;--fg:#1b1d24;--mut:#666d7d;--btn:#e3e6ee}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{padding:16px 20px}h1{font-size:18px;margin:0 0 4px}header p{margin:0;color:var(--mut)}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px;padding:0 20px 40px}
.card{background:var(--card);border-radius:10px;padding:12px}
video{width:100%;aspect-ratio:9/16;background:#000;border-radius:6px}
.t{font-weight:600;margin:8px 0 2px}.m{color:var(--mut);font-size:13px;margin:2px 0}
.score{color:var(--ac);font-weight:700}a{color:inherit}
.row{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
button{border:0;border-radius:6px;padding:6px 10px;cursor:pointer;background:var(--btn);color:var(--fg);font:inherit}
</style></head><body>
<header><h1>{{ meta.title }}</h1>
<p>{{ clips|length }} clip(s) &middot; <a href="{{ meta.url }}">source</a> &middot; review, then post them yourself</p></header>
<main>
{% for c in clips %}
<div class="card">
  <video controls preload="metadata" poster="{{ c.thumb }}" src="{{ c.file }}"></video>
  <div class="t">{{ c.title }}</div>
  <div class="m"><span class="score">{{ "%.1f"|format(c.score) }}</span> &middot; {{ c.duration|round(1) }}s &middot; {{ c.layout }}
   &middot; <a href="{{ c.source_link }}" target="_blank" rel="noopener">source {{ c.source_ts }}</a></div>
  <div class="m"><b>Hook:</b> {{ c.hook_text }}</div>
  <div class="m">{{ c.reason }}</div>
  <div class="row">
    <button data-copy="{{ c.title }}">Copy title</button>
    <button data-copy="{{ c.caption }}">Copy caption</button>
  </div>
</div>
{% endfor %}
</main>
<script>
document.querySelectorAll("button[data-copy]").forEach(b => b.onclick = async () => {
  const t = b.dataset.copy, old = b.textContent;
  try { await navigator.clipboard.writeText(t); }
  catch (e) { const a = document.createElement("textarea"); a.value = t; document.body.appendChild(a);
              a.select(); document.execCommand("copy"); a.remove(); }
  b.textContent = "Copied"; setTimeout(() => b.textContent = old, 1200);
});
</script></body></html>
"""


def write_outputs(out_dir: Path, meta: dict, clips: list[dict], dropped: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clips.json").write_text(
        json.dumps({"source": meta, "clips": clips, "dropped": dropped}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "captions.txt").write_text(captions_txt(clips), encoding="utf-8")
    env = Environment(autoescape=select_autoescape(default=True))
    page = out_dir / "review.html"
    page.write_text(env.from_string(TEMPLATE).render(meta=meta, clips=clips), encoding="utf-8")
    return page
