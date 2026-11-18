from __future__ import annotations

import html
import json
from pathlib import Path


def _ts(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClipForge review</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#8b91a1;--ok:#2fb86b;--no:#d65353;--ac:#ffd700}
@media (prefers-color-scheme: light){:root{--bg:#f4f5f8;--card:#fff;--fg:#1b1d24;--mut:#666d7d}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{padding:16px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;flex:1}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;padding:0 20px 40px}
.card{background:var(--card);border-radius:10px;padding:12px;border:2px solid transparent}
.card.approved{border-color:var(--ok)}.card.rejected{opacity:.45;border-color:var(--no)}
video{width:100%;aspect-ratio:9/16;background:#000;border-radius:6px}
.t{font-weight:600;margin:8px 0 2px}.m{color:var(--mut);font-size:13px}
.row{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
button{border:0;border-radius:6px;padding:6px 10px;cursor:pointer;background:#2a2f3b;color:#fff;font:inherit}
button.ok{background:var(--ok)}button.no{background:var(--no)}
.score{color:var(--ac);font-weight:700}
</style></head><body>
<header><h1>__TITLE__</h1><button id="exp">Export approved (JSON)</button></header>
<main id="grid"></main>
<script>
const clips = __DATA__;
const key = "clipforge:" + __KEY__;
let state = {}; try { state = JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} };
const grid = document.getElementById("grid");
function draw() {
  grid.innerHTML = "";
  clips.forEach(c => {
    const st = state[c.file] || "";
    const d = document.createElement("div"); d.className = "card " + st;
    d.innerHTML = `<video controls preload="metadata" poster="${c.thumb}" src="${c.file}"></video>
      <div class="t"></div><div class="m"></div><div class="m"></div>
      <div class="row"><button class="ok">Approve</button><button class="no">Reject</button>
      <button class="cp">Copy title + description</button></div>`;
    d.querySelector(".t").textContent = c.title;
    d.querySelectorAll(".m")[0].innerHTML = `<span class="score">${c.score.toFixed(2)}</span> · ${c.start_ts} → ${c.end_ts} (${c.duration.toFixed(0)}s) · layout: ${c.layout}`;
    d.querySelectorAll(".m")[1].textContent = c.reason || c.description;
    d.querySelector(".ok").onclick = () => { state[c.file] = "approved"; save(); draw(); };
    d.querySelector(".no").onclick = () => { state[c.file] = "rejected"; save(); draw(); };
    d.querySelector(".cp").onclick = () => navigator.clipboard.writeText(
      c.title + "\\n\\n" + c.description + "\\n" + (c.hashtags || []).join(" ") + " #shorts");
    grid.appendChild(d);
  });
}
document.getElementById("exp").onclick = () => {
  const out = clips.filter(c => state[c.file] === "approved");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {type: "application/json"}));
  a.download = "approved_clips.json"; a.click();
};
draw();
</script></body></html>
"""


def write_review(workdir: Path, title: str, items: list[dict]) -> Path:
    data = []
    for it in items:
        data.append({**it, "start_ts": _ts(it["start"]), "end_ts": _ts(it["end"])})
    # keep '</' out of the inline script
    blob = json.dumps(data).replace("</", "<\\/")
    page = (PAGE.replace("__TITLE__", html.escape(title))
            .replace("__DATA__", blob).replace("__KEY__", json.dumps(workdir.name)))
    out = workdir / "review.html"
    out.write_text(page, encoding="utf-8")
    return out
