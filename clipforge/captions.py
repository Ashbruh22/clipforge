from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path

ASSET_FONT = Path(__file__).parent / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"
# The bundled file is Google's static ExtraBold instance; its family name really is this.
FONT_FAMILY = "Montserrat Thin ExtraBold"
FALLBACK_FAMILY = "Arial"
FONT_CSS = "https://fonts.googleapis.com/css?family=Montserrat:800"

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},76,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,{bold},0,0,0,100,100,0,0,1,10,0,5,80,80,0,1
Style: Hook,{font},88,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,{bold},0,0,0,100,100,0,0,1,11,0,8,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
GOLD = r"{\c&H0000D7FF&\fscx110\fscy110}"
RESET = r"{\r}"
HOOK_Y = 300  # \an8: top of the text box; clear of the Shorts top UI band
HOOK_SECONDS = 2.0


def ensure_font(dest: Path) -> str:
    """Copy (or download) Montserrat ExtraBold into dest; returns the ASS font family to use."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / ASSET_FONT.name
    if ASSET_FONT.exists() and ASSET_FONT.stat().st_size > 10_000:
        shutil.copy(ASSET_FONT, target)
        return FONT_FAMILY
    try:  # packaged copy missing: fetch the same file from Google Fonts
        css = urllib.request.urlopen(FONT_CSS, timeout=20).read().decode()
        url = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
        target.write_bytes(urllib.request.urlopen(url, timeout=30).read())
        return FONT_FAMILY
    except Exception:
        return FALLBACK_FAMILY


def _t(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def mask_word(word: str, bad: set[str], mask: str = "*") -> str:
    core = re.sub(r"^\W+|\W+$", "", word)
    if core.lower() in bad and len(core) > 1:
        i = word.index(core)
        return word[:i] + core[0] + mask * (len(core) - 1) + word[i + len(core):]
    return word


def _clean(w: str) -> str:
    return re.sub(r"[{}\\]", "", w).upper()


def group_words(words: list[dict], min_words: int = 2, max_words: int = 4,
                max_chars: int = 22, gap: float = 0.6) -> list[list[dict]]:
    """2-4 words at a time, breaking on pauses and sentence ends, never leaving a lone word
    unless the speech itself pauses."""
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        chars = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
        if cur and (len(cur) >= max_words or chars > max_chars or w["start"] - cur[-1]["end"] > gap):
            groups.append(cur)
            cur = []
        cur.append(w)
        if len(cur) >= min_words and re.search(r"[.?!,;:]$", w["word"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[dict], hook_text: str = "", caption_y: int = 1330, font: str = FONT_FAMILY,
              profanity: dict | None = None) -> str:
    """Word-highlight captions + pop-in hook title. `words` times are relative to clip start."""
    bad = {w.lower() for w in (profanity or {}).get("words", [])} if (profanity or {}).get("enabled") else set()
    mask = (profanity or {}).get("mask", "*") or "*"
    if bad:
        words = [{**w, "word": mask_word(w["word"], bad, mask)} for w in words]
    bold = 0 if font == FONT_FAMILY else -1
    out = [HEADER.format(font=font, bold=bold)]

    if hook_text.strip():
        ws = hook_text.upper().split()
        ws = [_clean(w) for w in ws]
        txt = " ".join(ws) if len(ws) <= 3 else " ".join(ws[: (len(ws) + 1) // 2]) + r"\N" + " ".join(ws[(len(ws) + 1) // 2:])
        pop = (r"{\an8\pos(540,%d)\fscx55\fscy55\alpha&HFF&\t(0,160,\fscx112\fscy112\alpha&H00&)"
               r"\t(160,280,\fscx100\fscy100)\fad(0,200)}" % HOOK_Y)
        out.append(f"Dialogue: 1,{_t(0)},{_t(HOOK_SECONDS)},Hook,,0,0,0,,{pop}{txt}\n")

    groups = group_words(words)
    for gi, g in enumerate(groups):
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        for k, w in enumerate(g):
            start = w["start"]
            end = g[k + 1]["start"] if k + 1 < len(g) else w["end"] + 0.2
            if k + 1 == len(g) and nxt is not None:
                end = min(end, nxt)
            end = max(end, start + 0.05)
            parts = [(GOLD + _clean(x["word"]) + RESET) if j == k else _clean(x["word"])
                     for j, x in enumerate(g)]
            out.append(f"Dialogue: 0,{_t(start)},{_t(end)},Cap,,0,0,0,,"
                       + r"{\an5\pos(540,%d)}" % caption_y + " ".join(parts) + "\n")
    return "".join(out)
