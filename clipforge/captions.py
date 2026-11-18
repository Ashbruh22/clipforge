from __future__ import annotations

import re

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,84,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,2,5,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
HILITE = r"{\c&H0000D7FF&\fscx112\fscy112}"  # gold, slightly larger
RESET = r"{\r}"


def _t(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _clean(w: str) -> str:
    return re.sub(r"[{}\\]", "", w).upper()


def group_words(words: list[dict], max_words: int = 3, max_chars: int = 20, gap: float = 0.6):
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > gap
                    or sum(len(x["word"]) + 1 for x in cur) + len(w["word"]) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.?!,;:]$", w["word"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[dict], y: int = 1380) -> str:
    """Word-highlight captions. `words` times are relative to clip start."""
    lines = [HEADER]
    groups = group_words(words)
    for gi, g in enumerate(groups):
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        for k, w in enumerate(g):
            start = w["start"]
            if k + 1 < len(g):
                end = g[k + 1]["start"]
            else:
                end = w["end"] + 0.2
                if nxt is not None:
                    end = min(end, nxt)
            end = max(end, start + 0.05)
            parts = [(HILITE + _clean(x["word"]) + RESET) if j == k else _clean(x["word"])
                     for j, x in enumerate(g)]
            text = r"{\an5\pos(540," + str(y) + ")}" + " ".join(parts)
            lines.append(f"Dialogue: 0,{_t(start)},{_t(end)},Default,,0,0,0,,{text}\n")
    return "".join(lines)
