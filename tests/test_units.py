import json
import urllib.request

import numpy as np
import pytest

from clipforge import captions, cut, ingest, layout, rank, review, signals
from clipforge.config import Config, ConfigError, norm_channel
from clipforge.ingest import Refused, VideoMeta
from clipforge.media import MediaInfo
from clipforge.rank import Ranked
from clipforge.signals import Window

from conftest import fake_words, make_cfg


# ---------------------------------------------------------------- config / allowlist ------
def test_config_validation_and_unknown_keys(tmp_path):
    with pytest.raises(ConfigError):
        Config.from_dict({"ranker": "nope"})
    with pytest.raises(ConfigError):
        Config.from_dict({"signals": {"chat": 0, "audio": 0, "speech": 0, "scene": 0}})
    cfg = Config.from_dict({"bogus": 1, "signals": {"chat": 0.9}})
    assert cfg.signals["audio"] == 0.35 and cfg.signals["chat"] == 0.9  # partial dict merges over defaults
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "missing.yaml")


def test_streamer_lookup_is_lenient():
    cfg = make_cfg(__import__("pathlib").Path("/tmp/x"))
    assert cfg.streamer("exampleSTREAMER")["required_caption"].startswith("@Example")
    assert cfg.streamer("@someone")["layout"] == "split"


def test_allowlist_matching():
    cfg = Config.from_dict({"allowed_channels": ["@ExampleStreamer", "UCabcdefghijklmnopqrstuv"]})
    m = VideoMeta("v", "t", "u", handle="@examplestreamer")
    ingest.require_allowed(m, cfg)
    assert m.allowed_as == "@ExampleStreamer"
    m2 = VideoMeta("v", "t", "u", channel_id="UCabcdefghijklmnopqrstuv")
    ingest.require_allowed(m2, cfg)
    with pytest.raises(Refused, match="not in allowed_channels"):
        ingest.require_allowed(VideoMeta("v", "t", "u", handle="@Other", channel="Other"), cfg)
    assert norm_channel("https://www.youtube.com/@Foo/") == "foo"


def test_url_ingest_refuses_before_downloading(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(ingest, "fetch_meta", lambda url, c: VideoMeta("abc", "T", url, handle="@Other"))
    called = []
    monkeypatch.setattr(ingest.subprocess, "run", lambda *a, **k: called.append(a))
    with pytest.raises(Refused):
        ingest.ingest_url("https://youtu.be/abc", cfg)
    assert called == [] and not (tmp_path / "work").exists()


def test_local_file_needs_allowed_streamer(tmp_path):
    cfg = make_cfg(tmp_path)
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    with pytest.raises(Refused):
        ingest.ingest_local(f, None, cfg)
    with pytest.raises(Refused):
        ingest.ingest_local(f, "@Nope", cfg)


# ---------------------------------------------------------------- vtt / chat ---------------
VTT = """WEBVTT
Kind: captions

00:00:01.000 --> 00:00:03.000 align:start position:0%
hello everyone<00:00:01.500><c> welcome</c>

00:00:03.000 --> 00:00:03.010 align:start position:0%
hello everyone welcome

00:00:03.010 --> 00:00:05.000 align:start position:0%
hello everyone welcome
to the stream &amp; stuff

00:00:05.000 --> 00:00:05.010
to the stream &amp; stuff

00:00:05.010 --> 00:00:07.000
to the stream &amp; stuff
today we play
"""


def test_parse_vtt_dedupes_rolling_captions():
    cues = ingest.parse_vtt(VTT)
    texts = [c["text"] for c in cues]
    assert texts == ["hello everyone welcome", "to the stream & stuff", "today we play"]
    assert cues[0]["start"] == 1.0 and cues[1]["start"] == 3.01
    assert all(c["end"] > c["start"] for c in cues)


def _chat_line(sec, text):
    return json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(int(sec * 1000)), "actions": [
        {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {"message": {"runs": [{"text": text}]}}}}}]}})


def test_parse_chat_and_emoji(tmp_path):
    p = tmp_path / "c.json"
    emoji = json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": "2000", "actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"message": {"runs": [{"emoji": {"emojiId": "😂"}}, {"text": " lol"}]}}}}}]}})
    p.write_text(_chat_line(1, "hi") + "\n" + emoji + "\nnot json\n")
    assert ingest.parse_chat(p) == [(1.0, "hi"), (2.0, "😂 lol")]
    assert ingest.parse_chat(None) == []


# ---------------------------------------------------------------- signals -----------------
def test_chat_series_lag_and_hype(tmp_path):
    cfg = make_cfg(tmp_path, chat_lag_seconds=8)
    chat = [(t + 0.5, "hello") for t in range(100) for _ in range(1)]  # baseline: 1 msg/s
    chat += [(60.2, "LMAO"), (60.4, "W"), (60.6, "wow")] * 3  # spike at t=60 (+hype)
    s = signals.chat_series(chat, 120, cfg)
    assert int(np.argmax(s)) in range(60 - 8 - 6, 60 - 8 + 6)  # moment = spike time minus lag
    assert signals.chat_series(chat[:10], 120, cfg) is None  # too little chat to trust
    assert signals._tokens_match("W", ["w"]) and not signals._tokens_match("wow", ["w"])
    assert signals._tokens_match("CLIP IT!!", ["clip it"]) and signals._tokens_match("so 💀", ["💀"])


def test_rolling_z_and_pick_windows(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 1200).astype(np.float32)
    base[300:310] += 20
    base[900:905] += 20
    z = signals.rolling_z(base)
    assert z[305] > 5 and abs(z[600]) < 4
    cfg = make_cfg(tmp_path, window_s=90, peak_min_gap_s=60, candidates=10)
    comb = np.clip(z, 0, None) / z.max()
    sig = signals.Signals(1200, {"audio": comb}, comb, {"audio": 1.0})
    wins = signals.pick_windows(sig, 1200.0, cfg)
    assert len(wins) >= 2
    for a in wins:
        assert a.end - a.start == 90 and 0 <= a.start and a.end <= 1200
        for b in wins:
            if a is not b:
                assert a.end <= b.start or b.end <= a.start  # no overlap
                assert abs(a.peak - b.peak) >= 60  # and 60 s apart
    assert any(w.start <= 305 <= w.end for w in wins)


def test_speech_series():
    s = signals.speech_series([{"start": 0, "end": 4, "text": "a b c d e f g h"}], 20)
    assert s.max() > 0.5 and signals.speech_series([], 20) is None


# ---------------------------------------------------------------- rank --------------------
def test_extract_json_and_validate():
    arr = rank.extract_json_array('Sure!\n```json\n[{"id": 0}]\n```')
    assert arr == [{"id": 0}]
    with pytest.raises(ValueError):
        rank.extract_json_array("no array here")
    w = Window(3, 100, 50, 140, {"chat": 0.5})
    r = rank.validate_item({"id": 3, "score": 8.5, "start": 10, "end": 200, "hook_text": "one two three four five six seven",
                            "title": "x" * 90, "hashtags": ["a", "#b"], "reject_reason": None}, w)
    assert (r.start, r.end) == (50, 140) and r.hook_text.count(" ") == 5 and len(r.title) == 60
    assert r.hashtags == ["#a", "#b"] and r.signals == {"chat": 0.5}
    with pytest.raises(ValueError):
        rank.validate_item({"id": 3, "score": 11, "start": 60, "end": 90}, w)


def _ranked(i, score, a, b, rej=None):
    return Ranked(i, (a, b), score, a, b, title=f"t{i}", reject_reason=rej)


def test_select_ranked_rules(tmp_path):
    cfg = make_cfg(tmp_path, min_score=7, max_clips=2)
    rs = [_ranked(0, 9, 0, 30), _ranked(1, 6.9, 100, 130), _ranked(2, 8, 200, 230, "no payoff"),
          _ranked(3, 7.5, 300, 320), _ranked(4, 8.2, 400, 440), _ranked(5, 7.1, 500, 530),
          _ranked(6, 9.5, 600, 605)]  # too short
    out = rank.select_ranked(rs, cfg)
    assert [r.id for r in out] == [0, 4]


def test_rank_windows_retries_malformed_and_caches(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, rank_batch_size=2)
    wins = [Window(i, 50 + 100 * i, 100 * i, 100 * i + 90, {"chat": 0.5}) for i in range(3)]
    lines = {w.id: [(w.start + 1, "text")] for w in wins}
    calls = []

    def backend(prompt, c):
        calls.append(prompt)
        if len(calls) == 1:
            return "I cannot produce JSON, sorry"
        ids = [int(x) for x in __import__("re").findall(r"id=(\d+)", prompt)]
        return json.dumps([{"id": i, "score": 8, "start": wins[i].start + 5, "end": wins[i].start + 35,
                            "title": f"clip {i}", "hook_text": "h", "reason": "r"} for i in ids])

    monkeypatch.setitem(rank.BACKENDS, "claude_cli", backend)
    cache = tmp_path / "ranking.json"
    out = rank.rank_windows(wins, lines, "T", "C", cfg, cache)
    assert len(out) == 3 and len(calls) == 3  # 1 malformed retry + 2 batches
    n = len(calls)
    out2 = rank.rank_windows(wins, lines, "T", "C", cfg, cache)
    assert len(calls) == n and [r.id for r in out2] == [r.id for r in out]


def test_gemini_backend(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(rank.RankerError, match="GEMINI_API_KEY"):
        rank.call_gemini("p", cfg)
    monkeypatch.setenv("GEMINI_API_KEY", "k123")
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps({"candidates": [{"content": {"parts": [{"text": "[]"}]}}]}).encode()

    def fake(req, timeout=0):
        seen["url"], seen["hdr"] = req.full_url, dict(req.header_items())
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert rank.call_gemini("p", cfg) == "[]"
    assert "k123" not in seen["url"] and seen["hdr"]["X-goog-api-key"] == "k123"


def test_claude_env_strips_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    env = rank.claude_env()
    assert "ANTHROPIC_API_KEY" not in env


# ---------------------------------------------------------------- cut ---------------------
def test_snap_never_cuts_mid_word_and_breathes():
    ws = fake_words(0, 60)
    a, b = cut.snap(ws, 10.4, 40.2)
    for w in ws:  # cut points fall in gaps between words, not inside one
        assert not (w["start"] < a < w["end"]) and not (w["start"] < b < w["end"])
    starts = [w for w in ws if w["start"] >= a]
    assert starts[0]["start"] - a <= cut.BREATH + 1e-6 and starts[0]["start"] - a > 0
    assert cut.snap([], 0, 1) is None
    # snaps to the sentence start ("word10." ends a sentence) near the request
    s_words = {w["start"] for w in ws if ws.index(w) and ws[ws.index(w) - 1]["word"].endswith(".")}
    first = starts[0]["start"]
    assert first in s_words


def test_quality_gate(tmp_path):
    cfg = make_cfg(tmp_path)
    mk = lambda i, a, b, score, wps: cut.Clip(_ranked(i, score, a, b), a, b, [], wps)  # noqa: E731
    clips = [mk(0, 0, 30, 8, 2.5),       # keep
             mk(1, 100, 110, 9.9, 3),    # too short (10 s)
             mk(2, 200, 270, 8, 3),      # too long (70 s)
             mk(3, 300, 330, 7.5, 1.0),  # too few words
             mk(4, 400, 430, 9.2, 1.0),  # few words but score >= 9: exempt
             mk(5, 20, 50, 7, 3)]        # overlaps clip 0
    kept, dropped = cut.quality_gate(clips, cfg)
    assert sorted(c.r.id for c in kept) == [0, 4]
    reasons = {c.r.id: c.drop for c in dropped}
    assert "length" in reasons[1] and "length" in reasons[2] and "words/s" in reasons[3] and "overlaps" in reasons[5]
    assert [c.index for c in kept] == [1, 2] and kept[0].r.id == 4  # ordered by score


def test_pick_model_by_vram(monkeypatch, tmp_path):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(cut.media, "cuda_available", lambda: True)
    for vram, name in ((8188, "large-v3"), (6144, "large-v3"), (4096, "medium"), (2048, "small")):
        monkeypatch.setattr(cut.media, "vram_mib", lambda v=vram: v)
        assert cut.pick_model(cfg) == (name, "cuda", "float16")
    monkeypatch.setattr(cut.media, "cuda_available", lambda: False)
    assert cut.pick_model(cfg) == ("small", "cpu", "int8")


# ---------------------------------------------------------------- captions ----------------
def test_group_words_2_to_4():
    ws = fake_words(0, 30, wps=3)
    for g in captions.group_words(ws)[:-1]:
        assert 2 <= len(g) <= 4


def test_ass_hook_highlight_and_profanity():
    ws = [{"start": 1.0, "end": 1.3, "word": "what"}, {"start": 1.3, "end": 1.6, "word": "damn"},
          {"start": 1.6, "end": 2.0, "word": "happened?"}]
    ass = captions.build_ass(ws, "No way that just happened right now", 672, "Montserrat Thin ExtraBold",
                             {"enabled": True, "words": ["damn"], "mask": "*"})
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(lines) == 4  # 1 hook + 3 words
    assert ",Hook," in lines[0] and "0:00:00.00,0:00:02.00" in lines[0] and r"\t(" in lines[0]
    assert r"\N" in lines[0]  # 7-word hook is wrapped to two lines
    assert "D***" in ass and "DAMN" not in ass
    assert r"\pos(540,672)" in lines[1]
    assert lines[1].count(captions.GOLD) == 1 and lines[1].index(captions.GOLD) < lines[1].index("WHAT") + 1
    assert captions._t(3725.5) == "1:02:05.50"
    plain = captions.build_ass(ws, "", 1330)
    assert ",Hook," not in plain.split("[Events]")[1]


# ---------------------------------------------------------------- layout ------------------
INFO = MediaInfo(1920, 1080, 100.0, 30.0, True)


def test_lerp_expr_and_smoothing():
    e = layout.lerp_expr([(0, 100.0), (1, 200.0), (2, 200.0)])
    assert e.count("if(") == 2 and "lt(t,1.000)" in e
    keys = layout.smooth_track([i / 3 for i in range(30)], [500 + (i % 2) * 5 for i in range(30)], 600, 300, 1600)
    assert max(c for _, c in keys) - min(c for _, c in keys) < 20
    far = layout.smooth_track([i / 3 for i in range(30)], [400] * 15 + [1500] * 15, 600, 300, 1620)
    assert far[-1][1] > 900  # follows a real move


def test_scale_rect_and_split_graph():
    r = layout.scale_rect({"x": 1520, "y": 40, "w": 380, "h": 214}, INFO)
    assert r == (1520, 40, 380, 214)
    half = layout.scale_rect({"x": 760, "y": 20, "w": 190, "h": 107, "ref_width": 960}, INFO)
    assert half[0] == 1520
    g = layout.video_graph(layout.Plan("split", cam=r), INFO, 30)
    assert "vstack" in g and f"{layout.OUT_W}:{layout.TOP_H}" in g and layout.TOP_H == 672 and layout.BOT_H == 1248
    assert "crop=380:214:1520:40" in g
    assert "gblur" in layout.video_graph(layout.Plan("blur_fill"), INFO, 60) and "fps=60" in layout.video_graph(layout.Plan("blur_fill"), INFO, 60)


def test_facecam_estimate_matches_panel_aspect():
    x, y, w, h = layout.facecam_from_face((1700, 100, 100, 100), INFO)
    assert abs(w / h - 1080 / 672) < 0.05 and x + w <= 1920 and y + h <= 1080


def test_split_falls_back_when_no_facecam(tmp_path, monkeypatch):
    class NoFace:
        def largest(self, rgb): return None
        backend = "none"

    monkeypatch.setattr(layout, "detect_track", lambda *a, **k: [None, None])
    p = layout.plan_layout("split", tmp_path / "x.mp4", 0, 10, INFO, {"layout": "split"}, lambda: NoFace(),
                           None, [(0, 10)])
    assert p.kind == "center" and len(p.keys) == 1


# ---------------------------------------------------------------- review / watch ----------
def test_review_outputs_escape_and_files(tmp_path):
    c = {"index": 1, "file": "01_a.mp4", "thumb": "01_a.jpg", "title": "<b>x</b>", "hook_text": "h",
         "description": "d", "hashtags": ["#a"], "caption": "d\n#a\n@S #T", "score": 8.0, "reason": "because",
         "layout": "split", "source_ts": "0:01:00", "duration": 30.0, "source_link": "https://www.youtube.com/watch?v=v&t=60s"}
    page = review.write_outputs(tmp_path, {"id": "v", "title": "T <script>", "url": "https://y/v"}, [c], [])
    html = page.read_text()
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html and "T <script>" not in html
    assert "01_a.mp4" in html and "t=60s" in html and "Copy title" in html and "Copy caption" in html
    assert "TITLE: <b>x</b>" in (tmp_path / "captions.txt").read_text()
    assert json.loads((tmp_path / "clips.json").read_text())["clips"][0]["source_link"].endswith("t=60s")
    assert review.build_caption("Desc", ["#a", "#b"], "@S #T") == "Desc\n#a #b\n@S #T"


def test_watch_backfill_and_state(tmp_path, monkeypatch):
    from clipforge import pipeline

    cfg = make_cfg(tmp_path, watch={"interval_minutes": 1, "backfill": 1})
    vods = [("new2", "t"), ("new1", "t"), ("old", "t")]
    monkeypatch.setattr(ingest, "list_stream_vods", lambda e, limit=10: vods)
    ran = []
    monkeypatch.setattr(pipeline, "run", lambda url, c, **k: ran.append(url))
    pipeline.watch(cfg, once=True)
    assert ran == ["https://www.youtube.com/watch?v=new2"]  # backlog skipped, newest processed
    pipeline.watch(cfg, once=True)
    assert len(ran) == 1  # nothing new
    vods.insert(0, ("fresh", "t"))
    pipeline.watch(cfg, once=True)
    assert ran[-1].endswith("fresh")
    st = ingest.load_state(cfg)
    assert set(st["seen"]) == {"new2", "new1", "old", "fresh"}
    assert ingest.channel_streams_url("@Foo") == "https://www.youtube.com/@Foo/streams"
