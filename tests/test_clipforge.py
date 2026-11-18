import shutil
import subprocess

import numpy as np
import pytest

from clipforge import captions, media, reframe, render, review, select
from clipforge.config import Config
from clipforge.select import Candidate


def make_words(spec):
    """spec: list of (start, text) one sentence each, 0.3s per word."""
    words = []
    for start, text in spec:
        t = start
        for w in text.split():
            words.append({"start": t, "end": t + 0.28, "word": w, "prob": 0.9})
            t += 0.3
    return words


def test_sentences_split_on_punctuation_and_gap():
    ws = make_words([(0, "Hello there friend."), (5, "Second one here")])
    s = select.sentences(ws)
    assert [x["text"] for x in s] == ["Hello there friend.", "Second one here"]


def test_selection_prefers_loud_window_and_no_overlap():
    cfg = Config(min_dur=10, max_dur=30, min_gap=5)
    spec = [(i * 5.0, "this is sentence number %d with some words." % i) for i in range(60)]
    spec[30] = (150.0, "Wait what?! No way, that was insane!")
    words = make_words(spec)
    energy = np.full(int(320 / 0.5), -40.0, dtype=np.float32)
    energy += np.random.default_rng(0).normal(0, 0.5, len(energy)).astype(np.float32)
    energy[int(145 / 0.5): int(160 / 0.5)] += 25  # shout around t=150
    cands = select.build_candidates(words, energy, None, 320.0, cfg)
    picked = select.pick(cands, 3, cfg.min_gap)
    assert picked[0].start <= 150 <= picked[0].end
    for a in picked:
        for b in picked:
            if a is not b:
                assert a.end + cfg.min_gap <= b.start or b.end + cfg.min_gap <= a.start
    assert all(cfg.min_dur - 1 <= c.duration <= cfg.max_dur for c in picked)


def test_lerp_expr_and_smoothing():
    e = reframe.lerp_expr([(0, 100.0), (1, 200.0), (2, 200.0)])
    assert e.count("if(") == 2 and "lt(t,1.000)" in e
    keys = reframe.smooth_track([i / 3 for i in range(30)], [500 + (i % 2) * 5 for i in range(30)],
                                600, 300, 1600)
    assert max(c for _, c in keys) - min(c for _, c in keys) < 20  # jitter suppressed


def test_ass_highlights_current_word():
    ws = [{"start": 0.0, "end": 0.3, "word": "hello"}, {"start": 0.3, "end": 0.6, "word": "world."}]
    ass = captions.build_ass(ws)
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(lines) == 2
    assert "HELLO" in lines[0] and "WORLD." in lines[1]
    assert captions._t(3725.5) == "1:02:05.50"


def test_review_page(tmp_path):
    p = review.write_review(tmp_path, "T </script>", [
        {"file": "clips/a.mp4", "thumb": "clips/a.jpg", "title": "x", "description": "", "hashtags": [],
         "reason": "", "score": 0.5, "start": 1, "end": 40, "duration": 39, "layout": "fit"}])
    txt = p.read_text()
    assert "clips/a.mp4" in txt and "T </script>" not in txt


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")


@needs_ffmpeg
@pytest.mark.parametrize("layout_kind", ["fit", "face", "split"])
def test_render_smoke(tmp_path, layout_kind):
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=6",
                    "-f", "lavfi", "-i", "sine=f=440:d=6", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(src)], check=True)
    info = media.probe(src)
    if layout_kind == "fit":
        layout = reframe.Layout("fit")
    elif layout_kind == "face":
        layout = reframe.Layout("face", [(0.0, 400.0), (3.0, 800.0), (5.0, 800.0)])
    else:
        layout = reframe.Layout("split", face_box=(1000, 500, 150, 150))
    cand = Candidate(0, 0.5, 5.5, 0.5, "hi there")
    words = make_words([(0.6, "hello there my friends.")])
    cfg = Config(encoder="x264")
    out = render.render_clip(src, info, cand, words, layout, tmp_path / "clips" / "o.mp4", cfg)
    o = media.probe(out)
    assert (o.width, o.height) == (1080, 1920)
    assert 4.5 < o.duration < 5.6 and o.has_audio
    assert out.with_suffix(".jpg").exists()
