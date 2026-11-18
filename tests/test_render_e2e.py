import json
import os
import stat
import subprocess
import sys

import pytest

from clipforge import cut, layout, media, pipeline, rank, render
from clipforge.ingest import Refused
from clipforge.media import probe
from clipforge.rank import Ranked

from conftest import fake_words, make_cfg, make_video, needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("vid") / "stream.mp4")


def _clip(start=10.0, dur=20.0, score=8.0):
    r = Ranked(0, (0, 60), score, start, start + dur, hook_text="No way that happened",
               title="Great clip", description="d", hashtags=["#a"])
    words = [w for w in fake_words(start, start + dur)]
    return cut.Clip(r, start, start + dur, words, wps=2.5, index=1)


@pytest.mark.parametrize("kind", ["blur_fill", "center", "split"])
def test_render_layouts(tmp_path, video, kind):
    cfg = make_cfg(tmp_path)
    info = probe(video)
    plan = {"blur_fill": layout.Plan("blur_fill"),
            "center": layout.Plan("center", keys=[(0.0, 200.0), (10.0, 600.0), (20.0, 600.0)]),
            "split": layout.Plan("split", cam=(960, 20, 300, 180), caption_y=672)}[kind]
    out = tmp_path / "out" / "01_x.mp4"
    stats = render.render_clip(video, info, _clip(), plan, out, tmp_path / "build", cfg, "x264")
    o = probe(out)
    assert (o.width, o.height) == (1080, 1920) and o.has_audio
    assert 19.5 < o.duration < 20.5 and round(o.fps) == 30 == stats["fps"]
    assert out.with_suffix(".jpg").exists()
    ass = (tmp_path / "build" / "01_x.ass").read_text()
    assert "Montserrat Thin ExtraBold" in ass and ",Hook," in ass


def test_loudness_is_minus_14_lufs(tmp_path, video):
    cfg = make_cfg(tmp_path)
    out = tmp_path / "o" / "01_x.mp4"
    render.render_clip(video, probe(video), _clip(start=30, dur=30), layout.Plan("blur_fill"), out,
                       tmp_path / "b", cfg, "x264")
    m = render.measure_loudness(out, 0, probe(out).duration)
    assert m and abs(m["input_i"] - (-14.0)) < 1.5, m


def test_bundled_font_is_found_by_libass(tmp_path):
    """The ASS family name must resolve to the bundled Montserrat, not a silent fallback."""
    from clipforge import captions

    d = tmp_path / "fonts"
    assert captions.ensure_font(d) == captions.FONT_FAMILY and (d / "Montserrat-ExtraBold.ttf").exists()
    (tmp_path / "t.ass").write_text(captions.build_ass(
        [{"start": 0, "end": 1, "word": "hi"}, {"start": 0, "end": 1, "word": "there"}], "", 900))
    r = subprocess.run(
        ["ffmpeg", "-v", "verbose", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1,format=yuv420p",
         "-vf", "ass=t.ass:fontsdir=fonts", "-frames:v", "1", "-f", "null", "-"],
        cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0
    assert "MontserratThin-ExtraBold" in r.stderr


# ---------------------------------------------------------------- full offline pipeline -----
FAKE_CLAUDE = r'''#!{py}
import json, os, re, sys
log = {log!r}
prompt = sys.stdin.read()
with open(log, "a") as f:
    f.write(json.dumps({{"key": os.environ.get("ANTHROPIC_API_KEY"), "args": sys.argv[1:], "n": len(prompt)}}) + "\n")
n = sum(1 for _ in open(log))
if n == 1:
    print(json.dumps({{"result": "Sorry, here is some prose, not JSON", "is_error": False}}))
    sys.exit(0)
items = []
for m in re.finditer(r"### Candidate id=(\d+)\s+window (\d+)s-(\d+)s", prompt):
    i, a, b = int(m[1]), int(m[2]), int(m[3])
    items.append({{"id": i, "score": 8.5 - i * 0.1, "start": a + 8, "end": a + 33, "hook_text": "Wait for it",
                  "title": f"Epic moment number {{i}}", "description": "Something wild happens.",
                  "hashtags": ["#gaming", "#clutch"], "reason": "Loud reaction then payoff.",
                  "reject_reason": None}})
print(json.dumps({{"result": "```json\n" + json.dumps(items) + "\n```", "is_error": False, "type": "result"}}))
'''


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fake claude is a POSIX script")
def test_full_pipeline_offline(tmp_path, video, monkeypatch):
    # fake `claude` on PATH (the real one is never called), whisper mocked, no network
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "claude_calls.jsonl"
    exe = bindir / "claude"
    exe.write_text(FAKE_CLAUDE.format(py=sys.executable, log=str(log)))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-leak")
    monkeypatch.setattr(cut.WhisperRunner, "transcribe",
                        lambda self, s, e: [w for w in fake_words(max(0, s), min(150, e))])

    # sidecar captions next to the "local VOD"
    vtt = ["WEBVTT", ""]
    for i in range(0, 148, 3):
        vtt += [f"00:{i // 60:02d}:{i % 60:02d}.000 --> 00:{(i + 3) // 60:02d}:{(i + 3) % 60:02d}.000",
                f"line {i} words words words here", ""]
    (video.parent / "stream.en.vtt").write_text("\n".join(vtt))

    cfg = make_cfg(tmp_path)
    cfg.signals = {"chat": 0.45, "audio": 0.35, "speech": 0.15, "scene": 0.05}

    # dry run: ranks, prints, renders nothing
    assert pipeline.run(str(video), cfg, dry_run=True, streamer="@ExampleStreamer") is None
    assert not (tmp_path / "output").exists()
    first_calls = log.read_text().count("\n")
    assert first_calls >= 2  # malformed first reply was retried
    assert all(json.loads(l)["key"] is None for l in log.read_text().splitlines())  # key stripped
    assert all(json.loads(l)["args"] == ["-p", "--output-format", "json"] for l in log.read_text().splitlines())

    # real run: reuses the cached ranking
    page = pipeline.run(str(video), cfg, streamer="@ExampleStreamer")
    assert log.read_text().count("\n") == first_calls  # no new model calls
    out = tmp_path / "output" / "stream"
    assert page == out / "review.html" and page.exists()
    data = json.loads((out / "clips.json").read_text())
    clips = data["clips"]
    assert 1 <= len(clips) <= 3
    for c in clips:
        o = probe(out / c["file"])
        assert (o.width, o.height) == (1080, 1920) and 15 <= o.duration <= 59
        assert c["layout"] == "split" and c["words_per_second"] >= 1.5
        assert c["source_end"] - c["source_start"] == pytest.approx(o.duration, abs=0.6)
        assert "@ExampleStreamer #ExampleTag" in c["caption"] and c["hashtags"] == ["#gaming", "#clutch"]
    txt = (out / "captions.txt").read_text()
    assert clips[0]["title"] in txt and "@ExampleStreamer #ExampleTag" in txt
    html = page.read_text()
    assert all(c["file"] in html for c in clips) and "Copy title" in html
    # clips never overlap each other
    spans = sorted((c["source_start"], c["source_end"]) for c in clips)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))

    # rerender with a different layout: no ranking calls, files replaced
    pipeline.rerender("stream", cfg, layout_override="blur_fill")
    assert log.read_text().count("\n") == first_calls
    clips2 = json.loads((out / "clips.json").read_text())["clips"]
    assert {c["layout"] for c in clips2} == {"blur_fill"}


def test_nvenc_args_and_encoder_resolution(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    a = render.encoder_args("nvenc", cfg)
    assert a[:2] == ["-c:v", "h264_nvenc"] and "p7" in a and "hq" in a and "-cq" in a and a[a.index("-cq") + 1] == "19"
    x = render.encoder_args("x264", make_cfg(tmp_path, encode={"crf": 18, "preset": "slow", "nvenc_cq": 19}))
    assert x[x.index("-crf") + 1] == "18" and x[x.index("-preset") + 1] == "slow"
    monkeypatch.setattr(media, "nvenc_works", lambda: False)
    assert media.resolve_encoder("auto") == "x264" and media.resolve_encoder("nvenc") == "x264"
    monkeypatch.setattr(media, "nvenc_works", lambda: True)
    assert media.resolve_encoder("auto") == "nvenc" and media.resolve_encoder("x264") == "x264"


def test_calibrate_draws_grid(tmp_path, video, monkeypatch):
    from clipforge import ingest

    cfg = make_cfg(tmp_path)
    meta = ingest.VideoMeta("vid1", "T", "https://youtu.be/vid1", handle="@ExampleStreamer", duration=150)
    monkeypatch.setattr(ingest, "fetch_meta", lambda u, c: meta)
    wd = tmp_path / "work" / "vid1"
    wd.mkdir(parents=True)
    (wd / "source.mp4").write_bytes(video.read_bytes())
    out = pipeline.calibrate("https://youtu.be/vid1", cfg, at=5)
    import cv2

    img = cv2.imread(str(out))
    assert img.shape[:2] == (720, 1280) and out.name == "calibrate.png"
    other = ingest.VideoMeta("v2", "T", "u", handle="@Other")
    monkeypatch.setattr(ingest, "fetch_meta", lambda u, c: other)
    with pytest.raises(Refused):
        pipeline.calibrate("u", cfg)
