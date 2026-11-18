import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from clipforge.config import Config

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")


def make_cfg(tmp_path, **kw) -> Config:
    base = dict(
        allowed_channels=["@ExampleStreamer"],
        streamers={"@ExampleStreamer": {"layout": "split", "facecam": {"x": 960, "y": 20, "w": 300, "h": 180,
                                                                       "ref_width": 1280},
                                        "required_caption": "@ExampleStreamer #ExampleTag"}},
        work_dir=str(tmp_path / "work"), output_dir=str(tmp_path / "output"),
        encoder="x264", encode={"crf": 28, "preset": "ultrafast", "nvenc_cq": 19},
        window_s=40, candidates=3, min_score=7, max_clips=3)
    base.update(kw)
    return Config.from_dict(base)


def make_video(path: Path, seconds=150, bursts=((40, 52), (110, 122))) -> Path:
    """1280x720 test pattern with quiet audio and two loud bursts."""
    sr = 16000
    t = np.arange(seconds * sr) / sr
    rng = np.random.default_rng(1)
    a = 0.02 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(len(t))
    for s, e in bursts:
        m = (t >= s) & (t < e)
        a[m] = 0.5 * rng.standard_normal(m.sum()) * 0.6
    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r=30:d={seconds}",
                    "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
                    str(path)], check=True)
    wav.unlink()
    return path


def fake_words(start: float, end: float, wps: float = 2.5) -> list[dict]:
    """Words with a sentence end every 10 words."""
    out, t, i = [], start + 0.3, 0
    while t + 0.3 < end:
        i += 1
        out.append({"start": round(t, 3), "end": round(t + 0.3, 3),
                    "word": f"word{i}" + ("." if i % 10 == 0 else "")})
        t += 1 / wps
    return out
