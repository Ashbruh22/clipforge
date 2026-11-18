from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

from .media import MediaInfo

OUT_W, OUT_H = 1080, 1920
SPLIT_TOP_H = 1056
SPLIT_BOT_H = OUT_H - SPLIT_TOP_H  # 864
SAMPLE_FPS = 3
DET_W = 640


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


@dataclass
class Layout:
    kind: str  # fit | face | split
    keys: list[tuple[float, float]] = field(default_factory=list)  # face: (t, crop centre x px)
    face_box: tuple[float, float, float, float] | None = None  # split: median x,y,w,h in source px
    detect_rate: float = 0.0


def detect_faces(src, start: float, dur: float, info: MediaInfo):
    """Largest Haar face per sampled frame -> list of (t, x, y, w, h) in source px or None.

    Frames come from ffmpeg (not cv2.VideoCapture) so any codec the user's ffmpeg decodes works.
    """
    import cv2

    scale = info.width / DET_W
    h = _even(info.height / scale)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-vf", f"fps={SAMPLE_FPS},scale={DET_W}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = len(frames) // (DET_W * h)
    frames = frames[: n * DET_W * h].reshape(n, h, DET_W)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    out = []
    for i in range(n):
        fs = det.detectMultiScale(frames[i], scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        if len(fs):
            x, y, w, hh = max(fs, key=lambda r: r[2] * r[3])
            out.append((i / SAMPLE_FPS, x * scale, y * scale, w * scale, hh * scale))
        else:
            out.append(None)
    return out


def smooth_track(ts: list[float], xs: list[float], crop_w: float, lo: float, hi: float,
                 step: float = 1.0, deadband: float = 0.10) -> list[tuple[float, float]]:
    """Deadband follow: hold still for small movements, pan smoothly for real ones.

    Returns ~1 keyframe per `step` seconds, clamped so the crop stays inside the frame.
    """
    if not xs:
        return [(0.0, (lo + hi) / 2)]
    xs_a = np.array(xs, dtype=float)
    if len(xs_a) >= 5:  # median filter against false positives
        pad = np.pad(xs_a, 2, mode="edge")
        xs_a = np.median(np.stack([pad[i: i + len(xs_a)] for i in range(5)]), axis=0)
    cur = xs_a[0]
    cs = []
    for x in xs_a:
        if abs(x - cur) > deadband * crop_w:
            cur += (x - cur) * 0.35
        cs.append(float(np.clip(cur, lo, hi)))
    keys = [(float(t), c) for t, c in zip(ts, cs)]
    sampled = [keys[0]]
    for t, c in keys[1:]:
        if t - sampled[-1][0] >= step:
            sampled.append((t, c))
    if sampled[-1][0] < keys[-1][0]:
        sampled.append(keys[-1])
    return sampled


def plan_layout(src, start: float, dur: float, info: MediaInfo, mode: str = "auto") -> Layout:
    aspect = info.width / info.height
    if aspect <= 9 / 16 + 0.02 or mode == "fit":
        return Layout("fit")
    det = detect_faces(src, start, dur, info)
    hits = [(i, d) for i, d in enumerate(det) if d]
    rate = len(hits) / max(1, len(det))
    if mode == "auto" and rate < 0.3:
        return Layout("fit", detect_rate=rate)

    boxes = np.array([d[1:] for _, d in hits]) if hits else np.zeros((0, 4))
    med = np.median(boxes, axis=0) if len(boxes) else None
    kind = mode
    if mode == "auto":
        face_frac = med[2] / info.width
        cx = (med[0] + med[2] / 2) / info.width
        cy = (med[1] + med[3] / 2) / info.height
        corner = (cx < 0.33 or cx > 0.67) and (cy < 0.4 or cy > 0.6)
        kind = "split" if (face_frac < 0.14 and corner) else "face"
    if med is None:  # forced layout but no face found: centre
        return Layout("fit" if kind != "face" else "face", [(0.0, info.width / 2)], None, rate)
    if kind == "split":
        return Layout("split", face_box=tuple(med), detect_rate=rate)

    crop_w = _even(info.height * 9 / 16)
    ts = [i / SAMPLE_FPS for i, _ in hits]
    xs = [d[1] + d[3] / 2 for _, d in hits]
    keys = smooth_track(ts, xs, crop_w, crop_w / 2, info.width - crop_w / 2)
    if keys[0][0] > 0:
        keys.insert(0, (0.0, keys[0][1]))
    return Layout("face", keys, detect_rate=rate)


def lerp_expr(keys: list[tuple[float, float]]) -> str:
    """Piecewise-linear ffmpeg expression in `t`."""
    if len(keys) == 1:
        return f"{keys[0][1]:.1f}"
    expr = f"{keys[-1][1]:.1f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(keys, keys[1:]))):
        if t1 <= t0:
            continue
        expr = f"if(lt(t,{t1:.3f}),{x0:.1f}+({x1 - x0:.2f})*(t-{t0:.3f})/{t1 - t0:.3f},{expr})"
    return expr


def video_graph(layout: Layout, info: MediaInfo) -> str:
    """Filtergraph from [0:v] to [v0] at 1080x1920, 30 fps."""
    W, H = info.width, info.height
    if layout.kind == "fit":
        return (
            f"[0:v]fps=30,split=2[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},gblur=sigma=35[bg];"
            f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]"
        )
    if layout.kind == "face":
        cw = _even(H * 9 / 16)
        # crop x is the left edge; keyframes hold the centre
        left = [(t, c - cw / 2) for t, c in layout.keys]
        return (f"[0:v]fps=30,crop={cw}:{H}:x='{lerp_expr(left)}':y=0,"
                f"scale={OUT_W}:{OUT_H}:flags=lanczos[v0]")
    # split: gameplay on top, facecam below
    gw = _even(min(W, H * OUT_W / SPLIT_TOP_H))
    gx = _even((W - gw) / 2)
    fx0, fy0, fw0, fh0 = layout.face_box
    fw = _even(min(W, max(fw0 * 3.4, 240)))
    fh = _even(fw * SPLIT_BOT_H / OUT_W)
    if fh > H:
        fh = _even(H)
        fw = _even(fh * OUT_W / SPLIT_BOT_H)
    fx = _even(np.clip(fx0 + fw0 / 2 - fw / 2, 0, W - fw))
    fy = _even(np.clip(fy0 + fh0 / 2 - fh * 0.45, 0, H - fh))
    return (
        f"[0:v]fps=30,split=2[g][f];"
        f"[g]crop={gw}:{H}:{gx}:0,scale={OUT_W}:{SPLIT_TOP_H}:flags=lanczos[top];"
        f"[f]crop={fw}:{fh}:{fx}:{fy},scale={OUT_W}:{SPLIT_BOT_H}:flags=lanczos[bot];"
        f"[top][bot]vstack[v0]"
    )


def caption_y(layout: Layout) -> int:
    return SPLIT_TOP_H if layout.kind == "split" else 1380
