from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import log
from .media import MediaInfo

OUT_W, OUT_H = 1080, 1920
TOP_H = round(OUT_H * 0.35)  # 672: facecam
BOT_H = OUT_H - TOP_H  # 1248: gameplay
DET_W = 640
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")


def even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


Rect = tuple[int, int, int, int]  # x, y, w, h in source pixels


@dataclass
class Plan:
    kind: str  # split | center | blur_fill
    cam: Rect | None = None
    game: Rect | None = None
    keys: list[tuple[float, float]] = field(default_factory=list)  # center: (t, crop left edge)
    caption_y: int = 1330
    note: str = ""


# ---------------------------------------------------------------- face detection ------------
class FaceDetector:
    """MediaPipe when usable, OpenCV Haar otherwise. Frames are RGB uint8."""

    def __init__(self):
        self.backend = "haar"
        self._mp = None
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._mp = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5)
                self._api = "solutions"
            else:  # mediapipe >= 0.10.2x ships only the Tasks API and needs a model file
                from mediapipe.tasks import python as mpp
                from mediapipe.tasks.python import vision

                model = _model_path()
                opts = vision.FaceDetectorOptions(
                    base_options=mpp.BaseOptions(model_asset_path=str(model)),
                    min_detection_confidence=0.5)
                self._mp = vision.FaceDetector.create_from_options(opts)
                self._api = "tasks"
            self.backend = "mediapipe"
        except Exception as e:
            log(f"face detection: MediaPipe unavailable ({type(e).__name__}); using OpenCV Haar", "yellow")
            self._use_haar()

    def _use_haar(self) -> None:
        import cv2

        self.backend = "haar"
        self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def largest(self, rgb: np.ndarray) -> Rect | None:
        try:
            return self._largest(rgb)
        except Exception as e:
            if self.backend != "mediapipe":
                raise
            log(f"face detection: MediaPipe failed mid-run ({type(e).__name__}: {e}); switching to Haar", "yellow")
            self._use_haar()
            return self._largest(rgb)

    def _largest(self, rgb: np.ndarray) -> Rect | None:
        h, w = rgb.shape[:2]
        if self.backend == "mediapipe":
            if self._api == "solutions":
                res = self._mp.process(rgb)
                boxes = [(d.location_data.relative_bounding_box.xmin * w,
                          d.location_data.relative_bounding_box.ymin * h,
                          d.location_data.relative_bounding_box.width * w,
                          d.location_data.relative_bounding_box.height * h)
                         for d in (res.detections or [])]
            else:
                import mediapipe as mp

                res = self._mp.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
                boxes = [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width,
                          d.bounding_box.height) for d in res.detections]
        else:
            import cv2

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            boxes = list(self._haar.detectMultiScale(gray, 1.1, 6, minSize=(20, 20)))
        if not boxes:
            return None
        x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(bw), int(bh)


def _model_path() -> Path:
    p = Path.home() / ".cache" / "clipforge" / "blaze_face_short_range.tflite"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, p)
    return p


def sample_frames(src: Path, start: float, dur: float, fps: float, info: MediaInfo) -> tuple[np.ndarray, float]:
    """RGB frames at `fps`, DET_W wide, decoded by ffmpeg so any codec works. Returns (frames, scale)."""
    scale = info.width / DET_W
    h = even(info.height / scale)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
         "-vf", f"fps={fps},scale={DET_W}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (DET_W * h * 3)
    return np.frombuffer(raw[: n * DET_W * h * 3], dtype=np.uint8).reshape(n, h, DET_W, 3), scale


def detect_track(det: FaceDetector, src: Path, start: float, dur: float, info: MediaInfo, fps: float = 3.0):
    frames, scale = sample_frames(src, start, dur, fps, info)
    out = []
    for i, f in enumerate(frames):
        b = det.largest(f)
        out.append((i / fps, tuple(v * scale for v in b)) if b else None)
    return out


# ---------------------------------------------------------------- tracking math -------------
def smooth_track(ts: list[float], xs: list[float], crop_w: float, lo: float, hi: float,
                 step: float = 1.0, deadband: float = 0.10) -> list[tuple[float, float]]:
    """Deadband follow: hold still on small movements, pan smoothly on real ones; ~1 key/second."""
    if not xs:
        return [(0.0, (lo + hi) / 2)]
    a = np.array(xs, dtype=float)
    if len(a) >= 5:  # median filter against false positives
        pad = np.pad(a, 2, mode="edge")
        a = np.median(np.stack([pad[i: i + len(a)] for i in range(5)]), axis=0)
    cur, cs = a[0], []
    for x in a:
        if abs(x - cur) > deadband * crop_w:
            cur += (x - cur) * 0.35
        cs.append(float(np.clip(cur, lo, hi)))
    keys = list(zip(map(float, ts), cs))
    out = [keys[0]]
    for t, c in keys[1:]:
        if t - out[-1][0] >= step:
            out.append((t, c))
    if out[-1][0] < keys[-1][0]:
        out.append(keys[-1])
    return out


def lerp_expr(keys: list[tuple[float, float]]) -> str:
    """Piecewise-linear ffmpeg expression in `t`."""
    if len(keys) == 1:
        return f"{keys[0][1]:.1f}"
    expr = f"{keys[-1][1]:.1f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(keys, keys[1:]))):
        if t1 > t0:
            expr = f"if(lt(t,{t1:.3f}),{x0:.1f}+({x1 - x0:.2f})*(t-{t0:.3f})/{t1 - t0:.3f},{expr})"
    return expr


# ---------------------------------------------------------------- planning ----------------
def scale_rect(r: dict, info: MediaInfo) -> Rect:
    f = info.width / float(r.get("ref_width", 1920))
    x, y, w, h = (float(r[k]) * f for k in ("x", "y", "w", "h"))
    w, h = min(w, info.width), min(h, info.height)
    x, y = max(0, min(x, info.width - w)), max(0, min(y, info.height - h))
    return even(x), even(y), even(w), even(h)


def facecam_from_face(face: Rect, info: MediaInfo) -> Rect:
    """Estimate the facecam rectangle from a detected face (aspect matches the 1080x672 panel)."""
    x, y, w, h = face
    cw = min(info.width, max(w * 2.6, 200))
    ch = cw * TOP_H / OUT_W
    cx, cy = x + w / 2, y + h / 2
    return (even(np.clip(cx - cw / 2, 0, info.width - cw)), even(np.clip(cy - ch / 2, 0, info.height - ch)),
            even(cw), even(ch))


def find_facecam(det: FaceDetector, src: Path, ranges: list[tuple[float, float]], info: MediaInfo,
                 cache: Path | None = None) -> Rect | None:
    """Auto-detect a corner facecam: median face box over frames sampled across the clips."""
    if cache and cache.exists():
        v = json.loads(cache.read_text())
        return tuple(v) if v else None
    boxes, total = [], 0
    for s, e in ranges[:6]:
        for d in detect_track(det, src, s, min(e - s, 30), info, fps=1.0):
            total += 1
            if d:
                boxes.append(d[1])
    rect = None
    if total and len(boxes) / total >= 0.3:
        med = np.median(np.array(boxes), axis=0)
        if med[2] / info.width < 0.35:  # a huge face means a full-frame webcam shot, not a corner cam
            rect = facecam_from_face(tuple(med), info)
    if cache:
        cache.write_text(json.dumps(list(rect) if rect else None))
    return rect


def plan_layout(kind: str, src: Path, start: float, dur: float, info: MediaInfo, streamer: dict,
                det_factory, facecam_cache: Path | None, all_ranges: list[tuple[float, float]]) -> Plan:
    aspect = info.width / info.height
    if kind != "blur_fill" and aspect <= 9 / 16 + 0.02:
        return Plan("blur_fill", note="source already vertical")

    if kind == "split":
        cam = scale_rect(streamer["facecam"], info) if streamer.get("facecam") else \
            find_facecam(det_factory(), src, all_ranges, info, facecam_cache)
        if cam is None:
            log("  split: no facecam configured or detected -> falling back to center", "yellow")
            kind = "center"
        else:
            game = scale_rect(streamer["game"], info) if streamer.get("game") else None
            return Plan("split", cam=cam, game=game, caption_y=TOP_H)

    if kind == "center":
        cw = even(info.height * 9 / 16)
        track = detect_track(det_factory(), src, start, dur, info)
        hits = [(t, b) for d in track if d for t, b in [d]]
        if not hits:
            log("  center: no face found; using a fixed centre crop", "yellow")
            return Plan("center", keys=[(0.0, (info.width - cw) / 2)], caption_y=1330)
        ts = [t for t, _ in hits]
        xs = [b[0] + b[2] / 2 for _, b in hits]
        centres = smooth_track(ts, xs, cw, cw / 2, info.width - cw / 2)
        keys = [(t, c - cw / 2) for t, c in centres]
        if keys[0][0] > 0:
            keys.insert(0, (0.0, keys[0][1]))
        return Plan("center", keys=keys, caption_y=1330)
    return Plan("blur_fill", caption_y=1330)


def video_graph(plan: Plan, info: MediaInfo, fps: int) -> str:
    """Filtergraph from [0:v] to [v0], 1080x1920 at `fps`."""
    W, H = info.width, info.height
    if plan.kind == "blur_fill":
        return (f"[0:v]fps={fps},split=2[a][b];"
                f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
                f"gblur=sigma=40[bg];"
                f"[b]scale={OUT_W}:-2:flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
    if plan.kind == "center":
        cw = even(H * 9 / 16)
        return (f"[0:v]fps={fps},crop={cw}:{H}:x='{lerp_expr(plan.keys)}':y=0,"
                f"scale={OUT_W}:{OUT_H}:flags=lanczos[v0]")
    cx, cy, cw, ch = plan.cam
    if plan.game:
        gx, gy, gw, gh = plan.game
    else:  # centre crop with the gameplay panel's aspect
        gw = even(min(W, H * OUT_W / BOT_H))
        gh = even(min(H, gw * BOT_H / OUT_W))
        gx, gy = even((W - gw) / 2), even((H - gh) / 2)
    return (f"[0:v]fps={fps},split=2[g][c];"
            f"[g]crop={gw}:{gh}:{gx}:{gy},scale={OUT_W}:{BOT_H}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={OUT_W}:{BOT_H}[bot];"
            f"[c]crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{TOP_H}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={OUT_W}:{TOP_H}[top];"
            f"[top][bot]vstack[v0]")
