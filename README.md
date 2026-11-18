# ClipForge

Local CLI: YouTube stream/VOD URL (or local file) → ranked, captioned, vertical 1080x1920 Shorts + a review page.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[llm]"        # drop [llm] if you don't want AI reranking
clipforge doctor               # checks ffmpeg, yt-dlp, NVENC, CUDA, deps
```

Needs `ffmpeg` (with libass) on PATH. For GPU transcription on an NVIDIA card, faster-whisper needs
CUDA 12 + cuDNN 9 runtime libraries (`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` works on
most setups). `clipforge doctor` shows whether CUDA and NVENC are picked up.

## Use

```bash
clipforge run "https://www.youtube.com/watch?v=..." -n 8
clipforge run stream.mp4 --layout split --no-llm
```

Output goes to `clipforge_out/<video-id>/`: `clips/*.mp4` (+ thumbnails and `.ass` captions),
`review.html` (open in a browser; approve/reject, copy titles, export approved JSON),
`candidates.json`, `transcript.json`. Audio and transcript are cached, so re-runs with different
`-n`/`--layout` skip the slow steps (`--force` to redo).

Set `ANTHROPIC_API_KEY` to enable LLM reranking and generated titles/descriptions
(`--llm-model` to change model, `--no-llm` to disable).

## Pipeline

1. **Download** (yt-dlp, ≤1080p; chat replay if the VOD has one).
2. **Transcribe** (faster-whisper, word timestamps; `large-v3` batched fp16 on CUDA, `small` int8 on CPU).
3. **Signals**: audio loudness spikes vs. a rolling 5-minute baseline, chat-message rate, speech density.
4. **Candidates**: sentence-aligned windows (20–55 s) scored on energy, peak, chat, hook words/questions,
   speech coverage; greedy non-overlapping selection.
5. **LLM rerank** (optional): scores hook/self-containedness, writes title, description, hashtags.
6. **Reframe** per clip: `face` (pans to follow the speaker), `split` (gameplay on top, facecam below,
   chosen when a small face sits in a corner), or `fit` (blurred-background letterbox when no face).
7. **Render**: ffmpeg, NVENC (`h264_nvenc`) when available, else libx264 CRF 18; word-highlight
   captions; loudness normalised to -14 LUFS.

## Tests

`pip install -e ".[dev]" && pytest` (includes real ffmpeg render smoke tests on synthetic video).

## Known limits

- Face detection is OpenCV Haar (no model download, CPU). Fine for webcam streamers; swapping in a
  stronger detector lives in `reframe.detect_faces`.
- Pinned to OpenCV 4.x (5.x removed `CascadeClassifier`).
- Live (in-progress) streams are not supported; use the VOD once it ends.
