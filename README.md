# ClipForge

Local CLI: a YouTube stream/VOD URL becomes ranked, captioned, vertical 1080×1920 Shorts plus a local
review page. It never uploads anything; you review and post the clips yourself.

```
clipforge run <url> [--max-clips 10] [--layout split] [--dry-run]
clipforge watch                 # poll allowed channels, process new VODs (--once to poll one time)
clipforge calibrate <url>       # save a frame with a grid to read facecam coordinates from
clipforge rerender <video_id>   # re-render after editing config; no download, no re-ranking
clipforge doctor                # check ffmpeg, yt-dlp, claude CLI, GPU/CUDA, Python deps
```

## Setup

Requirements: Python 3.11+, `ffmpeg`/`ffprobe` (with libass) on PATH, and the
[Claude Code](https://claude.com/claude-code) CLI logged in with your subscription.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
clipforge doctor
```

GPU (NVIDIA): faster-whisper needs the CUDA 12 + cuDNN 9 runtime
(`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` works on most setups). NVENC comes with your
NVIDIA driver and an ffmpeg build that includes `h264_nvenc`; `doctor` runs a real test encode.
Without a GPU everything still works: Whisper falls back to CPU int8 `small`, video to libx264.

Edit `config.yaml` (an example is included): list the channels you may clip in `allowed_channels` and
give each a block under `streamers`. **Anything else is refused before a single byte is downloaded.**
Local files need `--streamer @Channel` so the same check applies; sidecar `<stem>.en.vtt` and
`<stem>.live_chat.json` files next to the video are picked up.

Optional `.env` (git-ignored): `GEMINI_API_KEY=...` for the Gemini fallback ranker.

## How it works

1. **Ingest**: yt-dlp downloads ≤1080p mp4, English auto captions (vtt) and live chat replay into
   `work/<id>/`; everything is cached.
2. **Signals** per second, normalised and combined with your weights:
   - chat velocity (messages per 10 s + a bonus for hype tokens), shifted back `chat_lag_seconds`;
   - audio loudness spikes versus a rolling 5-minute baseline (ffmpeg → numpy, streamed);
   - speech density from the captions;
   - scene cuts (PySceneDetect, sampled at ~3 fps, light weight).
   Missing signals (e.g. no chat replay) are dropped and the rest re-weighted.
   The top N peaks (default 40) become 90 s windows. The spec asks for "no overlap" *and* "≥60 s
   apart"; since 90 s windows only avoid overlap when peaks are ≥90 s apart, the effective gap is
   `max(peak_min_gap_s, window_s)`.
3. **Rank** with `claude -p ... --output-format json` (your subscription; the Anthropic API is never
   used and `ANTHROPIC_API_KEY` is stripped from the subprocess). Candidates are batched, malformed
   JSON is retried, and every judgement is cached per candidate, so changing `min_score` or
   `max_clips` costs no model calls. Each `claude -p` call carries Claude Code's fixed context
   overhead, so keep `rank_batch_size` at 5 or more. Set `ranker: gemini` to use Gemini instead.
4. **Cut**: faster-whisper runs only on each chosen range ±2 s (never the full VOD). Model by VRAM:
   ≥6 GB `large-v3`, ≥4 GB `medium`, less `small`; no CUDA → CPU int8 `small`. Start/end snap to
   sentence (else word) boundaries with 0.25 s of breathing room that never reaches into a neighbouring
   word. The cut is a frame-accurate re-encode.
5. **Layout** (1080×1920), per streamer or `--layout`:
   - `split`: facecam on top (35%), gameplay below. Facecam = the fixed rectangle in config, else
     auto-detected (MediaPipe, OpenCV Haar fallback); if none is found it falls back to `center`.
   - `center`: 9:16 crop that follows the main face (deadband smoothing).
   - `blur_fill`: full 16:9 frame over a blurred, zoomed copy of itself.
6. **Captions**: `.ass` with 2–4 words at a time and the spoken word highlighted, Montserrat ExtraBold
   (bundled, OFL), white with a thick black outline; centred on the split line in `split` mode, in the
   Shorts safe zone otherwise. A ≤6-word hook title pops in at the top for the first 2 s. Optional
   profanity masking (`profanity: {enabled: true, words: [...]}`) applies to captions only.
7. **Render**: two-pass `loudnorm` to −14 LUFS, AAC 192k, `+faststart`, 30 or 60 fps matching the
   source. `encoder: auto` uses NVENC (`h264_nvenc -preset p7 -tune hq -rc vbr -cq 19`, GPU decode)
   when ffmpeg lists it *and* a test encode works, otherwise x264 CRF 18 `slow`. Whisper and rendering
   never share the GPU: the model is released before rendering starts.
8. **Quality gate**: drops clips outside 15–59 s, under 1.5 words/s (unless scored ≥9), or overlapping
   a higher-scoring clip from the same VOD. Dropped clips and reasons are listed in `clips.json`.

Output in `output/<video_id>/`: `NN_<slug>.mp4` (+ `.jpg`), `clips.json`, `captions.txt`
(title, description, hashtags and your `required_caption`, ready to paste) and `review.html`.

## Calibrating the facecam

```bash
clipforge calibrate "https://www.youtube.com/watch?v=..." --at 600
```

It saves `work/<id>/calibrate.png`: the frame with a coordinate grid (every 100 px, thick every 500;
the currently configured facecam is outlined in red). Read the facecam's top-left corner and size off
the grid and set `facecam: {x, y, w, h}` for that streamer. If the numbers were measured on a frame
that is not 1920 wide, add `ref_width`. Then `clipforge rerender <video_id>` to check it.

## Tuning the signal weights

`signals: {chat, audio, speech, scene}` are relative weights; they are normalised, so only the ratios
matter. Use `--dry-run` and look at which windows survive, and the per-clip `signals` values in
`clips.json`.

- Clips are all loud but not funny: lower `audio`, raise `chat`.
- Good moments are quiet/dry (talk, reveals): raise `speech`, lower `audio`.
- Chat reacts late or early: adjust `chat_lag_seconds` (default 8).
- Too many false hype hits: trim `hype_tokens`, lower `hype_bonus`.
- Too few clips pass: lower `min_score`, raise `candidates`. Changing these re-uses cached judgements
  wherever possible.

## Tests

`pip install -e ".[dev]" && pytest`: unit tests for every module, real ffmpeg renders of all three
layouts (resolution, fps, captions with the bundled font, −14 LUFS), and a full offline pipeline run
using a fake `claude` executable and a mocked Whisper.

## Known limits

- `watch` skips streams that are still live and, on its first poll, only processes the newest VOD
  (`watch.backfill`) instead of the whole history. YouTube may take a while after a stream ends to
  publish captions and chat replay; a VOD without them still works (Whisper then transcribes only the
  candidate windows) but ranks with fewer signals.
- Face detection uses MediaPipe (its model, ~200 KB, is downloaded once to `~/.cache/clipforge`);
  if MediaPipe can't load, OpenCV Haar is used instead, which is weaker.
- Facecam position for `split` should be configured (fixed overlay); auto-detection is a fallback.
