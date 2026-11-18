from __future__ import annotations

import json
import time
from pathlib import Path

from . import media
from .config import Config, log


def transcribe(wav: Path, cache: Path, cfg: Config, duration: float) -> list[dict]:
    """Word-level transcript [{start,end,word,prob}], cached as JSON."""
    if cache.exists() and not cfg.force:
        return json.loads(cache.read_text())["words"]

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    cuda = cfg.device == "cuda" or (cfg.device == "auto" and media.cuda_available())
    name = cfg.whisper_model if cfg.whisper_model != "auto" else ("large-v3" if cuda else "small")
    log(f"transcribing with {name} on {'CUDA' if cuda else 'CPU'}")
    model = WhisperModel(name, device="cuda" if cuda else "cpu",
                         compute_type="float16" if cuda else "int8")
    kw = dict(language=cfg.language, word_timestamps=True, vad_filter=True)
    if cuda:  # batched decoding is several times faster on a GPU
        segments, info = BatchedInferencePipeline(model).transcribe(str(wav), batch_size=16, **kw)
    else:
        segments, info = model.transcribe(str(wav), beam_size=5, condition_on_previous_text=False, **kw)

    words: list[dict] = []
    last = time.time()
    for seg in segments:
        for w in seg.words or []:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3),
                          "word": w.word.strip(), "prob": round(w.probability, 3)})
        if time.time() - last > 5 and duration:
            log(f"  transcribed {seg.end / duration:.0%}")
            last = time.time()
    cache.write_text(json.dumps({"language": info.language, "model": name, "words": words}))
    return words
