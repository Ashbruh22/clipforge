from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from .config import Config, ConfigError, LAYOUTS, load_dotenv, log

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="ClipForge: stream/VOD -> YouTube Shorts. Review the clips, then post them yourself.")


def _cfg(ctx: typer.Context) -> Config:
    try:
        return Config.load(ctx.obj["config"])
    except ConfigError as e:
        log(f"config error: {e}", "red")
        raise typer.Exit(2)


@app.callback()
def main(ctx: typer.Context,
         config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml")):
    ctx.obj = {"config": config}
    load_dotenv()
    if os.environ.get("ANTHROPIC_API_KEY"):
        log("warning: ANTHROPIC_API_KEY is set. ClipForge never uses it; the claude CLI subprocess "
            "runs without it so your subscription is used.", "yellow")


def _layout_opt(v: Optional[str]) -> Optional[str]:
    if v is not None and v not in LAYOUTS:
        raise typer.BadParameter(f"must be one of {', '.join(LAYOUTS)}")
    return v


def _guard(fn):
    from .ingest import Refused

    try:
        return fn()
    except Refused as e:
        log(str(e), "red")
        raise typer.Exit(3)
    except FileNotFoundError as e:
        log(str(e), "red")
        raise typer.Exit(4)


@app.command()
def run(ctx: typer.Context,
        source: str = typer.Argument(..., metavar="URL", help="YouTube URL (or a local video file)"),
        max_clips: Optional[int] = typer.Option(None, "--max-clips", min=1),
        layout: Optional[str] = typer.Option(None, "--layout", help="split | center | blur_fill"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Stop after ranking; print timestamps"),
        streamer: Optional[str] = typer.Option(None, "--streamer", help="Allowed channel (required for local files)")):
    """Download, find, rank, cut and render the best moments of a VOD."""
    from . import pipeline

    cfg = _cfg(ctx)
    if max_clips:
        cfg.max_clips = max_clips
    _guard(lambda: pipeline.run(source, cfg, layout_override=_layout_opt(layout), dry_run=dry_run,
                                streamer=streamer))


@app.command()
def watch(ctx: typer.Context, once: bool = typer.Option(False, "--once", help="Poll once and exit")):
    """Poll allowed channels' /streams pages and process new VODs."""
    from . import pipeline

    _guard(lambda: pipeline.watch(_cfg(ctx), once=once))


@app.command()
def calibrate(ctx: typer.Context, url: str = typer.Argument(..., metavar="URL"),
              at: Optional[float] = typer.Option(None, "--at", help="Seconds into the VOD (default: 25%)")):
    """Save one frame with a coordinate grid so you can read facecam x/y/w/h off it."""
    from . import pipeline

    _guard(lambda: pipeline.calibrate(url, _cfg(ctx), at))


@app.command()
def rerender(ctx: typer.Context, video_id: str = typer.Argument(...),
             layout: Optional[str] = typer.Option(None, "--layout")):
    """Re-render a processed VOD after changing config. No re-download, no re-ranking."""
    from . import pipeline

    _guard(lambda: pipeline.rerender(video_id, _cfg(ctx), _layout_opt(layout)))


@app.command()
def doctor():
    """Check ffmpeg, yt-dlp, claude CLI, GPU/CUDA and Python dependencies."""
    from .env import print_env

    raise typer.Exit(0 if print_env() else 1)


if __name__ == "__main__":
    app()
