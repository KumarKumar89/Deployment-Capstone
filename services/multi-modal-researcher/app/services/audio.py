"""Safe podcast audio persistence (replaces legacy ``wave_file`` + print)."""

from __future__ import annotations

import wave
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.middleware import AppError

logger = get_logger(__name__)


def write_wave_file(
    path: Path,
    pcm: bytes,
    channels: int,
    rate: int,
    sample_width: int,
) -> Path:
    """Write raw PCM frames into a WAV file at ``path`` (atomically enough for
    single-writer-per-request semantics)."""
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
    except Exception as exc:  # OSError from wave module etc.
        logger.exception("podcast_write_failed", extra={"context": {"file": path.name}})
        raise AppError("Failed to persist the generated podcast audio.", status_code=500) from exc
    return path


def save_podcast_audio(filename: str, pcm: bytes, settings: Settings | None = None) -> Path:
    """Persist TTS PCM into the dedicated media output directory.

    The filename is sanitized by the caller; we still resolve and verify the
    final path stays inside the media dir (defense in depth against traversal).
    """
    cfg = settings or get_settings()
    media_dir = cfg.resolved_media_output_dir
    media_dir.mkdir(parents=True, exist_ok=True)
    target = (media_dir / filename).resolve()
    if not str(target).startswith(str(media_dir) + "/"):
        raise AppError("Refusing to write outside the media directory.", status_code=500)
    path = write_wave_file(
        target, pcm, cfg.tts_channels, cfg.tts_rate, cfg.tts_sample_width
    )
    logger.info(
        "podcast_saved",
        extra={
            "context": {
                "file": path.name,
                "bytes": path.stat().st_size,
                "directory": str(media_dir),
            }
        },
    )
    return path


def resolve_podcast_path(filename: str) -> Path:
    """Map a safe ``.wav`` filename to its absolute path inside the media dir."""
    from app.schemas.research import is_safe_podcast_filename

    if not is_safe_podcast_filename(filename):
        raise AppError("Invalid podcast filename.", status_code=400)
    settings = get_settings()
    media_dir = settings.resolved_media_output_dir
    candidate = (media_dir / filename).resolve()
    if not str(candidate).startswith(str(media_dir) + "/"):
        raise AppError("Invalid podcast filename.", status_code=400)
    if not candidate.is_file():
        raise AppError("Podcast file not found.", status_code=404)
    return candidate
