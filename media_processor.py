"""
media_processor.py — Extract and analyse audio/video evidence.

Pipeline for a video file:
  1. Extract audio track → temporary WAV file.
  2. Transcribe audio via Groq Whisper API (whisper-large-v3).
  3. Extract a set of key frames (one per N seconds).
  4. Send each frame to the Groq vision LLM for description.
  5. Merge transcript + frame descriptions into a structured report.

Pipeline for an audio file:
  1. Transcribe directly via Groq Whisper.

Design decisions:
  - moviepy is used for frame extraction and audio separation because it
    ships on all platforms without requiring a system FFmpeg binary.
    However, FFmpeg IS needed by moviepy under the hood — we check for it.
  - Temporary files are created in the system temp directory and cleaned
    up in a finally block so they are never left on disk.
  - No shell=True subprocess calls are made.

Security notes:
  - File path is resolved and validated before any processing.
  - File size is capped at MAX_VIDEO_BYTES.
  - Frame count is capped at MAX_FRAMES to limit API cost and time.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}

ALL_MEDIA_EXTS = _VIDEO_EXTS | _AUDIO_EXTS

MAX_VIDEO_BYTES: int = 200 * 1024 * 1024   # 200 MB
MAX_FRAMES:      int = 10                  # max key frames to analyse
FRAME_INTERVAL:  int = 30                  # extract one frame every N seconds

_WHISPER_MODEL = "whisper-large-v3"
_VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Public result type ────────────────────────────────────────────────────────


class MediaResult:
    """
    Holds the results of media (video / audio) processing.

    Attributes:
        file_path        : Absolute path of the processed file.
        media_type       : "Video" or "Audio".
        transcript       : Full Whisper transcription (may be empty for silent video).
        frame_descriptions: List of (timestamp_sec, description) for each analysed frame.
        duration_sec     : Duration in seconds (-1 if unknown).
        error            : Error message if processing failed.
    """

    def __init__(
        self,
        file_path: Path,
        media_type: str,
        transcript: str = "",
        frame_descriptions: Optional[list[tuple[int, str]]] = None,
        duration_sec: float = -1,
        error: str = "",
    ) -> None:
        self.file_path          = file_path
        self.media_type         = media_type
        self.transcript         = transcript.strip()
        self.frame_descriptions = frame_descriptions or []
        self.duration_sec       = duration_sec
        self.error              = error

    @property
    def full_content(self) -> str:
        """Return a structured text summary for LLM context injection."""
        parts: list[str] = []

        if self.duration_sec > 0:
            mins, secs = divmod(int(self.duration_sec), 60)
            parts.append(f"[Media Duration: {mins}m {secs}s]")

        if self.transcript:
            parts.append("[Audio Transcript]\n" + self.transcript)

        if self.frame_descriptions:
            frame_lines = ["[Visual Frame Analysis]"]
            for ts, desc in self.frame_descriptions:
                mins, secs = divmod(ts, 60)
                frame_lines.append(f"\n  Timestamp {mins:02d}:{secs:02d}")
                frame_lines.append("  " + desc.replace("\n", "\n  "))
            parts.append("\n".join(frame_lines))

        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not self.transcript and not self.frame_descriptions

    def __repr__(self) -> str:
        return (
            f"MediaResult(type={self.media_type!r}, "
            f"transcript_chars={len(self.transcript)}, "
            f"frames={len(self.frame_descriptions)}, "
            f"error={self.error!r})"
        )


# ── Entry-point ───────────────────────────────────────────────────────────────


def process_media(
    path: str | Path,
    groq_client,  # groq.Groq instance — required
) -> MediaResult:
    """
    Process an audio or video file.

    Args:
        path:        Path to the media file.
        groq_client: A ``groq.Groq`` instance.

    Returns:
        A :class:`MediaResult` with transcript and/or frame descriptions.

    Raises:
        FileNotFoundError — if *path* does not exist.
        ValueError        — for unsupported extension or file too large.
    """
    if groq_client is None:
        raise ValueError("A Groq client is required for media processing.")

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")

    size = resolved.stat().st_size
    if size == 0:
        raise ValueError("Media file is empty.")
    if size > MAX_VIDEO_BYTES:
        mb = size / (1024 * 1024)
        raise ValueError(
            f"Media file too large ({mb:.1f} MB). "
            f"Maximum allowed size is {MAX_VIDEO_BYTES // (1024 * 1024)} MB."
        )

    ext = resolved.suffix.lower()
    if ext not in ALL_MEDIA_EXTS:
        raise ValueError(
            f"Unsupported media type: '{ext}'. "
            f"Supported: {', '.join(sorted(ALL_MEDIA_EXTS))}"
        )

    if ext in _AUDIO_EXTS:
        return _process_audio(resolved, groq_client)
    else:
        return _process_video(resolved, groq_client)


# ── Audio processing ──────────────────────────────────────────────────────────


def _transcribe_audio_file(audio_path: Path, groq_client) -> str:
    """
    Send an audio file to Groq Whisper for transcription.
    Returns the transcript text or empty string on failure.
    """
    try:
        with open(audio_path, "rb") as f:
            resp = groq_client.audio.transcriptions.create(
                file=(audio_path.name, f),
                model=_WHISPER_MODEL,
                response_format="verbose_json",
                language="en",          # hint; Whisper auto-detects if wrong
                temperature=0.0,
            )
        # verbose_json gives segments with timestamps — flatten to text
        if hasattr(resp, "segments") and resp.segments:
            lines = []
            for seg in resp.segments:
                start = int(getattr(seg, "start", 0))
                mins, secs = divmod(start, 60)
                text = getattr(seg, "text", "").strip()
                lines.append(f"[{mins:02d}:{secs:02d}] {text}")
            return "\n".join(lines)
        # Fallback for plain text response
        return getattr(resp, "text", "") or ""
    except Exception as exc:
        logger.error("Whisper transcription failed for %s: %s", audio_path.name, exc)
        return ""


def _process_audio(path: Path, groq_client) -> MediaResult:
    """Transcribe a standalone audio file."""
    transcript = _transcribe_audio_file(path, groq_client)
    error = "" if transcript else "Audio transcription failed or produced no output."
    return MediaResult(
        file_path=path,
        media_type="Audio",
        transcript=transcript,
        error=error,
    )


# ── Video processing ──────────────────────────────────────────────────────────

_FRAME_PROMPT = """\
This is a frame extracted from a video uploaded as evidence in an Indian legal \
situation. Describe everything that is legally relevant in this frame.

Include:
- All visible text (signs, messages, documents, labels)
- Identifiable people and what they are doing
- Objects, locations, or settings
- Any visible actions or events
- Any damage, injury, or unusual conditions
- Any timestamps, dates, or location information visible

Clearly distinguish between what you can see versus what you are inferring. \
Be factual and concise. Do not make legal conclusions."""


def _describe_frame(
    frame_path: Path,
    groq_client,
    timestamp_sec: int,
) -> tuple[int, str]:
    """Send a single JPEG frame to the vision LLM for description."""
    try:
        with open(frame_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        resp = groq_client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": _FRAME_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.1,
        )
        desc = resp.choices[0].message.content or "(no description)"
        return (timestamp_sec, desc.strip())
    except Exception as exc:
        logger.warning("Frame description failed at %ds: %s", timestamp_sec, exc)
        return (timestamp_sec, f"(frame analysis failed: {exc})")


def _process_video(path: Path, groq_client) -> MediaResult:
    """
    Full video pipeline:
    1. Extract audio → Whisper transcript.
    2. Extract key frames → vision descriptions.
    """
    tmp_dir = tempfile.mkdtemp(prefix="ala_video_")
    transcript = ""
    frame_descriptions: list[tuple[int, str]] = []
    duration_sec: float = -1
    errors: list[str] = []

    try:
        # ── Check moviepy availability ────────────────────────────────
        try:
            from moviepy.editor import VideoFileClip  # type: ignore[import]
        except ImportError:
            return MediaResult(
                file_path=path,
                media_type="Video",
                error=(
                    "moviepy is not installed. "
                    "Install it: pip install moviepy\n"
                    "FFmpeg is also required: https://ffmpeg.org/download.html"
                ),
            )

        # ── Load video ────────────────────────────────────────────────
        try:
            clip = VideoFileClip(str(path))
            duration_sec = clip.duration or -1
        except Exception as exc:
            return MediaResult(
                file_path=path,
                media_type="Video",
                error=f"Could not open video: {exc}",
            )

        # ── Step 1: Audio transcription ───────────────────────────────
        audio_path = Path(tmp_dir) / "audio.wav"
        try:
            if clip.audio is not None:
                clip.audio.write_audiofile(
                    str(audio_path),
                    fps=16000,
                    nbytes=2,
                    codec="pcm_s16le",
                    logger=None,  # suppress moviepy progress bars
                )
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    transcript = _transcribe_audio_file(audio_path, groq_client)
                else:
                    errors.append("Audio extraction produced an empty file.")
            else:
                logger.info("Video has no audio track: %s", path.name)
        except Exception as exc:
            logger.warning("Audio extraction failed: %s", exc)
            errors.append(f"Audio extraction failed: {exc}")

        # ── Step 2: Key frame extraction ──────────────────────────────
        total_dur = int(duration_sec) if duration_sec > 0 else 0
        timestamps = list(range(0, max(1, total_dur), FRAME_INTERVAL))[:MAX_FRAMES]

        for ts in timestamps:
            frame_file = Path(tmp_dir) / f"frame_{ts:06d}.jpg"
            try:
                frame = clip.get_frame(min(ts, clip.duration - 0.1))
                # Write frame as JPEG using Pillow
                from PIL import Image  # type: ignore[import]
                import numpy as np     # type: ignore[import]
                img = Image.fromarray(frame.astype("uint8"))
                img.save(str(frame_file), format="JPEG", quality=85)
                ts_desc = _describe_frame(frame_file, groq_client, ts)
                frame_descriptions.append(ts_desc)
            except ImportError:
                errors.append("Pillow/numpy not installed; frame extraction skipped.")
                break
            except Exception as exc:
                logger.warning("Frame extraction failed at %ds: %s", ts, exc)
                errors.append(f"Frame at {ts}s failed: {exc}")

        clip.close()

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return MediaResult(
        file_path=path,
        media_type="Video",
        transcript=transcript,
        frame_descriptions=frame_descriptions,
        duration_sec=duration_sec,
        error="; ".join(errors) if errors else "",
    )
