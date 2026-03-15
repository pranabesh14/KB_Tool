"""
transcription.py — Video/audio transcription pipeline.

Supports (no system ffmpeg install required):
  • Video : .mp4, .mkv, .avi, .mov, .webm, .m4v, .flv, .wmv, .ts, .mpg
  • Audio : .mp3, .wav, .aac, .flac, .ogg, .m4a

Audio extraction uses moviepy (bundles its own ffmpeg via imageio-ffmpeg).
Transcription uses OpenAI Whisper (local, CPU/GPU).

Public API:
    transcribe_video(source: str) -> str   # returns path to cached JSON
    is_supported_video(filename: str) -> bool
"""

import hashlib
import json
import os
import tempfile

import numpy as np
import whisper

from modules.config import DATA_DIR, get_logger

logger = get_logger("transcription")

SUPPORTED_EXTENSIONS = {
    # video
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v",
    ".flv", ".wmv", ".ts", ".m2ts", ".mpg", ".mpeg",
    # audio
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a",
}

# pure-audio formats that can be read directly by Whisper without moviepy
_DIRECT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


# ─── Whisper model (lazy singleton) ───────────────────────────────────────────

_WHISPER_MODEL: whisper.Whisper | None = None


def _get_whisper() -> whisper.Whisper:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    for size in ("base", "tiny"):
        try:
            _WHISPER_MODEL = whisper.load_model(size)
            logger.info("Whisper '%s' model loaded.", size)
            return _WHISPER_MODEL
        except Exception as exc:
            logger.warning("Could not load Whisper '%s': %s", size, exc)
    raise RuntimeError("No Whisper model could be loaded.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cache_key(source: str) -> str:
    return "transcript_" + hashlib.md5(source.encode()).hexdigest()[:12]


def _json_cache_path(source: str) -> str:
    return os.path.join(DATA_DIR, _cache_key(source) + ".json")


def _extract_audio_numpy(source: str) -> np.ndarray:
    """
    Extract audio from any video or audio file and return a float32 numpy
    array at 16 kHz mono — exactly what Whisper expects.

    Strategy:
      1. Pure-audio formats (.wav, .mp3, .flac, .ogg) that Whisper can read
         natively are passed directly — no moviepy needed.
      2. Everything else (video containers + .aac, .m4a) goes through moviepy,
         which uses its bundled imageio-ffmpeg binary (no system install needed).
    """
    ext = os.path.splitext(source)[1].lower()

    # ── Path 1: Whisper reads it directly ─────────────────────────────────────
    if ext in _DIRECT_AUDIO_EXTENSIONS:
        logger.info("Loading audio directly with Whisper: %s", source)
        return whisper.load_audio(source)   # returns float32 @ 16kHz mono

    # ── Path 2: extract via moviepy ───────────────────────────────────────────
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError as exc:
        raise RuntimeError(
            "moviepy is not installed. Run: pip install moviepy"
        ) from exc

    logger.info("Extracting audio via moviepy: %s", source)

    # Decide whether to open as video or audio clip
    if ext in {".mp3", ".aac", ".m4a", ".ogg", ".flac", ".wav"}:
        clip = AudioFileClip(source)
    else:
        video_clip = VideoFileClip(source, audio=True)
        clip = video_clip.audio
        if clip is None:
            raise RuntimeError(f"No audio track found in: {source}")

    # Write to a temporary WAV file then let Whisper load it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        clip.write_audiofile(
            tmp_path,
            fps=16000,
            nbytes=2,        # 16-bit PCM
            ffmpeg_params=["-ac", "1"],  # mono
            logger=None,     # suppress moviepy progress bar
        )
        clip.close()
        audio_np = whisper.load_audio(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return audio_np


# ─── Public API ───────────────────────────────────────────────────────────────

def transcribe_video(source: str) -> str:
    """
    Transcribe a local video or audio file.

    Parameters
    ----------
    source : str
        Absolute or relative path to a video/audio file.

    Returns
    -------
    str
        Path to the cached JSON transcript file.

    JSON schema
    -----------
    {
      "source":   "<original file path>",
      "text":     "<full transcript text>",
      "segments": [{"start": 0.0, "end": 3.4, "text": "…"}, …]
    }
    """
    json_path = _json_cache_path(source)
    if os.path.exists(json_path):
        logger.info("Cache hit: %s → %s", source, json_path)
        return json_path

    logger.info("Transcribing: %s", source)
    audio_np = _extract_audio_numpy(source)
    result   = _get_whisper().transcribe(audio_np)

    segments = [
        {
            "start": float(s.get("start", 0.0)),
            "end":   float(s.get("end",   0.0)),
            "text":  (s.get("text") or "").strip(),
        }
        for s in (result.get("segments") or [])
    ]

    payload = {
        "source":   source,
        "text":     result.get("text", ""),
        "segments": segments,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    logger.info("Saved transcript → %s (%d segments)", json_path, len(segments))
    return json_path


def is_supported_video(filename: str) -> bool:
    """Return True if the file extension is a recognised video/audio format."""
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS or MM:SS string."""
    sec = int(round(seconds))
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"



def seconds_to_hhmmss(seconds: float) -> str:
    """Convert a float second value to HH:MM:SS or MM:SS string."""
    sec = int(round(seconds))
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"