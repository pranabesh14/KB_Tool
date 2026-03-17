“””
transcription.py — Video/audio transcription pipeline.

## Two input modes

1. Local file  (.mp4, .mkv, .avi, .mov, .webm, .m4v, .flv, .wmv, .ts, .mpg,
   .mp3, .wav, .aac, .flac, .ogg, .m4a)
   → moviepy extracts audio (bundled ffmpeg, no system install needed)
   → Whisper transcribes locally
1. Platform URL  (Dailymotion, Vimeo, and 1000+ sites supported by yt-dlp)
   → yt-dlp downloads audio-only stream to a temp file
   → SSL errors handled with automatic retry using –no-check-certificate
   → Whisper transcribes locally

Whisper runs 100% locally on your machine (openai-whisper package).
Model weights download once to ~/.cache/whisper/ on first use (~74 MB for base).
No OpenAI API key needed — “openai-whisper” is the open-source local model.

## Public API

transcribe(source: str) -> str
Accepts a local file path OR a platform URL.
Returns path to the cached JSON transcript.

transcribe_video(source: str) -> str
Alias for transcribe(). Kept for backward compatibility.

is_supported_video(filename: str) -> bool
Returns True if the file extension is recognised.

seconds_to_hhmmss(seconds: float) -> str
Utility: convert float seconds to HH:MM:SS / MM:SS string.
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import hashlib
import json
import os
import tempfile
from typing import List, Optional

import numpy as np
import whisper

from modules.config import DATA_DIR, WHISPER_MODEL_SIZE, get_logger

logger = get_logger(“transcription”)

# ─── Supported local file extensions ──────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
# video containers
“.mp4”, “.mkv”, “.avi”, “.mov”, “.webm”, “.m4v”,
“.flv”, “.wmv”, “.ts”, “.m2ts”, “.mpg”, “.mpeg”,
# audio
“.mp3”, “.wav”, “.aac”, “.flac”, “.ogg”, “.m4a”,
}

# formats Whisper can load directly (no moviepy needed)

_DIRECT_AUDIO_EXTENSIONS = {”.wav”, “.mp3”, “.flac”, “.ogg”}

# ─── Whisper model (lazy singleton) ───────────────────────────────────────────

_WHISPER_MODEL: Optional[whisper.Whisper] = None

def _build_fallback_order(preferred: str) -> List[str]:
“”“Return fallback sizes from preferred down to tiny.”””
order = [“large”, “medium”, “small”, “base”, “tiny”]
try:
idx = order.index(preferred.lower())
return order[idx:]
except ValueError:
return [“base”, “tiny”]

def _get_whisper() -> whisper.Whisper:
“””
Load and cache the Whisper model (lazy singleton).

```
- Model size is set by WHISPER_MODEL_SIZE in .env (default: base).
- Weights download once to ~/.cache/whisper/ on first use.
- Subsequent calls reuse cached weights — no internet needed after first run.
- Falls back to smaller sizes if the requested one fails to load.
"""
global _WHISPER_MODEL
if _WHISPER_MODEL is not None:
    return _WHISPER_MODEL

for size in _build_fallback_order(WHISPER_MODEL_SIZE):
    try:
        logger.info("Loading Whisper '%s' model (local, no API call) …", size)
        _WHISPER_MODEL = whisper.load_model(size)
        logger.info("Whisper '%s' ready.", size)
        return _WHISPER_MODEL
    except Exception as exc:
        logger.warning("Could not load Whisper '%s': %s", size, exc)

raise RuntimeError(
    f"Could not load any Whisper model (tried: {_build_fallback_order(WHISPER_MODEL_SIZE)}). "
    "Ensure openai-whisper is installed and you have internet access for the "
    "first-time model weight download."
)
```

# ─── Source type detection ─────────────────────────────────────────────────────

def _is_url(source: str) -> bool:
return source.startswith(“http://”) or source.startswith(“https://”)

# ─── Cache helpers ─────────────────────────────────────────────────────────────

def *cache_key(source: str) -> str:
return “transcript*” + hashlib.md5(source.encode()).hexdigest()[:12]

def _json_cache_path(source: str) -> str:
return os.path.join(DATA_DIR, _cache_key(source) + “.json”)

# ─── Shared audio → numpy converter (uses moviepy bundled ffmpeg) ─────────────

def _convert_to_numpy(file_path: str) -> np.ndarray:
“””
Convert any video/audio file to a float32 numpy array at 16kHz mono.
Uses moviepy (bundled ffmpeg) — no system ffmpeg install needed.

```
Windows-safe: all file handles are fully closed before the temp WAV
is read by Whisper and deleted, avoiding WinError32 file-lock errors.

Pure-audio formats (.wav .mp3 .flac .ogg) skip moviepy entirely —
Whisper loads them directly for speed.
"""
ext = os.path.splitext(file_path)[1].lower()

# ── Fast path — Whisper reads these natively ───────────────────────────────
if ext in _DIRECT_AUDIO_EXTENSIONS:
    logger.info("Loading audio directly via Whisper: %s", os.path.basename(file_path))
    return whisper.load_audio(file_path)

# ── moviepy path ───────────────────────────────────────────────────────────
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError as exc:
    raise RuntimeError("moviepy is not installed. Run: pip install moviepy") from exc

logger.info("Extracting audio via moviepy: %s", os.path.basename(file_path))

# Create temp file path WITHOUT opening it
# (NamedTemporaryFile with delete=False still holds a handle on Windows)
tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
os.close(tmp_fd)  # close the OS file descriptor immediately

video_clip = None
clip       = None

try:
    # ── Open clip ─────────────────────────────────────────────────────────
    if ext in {".aac", ".m4a", ".mp3"}:
        clip = AudioFileClip(file_path)
    else:
        video_clip = VideoFileClip(file_path, audio=True)
        clip = video_clip.audio
        if clip is None:
            raise RuntimeError(f"No audio track found in: {file_path}")

    # ── Write WAV ─────────────────────────────────────────────────────────
    clip.write_audiofile(
        tmp_path,
        fps=16000,
        nbytes=2,                       # 16-bit PCM
        ffmpeg_params=["-ac", "1"],     # mono
        logger=None,                    # suppress moviepy progress bar
    )

finally:
    # ── Close ALL handles before touching the file ─────────────────────────
    # Order matters: audio clip first, then video clip
    try:
        if clip is not None:
            clip.close()
    except Exception:
        pass
    try:
        if video_clip is not None:
            video_clip.close()
    except Exception:
        pass

# ── Read WAV into numpy AFTER all handles are closed ──────────────────────
# On Windows this is critical — file must not be open anywhere
try:
    audio_np = whisper.load_audio(tmp_path)
finally:
    # ── Delete temp file ───────────────────────────────────────────────────
    try:
        os.remove(tmp_path)
    except OSError as exc:
        # Log but don't raise — the numpy array is already loaded
        logger.warning("Could not delete temp WAV %s: %s", tmp_path, exc)

return audio_np
```

# ─── Audio extraction: local files ────────────────────────────────────────────

def _extract_audio_from_file(source: str) -> np.ndarray:
“”“Extract audio from a local video/audio file.”””
return _convert_to_numpy(source)

# ─── Audio extraction: platform URLs via yt-dlp ───────────────────────────────

def _ydl_download(url: str, output_template: str, verify_ssl: bool) -> Optional[str]:
“””
Run yt-dlp to download the best audio stream as-is.

```
NO postprocessors are used — yt-dlp just saves the raw stream
(typically .webm, .m4a, or .mp4).  moviepy (bundled ffmpeg) handles
the conversion to WAV later, so no system ffmpeg binary is needed.

Returns the downloaded file path on success, None on SSL/network error.
Raises RuntimeError for non-SSL failures.
"""
import yt_dlp

ydl_opts = {
    "format":             "bestaudio/best",
    "outtmpl":            output_template,
    "quiet":              True,
    "no_warnings":        True,
    "noplaylist":         True,
    "nocheckcertificate": not verify_ssl,
    # ── Suppress ALL side files (no metadata.json, thumbnails, subs etc.) ──
    "writethumbnail":         False,
    "writeinfojson":          False,
    "writedescription":       False,
    "writesubtitles":         False,
    "writeautomaticsub":      False,
    "write_all_thumbnails":   False,
    "writeannotations":       False,
    "no_write_playlist_metafiles": True,
    "skip_download":          False,
    # ── Disable network features that can fail on restricted networks ───────
    "geo_bypass":             True,    # bypass geo-restriction where possible
    "retries":                3,       # retry on transient network errors
    "fragment_retries":       3,
    "extractor_retries":      3,
    "socket_timeout":         30,      # don't hang indefinitely
    # ── NO postprocessors — avoids requiring system ffmpeg binary ───────────
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
except Exception as exc:
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in ("ssl", "certificate", "handshake", "tls")):
        logger.warning("SSL error during yt-dlp download: %s", exc)
        return None
    raise RuntimeError(f"yt-dlp download failed: {exc}") from exc

# Locate the downloaded file
parent = os.path.dirname(output_template)
files  = [
    os.path.join(parent, f)
    for f in os.listdir(parent)
    if os.path.isfile(os.path.join(parent, f))
]
return files[0] if files else None
```

def _extract_audio_from_url(url: str) -> np.ndarray:
“””
Download audio from a platform URL (Dailymotion, YouTube, Vimeo, etc.)
using yt-dlp, then extract audio via moviepy (no system ffmpeg needed).

```
Flow:
  1. yt-dlp downloads the raw audio stream (no postprocessing)
     → file is typically .webm, .m4a, or .mp4
  2. moviepy (bundled imageio-ffmpeg) converts it to 16kHz mono WAV
  3. Whisper loads the WAV for transcription

SSL retry strategy:
  Attempt 1 — normal SSL verification.
  Attempt 2 — SSL verification disabled (corporate proxy / SSL inspection).
"""
try:
    import yt_dlp  # noqa: F401
except ImportError as exc:
    raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from exc

with tempfile.TemporaryDirectory() as tmp_dir:
    output_template = os.path.join(tmp_dir, "audio.%(ext)s")

    # ── Attempt 1: normal SSL ──────────────────────────────────────────────
    logger.info("Downloading audio stream from: %s", url)
    audio_path = _ydl_download(url, output_template, verify_ssl=True)

    # ── Attempt 2: retry without SSL verification ──────────────────────────
    if audio_path is None:
        logger.warning(
            "SSL error on first attempt. Retrying with SSL verification "
            "disabled (likely corporate proxy / SSL inspection)."
        )
        audio_path = _ydl_download(url, output_template, verify_ssl=False)

    if not audio_path or not os.path.exists(audio_path):
        raise RuntimeError(
            f"yt-dlp could not download audio from: {url}\n"
            "Possible causes:\n"
            "  • URL is not supported by yt-dlp\n"
            "  • Video is private or geo-restricted\n"
            "  • Network / firewall is blocking the download\n"
            "Tip: run  yt-dlp --list-formats <url>  to diagnose."
        )

    logger.info(
        "Downloaded: %s (%.1f MB) — converting via moviepy …",
        os.path.basename(audio_path),
        os.path.getsize(audio_path) / (1024 * 1024),
    )

    # ── Convert via moviepy (bundled ffmpeg — no system install needed) ────
    # Must happen INSIDE the with-block before tmp_dir is deleted
    return _convert_to_numpy(audio_path)
```

# ─── Public API ───────────────────────────────────────────────────────────────

def transcribe(source: str) -> str:
“””
Transcribe a local video/audio file OR a platform URL.

```
Parameters
----------
source : str
    • Local file path  : /path/to/recording.mp4
    • Platform URL     : https://www.dailymotion.com/video/...
                         https://vimeo.com/...
                         (any site supported by yt-dlp)

Returns
-------
str
    Path to the cached JSON transcript file.

JSON schema
-----------
{
  "source":   "<original path or URL>",
  "text":     "<full transcript>",
  "segments": [{"start": 0.0, "end": 3.4, "text": "…"}, …]
}
"""
json_path = _json_cache_path(source)
if os.path.exists(json_path):
    logger.info("Cache hit — skipping transcription: %s", source[:80])
    return json_path

logger.info("Starting transcription: %s", source[:80])

audio_np = (
    _extract_audio_from_url(source)
    if _is_url(source)
    else _extract_audio_from_file(source)
)

logger.info("Running Whisper transcription locally …")
result = _get_whisper().transcribe(audio_np)

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

logger.info(
    "Transcript saved → %s  (%d segments, ~%d words)",
    os.path.basename(json_path),
    len(segments),
    len(result.get("text", "").split()),
)
return json_path
```

def transcribe_video(source: str) -> str:
“”“Alias for transcribe(). Kept for backward compatibility.”””
return transcribe(source)

def is_supported_video(filename: str) -> bool:
“”“Return True if the file extension is a recognised video/audio format.”””
return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS

def seconds_to_hhmmss(seconds: float) -> str:
“”“Convert float seconds to HH:MM:SS or MM:SS string.”””
sec = int(round(seconds))
h, rem = divmod(sec, 3600)
m, s   = divmod(rem, 60)
return f”{h:02d}:{m:02d}:{s:02d}” if h else f”{m:02d}:{s:02d}”