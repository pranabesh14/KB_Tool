“””
transcription.py - Video/audio transcription pipeline.

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
No OpenAI API key needed - “openai-whisper” is the open-source local model.

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
import uuid
from typing import List, Optional

import numpy as np
import whisper

from modules.config import DATA_DIR, TEMP_DIR, WHISPER_MODEL_SIZE, get_logger

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
- Subsequent calls reuse cached weights - no internet needed after first run.
- Falls back to smaller sizes if the requested one fails to load.
"""
global _WHISPER_MODEL
if _WHISPER_MODEL is not None:
    return _WHISPER_MODEL

for size in _build_fallback_order(WHISPER_MODEL_SIZE):
    try:
        logger.info("Loading Whisper '%s' model (local, no API call) ...", size)
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
Uses moviepy (bundled ffmpeg) - no system ffmpeg install needed.

```
Temp files are written to the project TEMP_DIR (not the system temp
directory) using a unique UUID filename to avoid collisions and
Windows path/permission issues.

All moviepy handles are fully closed before reading or deleting the
temp WAV - prevents WinError 32 (file in use) and WinError 2 (file
not found due to mixed slashes or premature cleanup).
"""
ext = os.path.splitext(file_path)[1].lower()

# ── Fast path - Whisper reads these natively, no moviepy needed ───────────
if ext in _DIRECT_AUDIO_EXTENSIONS:
    logger.info("Loading audio directly via Whisper: %s", os.path.basename(file_path))
    return whisper.load_audio(file_path)

# ── moviepy path ───────────────────────────────────────────────────────────
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError as exc:
    raise RuntimeError("moviepy is not installed. Run: pip install moviepy") from exc

logger.info("Extracting audio via moviepy: %s", os.path.basename(file_path))

# Use project TEMP_DIR with a UUID name - avoids system temp permission
# issues and mixed-slash path problems on Windows
tmp_path = os.path.join(TEMP_DIR, f"audio_{uuid.uuid4().hex}.wav")
# Normalise to OS path separators (critical on Windows)
tmp_path = os.path.normpath(tmp_path)

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

    # ── Write WAV to project temp dir ─────────────────────────────────────
    clip.write_audiofile(
        tmp_path,
        fps=16000,
        nbytes=2,                       # 16-bit PCM
        ffmpeg_params=["-ac", "1"],     # mono
        logger=None,                    # suppress moviepy progress bar
    )

finally:
    # ── Close ALL handles before reading/deleting ─────────────────────────
    # Clip first, then the parent video clip
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

# ── Read WAV into numpy AFTER all handles are confirmed closed ─────────────
if not os.path.isfile(tmp_path):
    raise RuntimeError(
        f"Temp WAV was not created at {tmp_path}. "
        "moviepy/ffmpeg may have failed silently - check kb_tool.log."
    )

try:
    logger.debug("Loading temp WAV into numpy: %s", tmp_path)
    audio_np = whisper.load_audio(tmp_path)
finally:
    # ── Delete temp WAV - log warning if it fails but don't raise ─────────
    try:
        os.remove(tmp_path)
        logger.debug("Deleted temp WAV: %s", tmp_path)
    except OSError as exc:
        logger.warning("Could not delete temp WAV %s: %s", tmp_path, exc)

return audio_np
```

# ─── Audio extraction: local files ────────────────────────────────────────────

def _extract_audio_from_file(source: str) -> np.ndarray:
“”“Extract audio from a local video/audio file.”””
return _convert_to_numpy(source)

# ─── Audio extraction: platform URLs via yt-dlp ───────────────────────────────

def _get_bundled_ffmpeg() -> Optional[str]:
“””
Return the path to the ffmpeg binary bundled with imageio-ffmpeg (installed
via pip as a moviepy dependency). No system install or admin rights needed.
Returns None if imageio-ffmpeg is not available.
“””
try:
import imageio_ffmpeg
path = imageio_ffmpeg.get_ffmpeg_exe()
if os.path.isfile(path):
logger.info(“Using bundled ffmpeg from imageio-ffmpeg: %s”, path)
return path
except Exception as exc:
logger.warning(“Could not locate bundled ffmpeg: %s”, exc)
return None

def _ydl_download(url: str, output_template: str, verify_ssl: bool) -> Optional[str]:
“””
Download and extract audio from a platform URL using yt-dlp.

```
Uses the ffmpeg binary bundled with imageio-ffmpeg (installed via pip
as a moviepy dependency) - no system ffmpeg install or admin rights needed.

With the bundled ffmpeg available, yt-dlp can:
  • Handle MPEG-TS / fragmented MP4 containers (Dailymotion, YouTube etc.)
  • Remux to clean m4a/mp3 via the FFmpegExtractAudio postprocessor
  • Select the best available audio quality

Returns downloaded file path on success, None on SSL error.
Raises RuntimeError for all other failures.
"""
import yt_dlp

ffmpeg_path = _get_bundled_ffmpeg()

if ffmpeg_path:
    # ── Full quality mode - bundled ffmpeg available ───────────────────────
    # Use FFmpegExtractAudio postprocessor to get clean m4a output
    # yt-dlp will use our bundled ffmpeg, not the system one
    ydl_opts = {
        "format":             "bestaudio/best",
        "outtmpl":            output_template,
        "quiet":              True,
        "no_warnings":        True,
        "noplaylist":         True,
        "nocheckcertificate": not verify_ssl,
        "ffmpeg_location":    os.path.dirname(ffmpeg_path),  # dir containing ffmpeg binary
        "postprocessors": [{
            "key":            "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "128",
        }],
        # ── Suppress side files ────────────────────────────────────────────
        "writethumbnail":              False,
        "writeinfojson":               False,
        "writedescription":            False,
        "writesubtitles":              False,
        "writeautomaticsub":           False,
        "write_all_thumbnails":        False,
        "writeannotations":            False,
        "no_write_playlist_metafiles": True,
        # ── Network reliability ────────────────────────────────────────────
        "geo_bypass":        True,
        "retries":           3,
        "fragment_retries":  3,
        "extractor_retries": 3,
        "socket_timeout":    30,
    }
else:
    # ── Fallback mode - no ffmpeg at all ──────────────────────────────────
    # Request only formats that don't need remuxing
    logger.warning(
        "imageio-ffmpeg not found. Falling back to no-remux format selection. "
        "Some platforms (Dailymotion, YouTube) may not work. "
        "Install imageio-ffmpeg: pip install imageio-ffmpeg"
    )
    ydl_opts = {
        "format": (
            "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio[ext=ogg]"
            "/bestaudio[ext=wav]/bestaudio"
        ),
        "outtmpl":            output_template,
        "quiet":              True,
        "no_warnings":        True,
        "noplaylist":         True,
        "nocheckcertificate": not verify_ssl,
        "writethumbnail":              False,
        "writeinfojson":               False,
        "writedescription":            False,
        "writesubtitles":              False,
        "writeautomaticsub":           False,
        "write_all_thumbnails":        False,
        "writeannotations":            False,
        "no_write_playlist_metafiles": True,
        "geo_bypass":        True,
        "retries":           3,
        "fragment_retries":  3,
        "extractor_retries": 3,
        "socket_timeout":    30,
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

# Locate the downloaded file (extension may differ after postprocessing)
parent = os.path.dirname(output_template)
files  = [
    os.path.join(parent, f)
    for f in os.listdir(parent)
    if os.path.isfile(os.path.join(parent, f))
]
if not files:
    logger.warning("yt-dlp ran but no file found in: %s", parent)
    return None

downloaded = os.path.normpath(files[0])
logger.info("Downloaded: %s", os.path.basename(downloaded))
return downloaded
```

def _extract_audio_from_url(url: str) -> np.ndarray:
“””
Download audio from a platform URL (Dailymotion, YouTube, Vimeo, etc.)
using yt-dlp, then extract audio via moviepy (no system ffmpeg needed).

```
Downloads to project TEMP_DIR (not system temp) for predictable paths
and to avoid Windows permission issues with system temp directories.

SSL retry strategy:
  Attempt 1 - normal SSL verification.
  Attempt 2 - SSL verification disabled (corporate proxy / SSL inspection).
"""
try:
    import yt_dlp  # noqa: F401
except ImportError as exc:
    raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from exc

import shutil

# Use a unique subfolder inside project TEMP_DIR for this download
download_dir    = os.path.normpath(os.path.join(TEMP_DIR, f"dl_{uuid.uuid4().hex}"))
os.makedirs(download_dir, exist_ok=True)
output_template = os.path.join(download_dir, "audio.%(ext)s")

try:
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

    if not audio_path or not os.path.isfile(audio_path):
        raise RuntimeError(
            f"yt-dlp could not download audio from: {url}\n"
            "Possible causes:\n"
            "  • URL is not supported by yt-dlp\n"
            "  • Video is private or geo-restricted\n"
            "  • Network / firewall is blocking the download\n"
            "Tip: run  yt-dlp --list-formats <url>  to diagnose."
        )

    audio_path = os.path.normpath(audio_path)
    logger.info(
        "Downloaded: %s (%.1f MB) - converting via moviepy ...",
        os.path.basename(audio_path),
        os.path.getsize(audio_path) / (1024 * 1024),
    )

    # ── Convert via moviepy (bundled ffmpeg) ───────────────────────────────
    return _convert_to_numpy(audio_path)

finally:
    # ── Clean up the download subfolder entirely ───────────────────────────
    try:
        shutil.rmtree(download_dir, ignore_errors=True)
        logger.debug("Cleaned up download dir: %s", download_dir)
    except Exception as exc:
        logger.warning("Could not clean up download dir %s: %s", download_dir, exc)
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
  "segments": [{"start": 0.0, "end": 3.4, "text": "..."}, ...]
}
"""
json_path = _json_cache_path(source)
if os.path.exists(json_path):
    logger.info("Cache hit - skipping transcription: %s", source[:80])
    return json_path

logger.info("Starting transcription: %s", source[:80])

audio_np = (
    _extract_audio_from_url(source)
    if _is_url(source)
    else _extract_audio_from_file(source)
)

logger.info("Running Whisper transcription locally ...")
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