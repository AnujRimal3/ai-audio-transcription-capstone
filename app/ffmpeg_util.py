"""
ffmpeg_util.py
--------------

Utility functions for working with FFmpeg.

This module is responsible for locating a usable FFmpeg
binary and converting arbitrary audio/video files into
a format compatible with Whisper transcription.

Responsibilities
----------------
- Locate FFmpeg executable
- Prefer user-configured path if available
- Fallback to bundled binaries
- Fallback to system PATH
- Convert media to 16kHz mono WAV format

Why Conversion Is Needed
------------------------
Whisper models perform best when input audio is:

- mono
- 16kHz sample rate
- WAV container

This conversion ensures consistent transcription quality
regardless of the original media format.

Course: C486 - Senior Capstone Computer Science
Team 6

Team Members:
- Anuj Rimal
- Hoyda Al Yahiri
- Zack Ganser
- Saahil Patel

Project:
AI Audio Transcription & Summarization System
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from app.paths import bundled_bin_dir

# ---------------------------------------------------------
# Locate the FFmpeg executable used for audio conversion
# ---------------------------------------------------------
# Search priority:
# 1) Explicit path provided by the user in settings
# 2) Bundled copy inside assets/bin
# 3) System PATH
def find_ffmpeg(explicit_path: str = "") -> str:
    """
    Locate a working FFmpeg executable.

    FFmpeg is required to convert input audio/video files into
    a format that Whisper can transcribe reliably.
    """

    # ---------------------------------------------------------
    # 1. Use user-configured path if provided
    # ---------------------------------------------------------
    # If the user explicitly set a path in Settings, return it
    # immediately and assume they know what they are doing.
    if explicit_path:
        return explicit_path

    # ---------------------------------------------------------
    # 2. Check bundled FFmpeg binaries
    # ---------------------------------------------------------
    # The application may ship with its own FFmpeg binary stored
    # inside assets/bin.
    bin_dir = bundled_bin_dir()

    # Candidate list of FFmpeg locations to test
    candidates = []

    # Unix/macOS style binary
    if (bin_dir / "ffmpeg").exists():
        candidates.append(str(bin_dir / "ffmpeg"))

    # Windows binary
    if (bin_dir / "ffmpeg.exe").exists():
        candidates.append(str(bin_dir / "ffmpeg.exe"))

    # ---------------------------------------------------------
    # 3. Fall back to system PATH
    # ---------------------------------------------------------
    # If FFmpeg is installed globally, it should be callable simply
    # as "ffmpeg".
    candidates.append("ffmpeg")

    # ---------------------------------------------------------
    # Validate candidates
    # ---------------------------------------------------------
    # We run `ffmpeg -version` to verify that the executable exists
    # and launches successfully. -Zack
    for c in candidates:
        try:
            subprocess.run([c, "-version"], capture_output=True, check=True, text=True)
            return c
        except Exception:
            # Ignore failures and try the next candidate
            continue

    # Raise an error if no suitable FFMpeg candidates were found
    raise RuntimeError("FFmpeg not found. Put it in assets/bin or set FFmpeg path in Settings.")

# ---------------------------------------------------------
# Convert input media into Whisper-compatible audio
# ---------------------------------------------------------
# Whisper performs best with:
# - mono audio
# - 16kHz sample rate
# - WAV container
def convert_to_wav_16k_mono(ffmpeg_path: str, input_path: Path, output_wav: Path) -> None:
    """
    Convert an input audio/video file into a 16kHz mono WAV file.

    Whisper models expect audio roughly in this format for best
    performance. Converting beforehand ensures consistent input.
    """

    # Ensure the output directory exists before running FFmpeg
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Construct FFmpeg command
    cmd = [
        ffmpeg_path,
        "-y",                       # overwrite output file if it exists
        "-i", str(input_path),      # input media file
        "-ac", "1",                 # convert to mono
        "-ar", "16000",             # resample to 16kHz
        "-vn",                      # disable video
        str(output_wav),            # output file
    ]

    # Run FFmpeg conversion process
    p = subprocess.run(cmd, capture_output=True, text=True)

    # Raise an error if conversion failed
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{p.stderr.strip()}")