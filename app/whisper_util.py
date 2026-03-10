"""
whisper_util.py
---------------

Wrapper utilities for running faster-whisper transcription.

This module provides a simplified interface for performing
speech-to-text transcription using local Whisper models.

Responsibilities
----------------
- Initialize faster-whisper model
- Run transcription on WAV audio
- Convert results into application data structures
- Normalize segment output

Output
------
Returns a TranscriptResult object containing:

- detected language
- list of timestamped transcript segments

This module acts as the bridge between the faster-whisper
library and the rest of the application.

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
from pathlib import Path
from faster_whisper import WhisperModel
from app.models import TranscriptResult, TranscriptSegment

# ---------------------------------------------------------
# Run Whisper transcription on an audio file
# ---------------------------------------------------------
def transcribe_file(
    wav_path: Path,
    model_name_or_path: str,
    device: str = "auto",
    compute_type: str = "auto",
    language: str = ""
) -> TranscriptResult:
    """
        Transcribe an audio file using faster-whisper.
    """

    # ---------------------------------------------------------
    # Initialize the Whisper model
    # ---------------------------------------------------------
    # This loads the model into memory. If the model reference
    # is a HuggingFace name, faster-whisper may download it
    # and cache it locally the first time it runs. -Zack
    model = WhisperModel(model_name_or_path, device=device, compute_type=compute_type)

    # ---------------------------------------------------------
    # Run transcription
    # ---------------------------------------------------------
    # model.transcribe() returns:
    #   1) an iterator of segments
    #   2) metadata (including detected language)
    #
    # I disable VAD filtering here because:
    # - audio has already been normalized
    # - we want full coverage of the recording
    #
    # beam_size controls decoding search width; higher values
    # can improve accuracy slightly at the cost of speed. -Zack
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language or None,
        vad_filter=False,
        beam_size=5
    )

    # Container for converted transcript segments
    segments: list[TranscriptSegment] = []

    # ---------------------------------------------------------
    # Convert Whisper segments into app data structure
    # ---------------------------------------------------------
    # faster-whisper returns internal segment objects. I converted
    # them into the application's TranscriptSegment dataclass so
    # the rest of the app has a consistent data format. -Zack
    for s in segments_iter:
        segments.append(TranscriptSegment(start=float(s.start), end=float(s.end), text=s.text.strip()))

    # ---------------------------------------------------------
    # Build final result object
    # ---------------------------------------------------------
    # Language priority:
    # 1. detected language from Whisper
    # 2. user-specified language
    # 3. fallback "unknown"
    return TranscriptResult(language=info.language or (language or "unknown"), segments=segments)