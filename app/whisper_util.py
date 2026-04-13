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
import re

def cuda_available() -> bool:
    """
    Return True if CTranslate2 reports that CUDA is available.
    """
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        return bool(supported)
    except Exception:
        return False


def resolve_whisper_runtime(device: str, compute_type: str) -> tuple[str, str]:
    """
    Resolve the final device + compute type for faster-whisper.

    Rules:
    - device="cpu"  -> always CPU
    - device="cuda" -> require CUDA or fail
    - device="auto" -> prefer CUDA when available, otherwise CPU

    Compute type:
    - compute_type="auto" -> float16 on CUDA, int8 on CPU
    - otherwise preserve the user's explicit choice
    """
    requested_device = (device or "auto").strip().lower()
    requested_compute = (compute_type or "auto").strip().lower()

    has_cuda = cuda_available()

    if requested_device == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "CUDA was selected in Settings, but no compatible CUDA runtime was detected."
            )
        final_device = "cuda"
    elif requested_device == "cpu":
        final_device = "cpu"
    else:
        final_device = "cuda" if has_cuda else "cpu"

    if requested_compute == "auto":
        final_compute = "float16" if final_device == "cuda" else "int8"
    else:
        final_compute = requested_compute

    return final_device, final_compute

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

def merge_segments_into_sentences(
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """
    Merge Whisper segments into fuller sentence-like units.

    A new sentence is emitted when punctuation (. ! ?) is reached
    or when the speaker changes.

    Timestamps span from the first contributing segment to the last.
    """
    if not segments:
        return []

    merged: list[TranscriptSegment] = []

    current_parts: list[str] = []
    current_start: float | None = None
    current_end: float | None = None
    current_speaker: str | None = None

    sentence_end_re = re.compile(r'[.!?]["\']?$')

    def flush() -> None:
        nonlocal current_parts, current_start, current_end, current_speaker

        if not current_parts:
            return
        if current_start is None or current_end is None:
            return

        text = " ".join(part.strip() for part in current_parts if part.strip()).strip()
        if text:
            merged.append(
                TranscriptSegment(
                    start=current_start,
                    end=current_end,
                    text=text,
                    speaker=current_speaker or "UNKNOWN",
                )
            )

        current_parts = []
        current_start = None
        current_end = None
        current_speaker = None

    for seg in segments:
        seg_text = (seg.text or "").strip()
        if not seg_text:
            continue

        seg_speaker = getattr(seg, "speaker", "UNKNOWN") or "UNKNOWN"

        # If speaker changes, flush current sentence first
        if current_parts and current_speaker != seg_speaker:
            flush()

        if current_start is None:
            current_start = float(seg.start)
            current_speaker = seg_speaker

        current_parts.append(seg_text)
        current_end = float(seg.end)

        # Emit when sentence-ending punctuation is reached
        if sentence_end_re.search(seg_text):
            flush()

    # Flush any remaining partial sentence
    flush()

    return merged