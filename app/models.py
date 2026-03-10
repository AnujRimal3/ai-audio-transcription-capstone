"""
models.py
---------

Core data structures used throughout LocalTranscriber.

This module defines simple dataclasses that represent
transcription results and transcript segments.

Responsibilities
----------------
- Define structured transcription data models
- Provide a consistent interface between modules
- Simplify JSON serialization of transcript results

Primary Structures
------------------
TranscriptSegment
    Represents a single spoken segment with timestamps.

TranscriptResult
    Represents the full transcription result including
    detected language and ordered segment list.

These models are used by:
    whisper_util.py
    workers.py
    ui_main.py
    storage/export logic

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
from dataclasses import dataclass

# ---------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------
# Represents a single timestamped portion of a Whisper transcription.
#
# Whisper divides audio into multiple segments during transcription.
# Each segment corresponds to a small window of speech detected in
# the audio stream.
#
# Example segment:
#   start = 12.34
#   end   = 15.87
#   text  = "Hello everyone and welcome back."
#
# This structure is used throughout the application to store and
# transport transcription data in a consistent format. -Zack
@dataclass
class TranscriptSegment:
    # Start timestamp of the spoken segment (seconds)
    start: float

    # End timestamp of the spoken segment (seconds)
    end: float

    # Transcribed text for the segment
    text: str

# ---------------------------------------------------------
# TranscriptResult
# ---------------------------------------------------------
# Represents the full output of a transcription job.
#
# This structure contains:
#   - The detected language of the audio
#   - A list of transcript segments produced by Whisper
#
# Example structure:
#
# TranscriptResult(
#     language="en",
#     segments=[
#         TranscriptSegment(0.00, 3.21, "Hello everyone"),
#         TranscriptSegment(3.21, 6.78, "Welcome to today's meeting")
#     ]
# )
#
# The UI and summarization pipeline both consume this object. -Zack
@dataclass
class TranscriptResult:
    # Detected or user-specified language code (e.g., "en")
    language: str

    # List of timestamped transcript segments
    segments: list[TranscriptSegment]