"""
paths.py
--------

Centralized path utilities for the LocalTranscriber application.

This module is responsible for determining where files should be stored
and accessed depending on whether the application is running in:

1. Development mode (normal Python execution)
2. Packaged mode (e.g., PyInstaller executable)

Responsibilities
----------------
- Determine application root directory
- Resolve OS-specific user data directories
- Provide paths for persistent data storage
- Provide paths to bundled assets (binaries and models)

Directory Structure
-------------------
assets/
    bin/            -> bundled executables (ffmpeg, llama.cpp)
    models/
        whisper/    -> bundled Whisper models
        llm/        -> bundled GGUF summarization models

User Data Directory
-------------------
This directory stores persistent user data including:

- settings.json
- SQLite database
- generated artifacts
- job metadata

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
import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# Application name
# ---------------------------------------------------------
# Used when creating OS-specific directories for storing
# user configuration files, database files, and artifacts.
APP_NAME = "LocalTranscriber"

# ---------------------------------------------------------
# Detect if the application is running as a packaged binary
# ---------------------------------------------------------
def is_frozen() -> bool:
    """
    Determine whether the application is running from a bundled
    executable (e.g., created with PyInstaller).

    PyInstaller sets the attribute `sys.frozen = True` when the
    application is running as a packaged executable.

    Returns True if the app is running from a packaged executable,
    otherwise False (normal Python execution).
    """
    return getattr(sys, "frozen", False)

# ---------------------------------------------------------
# Determine the application root directory
# ---------------------------------------------------------
def app_root() -> Path:
    """
    Return the root directory of the application.

    Behavior differs depending on how the app is running:

    Development mode:
        Returns the repository root directory.

    Packaged executable (PyInstaller):
        Returns the temporary extraction directory used by
        PyInstaller (`sys._MEIPASS`).
    """

    # When running from a packaged executable,
    # PyInstaller extracts bundled assets to a temporary folder
    # Accessible via sys._MEIPASS
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]

    # During development, assume the repo structure:
    # repo_root/app/paths.py
    # So we move two directories upward from this file
    return Path(__file__).resolve().parents[1]

# ---------------------------------------------------------
# User data directory
# ---------------------------------------------------------
def user_data_dir() -> Path:
    """
    Return the directory where user-specific application data
    should be stored.

    This location depends on the operating system:

    Windows:
        %APPDATA%\\LocalTranscriber

    macOS:
        ~/Library/Application Support/LocalTranscriber

    Linux / Unix:
        ~/.localtranscriber

    The directory is created automatically if it does not exist.
    """

    # Windows
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA", str(Path.home()))
        p = Path(base) / APP_NAME

    # macOS
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / APP_NAME

    # Linux / other Unix systems
    else:
        p = Path.home() / f".{APP_NAME.lower()}"

    # Ensure the directory exists
    p.mkdir(parents=True, exist_ok=True)

    return p

# ---------------------------------------------------------
# Jobs directory
# ---------------------------------------------------------
def jobs_dir() -> Path:
    """
    Directory used to store job metadata.

    Each transcription job can store metadata such as:
    - job ID
    - timestamps
    - processing state
    - related artifacts
    """

    p = user_data_dir() / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------------------------------------
# Artifacts directory
# ---------------------------------------------------------
def artifacts_dir() -> Path:
    """
    Directory used to store processing outputs generated
    by transcription jobs.

    Artifacts may include:
    - converted audio files
    - transcript JSON files
    - generated summaries
    """

    p = user_data_dir() / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------------------------------------
# Bundled binaries directory
# ---------------------------------------------------------
def bundled_bin_dir() -> Path:
    """
    Directory containing bundled executables shipped
    with the application.

    Examples:
    - ffmpeg
    - llama.cpp binaries

    Structure should look like:
        assets/bin/
    """

    return app_root() / "assets" / "bin"

# ---------------------------------------------------------
# Bundled models directory
# ---------------------------------------------------------
def bundled_models_dir() -> Path:
    """
    Base directory containing bundled machine learning models.

    Structure should look like:
        assets/models/
    """

    return app_root() / "assets" / "models"

# ---------------------------------------------------------
# Bundled Whisper model directory
# ---------------------------------------------------------
def bundled_whisper_small_dir() -> Path:
    """
    Directory containing the bundled Whisper "small" model.

    The model is stored in CTranslate2 format for use with
    faster-whisper.

    Structure should look like:
        assets/models/whisper/small/
    """

    return bundled_models_dir() / "whisper" / "small"

# ---------------------------------------------------------
# Bundled LLM directory
# ---------------------------------------------------------
def bundled_llm_dir() -> Path:
    """
    Directory containing bundled GGUF language models used
    for summarization.

    Structure should look like:
        assets/models/llm/
    """

    return bundled_models_dir() / "llm"