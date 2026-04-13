"""
settings.py
-----------

Application configuration management for LocalTranscriber.

This module defines the AppSettings dataclass and provides
functions for loading and saving user configuration.

Responsibilities
----------------
- Define all configurable application settings
- Load settings from settings.json
- Persist settings changes
- Provide sensible defaults
- Auto-detect bundled models when available

Settings Categories
-------------------
Transcription
    Whisper model selection
    device configuration
    compute precision

External Binaries
    ffmpeg executable path
    llama.cpp binary path

Model Paths
    Whisper model directory
    GGUF summarization model

LLM Runtime
    context size
    CPU thread configuration

Settings are stored in the user's application data directory.

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
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from app.paths import user_data_dir, bundled_bin_dir, bundled_whisper_small_dir, bundled_llm_dir

# ---------------------------------------------------------
# Settings file location
# ---------------------------------------------------------
# The application stores persistent configuration in a JSON file
# located in the user's application data directory.
SETTINGS_FILE = user_data_dir() / "settings.json"

# ---------------------------------------------------------
# Application settings structure
# ---------------------------------------------------------
# This dataclass represents ALL configurable settings used
# throughout the application.
@dataclass
class AppSettings:
    # ----------------------------
    # Transcription configuration
    # ----------------------------

    # Default Whisper model name
    # This is primarily used by the UI; if a bundled model path
    # Exists the application will prefer that instead
    whisper_model: str = "small"

    # Language override for transcription
    # Empty string means Whisper will auto-detect language
    language: str = ""

    # Device used for transcription inference
    # Possible values: auto / cpu / cuda
    device: str = "cpu"

    # Compute precision used during inference
    # Possible values: auto / int8 (good for CPU inference) / float16 (good for GPU inference) / float32
    compute_type: str = "int8"

    # ----------------------------
    # External binary overrides
    # ----------------------------

    # Optional path to FFmpeg executable
    # If blank, the app will try:
    #   1. bundled FFmpeg
    #   2. system PATH
    ffmpeg_path: str = ""

    # Optional path to llama.cpp executable
    # If blank, the app will try:
    #   1. bundled binaries
    #   2. system PATH
    llama_path: str = ""

    # ----------------------------
    # Model path overrides
    # ----------------------------

    # Path to a faster-whisper CTranslate2 model directory
    # If empty, the app may fall back to a bundled model or
    # A HuggingFace model name
    whisper_model_path: str = ""

    # Path to a GGUF LLM model file used for summarization
    llama_model_path: str = ""

    # ----------------------------
    # LLM runtime configuration
    # ----------------------------

    # Context window size for llama.cpp
    llama_ctx: int = 4096

    # Number of CPU threads used for inference
    # 0 means "let llama.cpp decide automatically"
    llama_threads: int = 0

    llama_use_gpu: bool = False
    llama_gpu_layers: int = 999

    # Diarization
    enable_diarization: bool = False
    diarization_python_path: str = ""
    hf_token: str = ""

# ---------------------------------------------------------
# Create default settings
# ---------------------------------------------------------
def default_settings() -> AppSettings:
    """
    Generate default application settings.

    This function also attempts to auto-detect bundled models
    so the application can run without requiring manual setup.
    """

    # Start with default values from the dataclass
    s = AppSettings()

    # ---------------------------------------------------------
    # Prefer bundled Whisper model if available
    # ---------------------------------------------------------
    # If a bundled faster-whisper model directory exists,
    # automatically configure the app to use it.
    if bundled_whisper_small_dir().exists():
        s.whisper_model_path = str(bundled_whisper_small_dir())

    # ---------------------------------------------------------
    # Prefer bundled GGUF model if exactly one exists
    # ---------------------------------------------------------
    # If the application ships with a single LLM model inside
    # assets/models/llm, automatically configure it.
    llm_dir = bundled_llm_dir()

    if llm_dir.exists():
        # Look for GGUF models
        ggufs = list(llm_dir.glob("*.gguf"))

        # If exactly one model exists, auto-select it
        if len(ggufs) == 1:
            s.llama_model_path = str(ggufs[0])

    return s

# ---------------------------------------------------------
# Load settings from disk
# ---------------------------------------------------------
def load_settings() -> AppSettings:
    """
    Load application settings from the settings JSON file.

    If the file does not exist yet:
    1. Default settings are created
    2. They are written to disk
    3. The defaults are returned
    """

    # If settings file does not exist, create defaults
    if not SETTINGS_FILE.exists():
        s = default_settings()

        # Save the default configuration
        save_settings(s)
        return s

    # Read JSON settings file
    data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))

    # Start with defaults to ensure missing fields are filled
    s = default_settings()

    # Apply values stored in the JSON file
    for k, v in data.items():

        # Only set attributes that actually exist on the dataclass
        # (protects against outdated or corrupted settings files -Zack)
        if hasattr(s, k):
            setattr(s, k, v)

    return s

# ---------------------------------------------------------
# Save settings to disk
# ---------------------------------------------------------
def save_settings(s: AppSettings) -> None:
    """
    Persist application settings to the JSON settings file.

    The dataclass is converted to a dictionary using `asdict()`
    before being serialized to JSON.
    """

    SETTINGS_FILE.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")