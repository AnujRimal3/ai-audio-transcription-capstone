"""
AI Audio Transcription & Summarization System
---------------------------------------------

Course: C486 - Senior Capstone Computer Science
Team 6

Team Members:
- Anuj Rimal
- Hoyda Al Yahiri
- Zack Ganser
- Saahil Patel

Project Overview:
This application provides a fully offline audio transcription and
summarization pipeline. Users can upload audio or video files, which are:

1. Converted to 16kHz mono WAV using FFmpeg
2. Transcribed locally using faster-whisper
3. Summarized locally using llama.cpp with a GGUF language model
4. Stored and displayed through a PySide6 desktop interface

The system is designed to:
- Operate fully offline
- Preserve data privacy
- Provide timestamped transcripts
- Generate structured summaries using chunked map-reduce LLM prompting

Architecture Overview:
- PySide6 GUI (main thread)
- QThread-based background workers
- SQLite job persistence
- Modular utilities for:
    - FFmpeg handling
    - Whisper transcription
    - Llama summarization
    - Model/binary discovery

All heavy computation runs in background threads to maintain UI responsiveness.

Repository Structure:
- app/          → Core application logic
- assets/       → Models and binaries
- scripts/      → Setup and bootstrap scripts
- artifacts/    → Generated transcripts and summaries (note this directory is stored in the users' APPDATA)

---------------------------------------------
Team 6 – AI Audio Transcription Capstone
---------------------------------------------
"""

import sys
from PySide6.QtWidgets import QApplication
from app.ui_main import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()