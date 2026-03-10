"""
ui_main.py
----------

Main graphical user interface for LocalTranscriber.

This module builds the primary application window using PySide6
and provides controls for running transcription and viewing results.

Responsibilities
----------------
- File selection
- Starting transcription jobs
- Displaying transcript output
- Displaying summary output
- Exporting transcript and summary files
- Managing application settings

UI Layout
---------
Top Bar
    file selection
    run button
    settings

Main Panel
    transcript viewer
    summary viewer

Bottom Panel
    export transcript
    export summary

The UI communicates with the background worker thread using
Qt signals to receive progress updates and results.

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
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit,
    QLineEdit, QComboBox, QDialog, QFormLayout, QMessageBox, QSpinBox
)

from app.settings import load_settings, save_settings, AppSettings
from app.storage import init_db
from app.workers import TranscribeSummarizeWorker


class SettingsDialog(QDialog):
    """
    Settings dialog allowing the user to configure:

    - Whisper transcription model and device
    - Optional explicit model paths
    - External binary locations (FFmpeg / llama.cpp)
    - LLM runtime configuration (context size, threads)

    Changes are saved to settings.json when the user clicks Save.
    """

    def __init__(self, parent: QWidget, settings: AppSettings):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        # Reference to the current settings object
        self.settings = settings

        # Main vertical layout
        layout = QVBoxLayout(self)

        # Form layout organizes label / widget pairs
        form = QFormLayout()

        # ----------------------------
        # Whisper transcription settings
        # ----------------------------

        # Dropdown listing supported Whisper models
        self.model_box = QComboBox()
        self.model_box.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_box.setCurrentText(settings.whisper_model)

        # Optional explicit path to a local CTranslate2 Whisper model directory
        self.whisper_path_edit = QLineEdit(settings.whisper_model_path)

        # Language override (blank = auto-detect)
        self.lang_edit = QLineEdit(settings.language)

        # Device selection
        self.device_box = QComboBox()
        self.device_box.addItems(["auto", "cpu", "cuda"])
        self.device_box.setCurrentText(settings.device)

        # Inference compute type
        self.compute_box = QComboBox()
        self.compute_box.addItems(["auto", "int8", "float16", "float32"])
        self.compute_box.setCurrentText(settings.compute_type)

        # ----------------------------
        # External binary paths
        # ----------------------------

        # Optional path override for FFmpeg
        self.ffmpeg_edit = QLineEdit(settings.ffmpeg_path)

        # Optional path override for llama.cpp binary
        self.llama_bin_edit = QLineEdit(settings.llama_path)

        # ----------------------------
        # LLM model configuration
        # ----------------------------

        # GGUF model path used for summarization
        self.llama_model_edit = QLineEdit(settings.llama_model_path)

        # Context window size for llama.cpp
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 32768)
        self.ctx_spin.setValue(settings.llama_ctx)

        # Thread count (0 lets llama.cpp choose automatically)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 128)
        self.threads_spin.setValue(settings.llama_threads)

        # Add fields to the form layout
        form.addRow("Whisper model (fallback)", self.model_box)
        form.addRow("Whisper model path (CTranslate2 dir)", self.whisper_path_edit)
        form.addRow("Language (blank=auto)", self.lang_edit)
        form.addRow("Device", self.device_box)
        form.addRow("Compute type", self.compute_box)

        form.addRow("FFmpeg path (optional)", self.ffmpeg_edit)
        form.addRow("llama.cpp binary path (optional)", self.llama_bin_edit)

        form.addRow("LLM model path (GGUF)", self.llama_model_edit)
        form.addRow("LLM context", self.ctx_spin)
        form.addRow("LLM threads (0=auto)", self.threads_spin)

        layout.addLayout(form)

        # ----------------------------
        # Save / Cancel buttons
        # ----------------------------

        btns = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self.on_save)
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def on_save(self):
        """
        Apply UI values to the settings object and write them to disk.
        """

        s = self.settings
        s.whisper_model = self.model_box.currentText()
        s.whisper_model_path = self.whisper_path_edit.text().strip()
        s.language = self.lang_edit.text().strip()
        s.device = self.device_box.currentText()
        s.compute_type = self.compute_box.currentText()
        s.ffmpeg_path = self.ffmpeg_edit.text().strip()
        s.llama_path = self.llama_bin_edit.text().strip()
        s.llama_model_path = self.llama_model_edit.text().strip()
        s.llama_ctx = int(self.ctx_spin.value())
        s.llama_threads = int(self.threads_spin.value())

        # Persist settings to settings.json
        save_settings(s)

        # Close dialog
        self.accept()


class MainWindow(QMainWindow):
    """
    Main application window.

    Provides UI for:
    - Selecting a media file
    - Running transcription + summarization
    - Viewing transcript and summary
    - Exporting results
    - Opening settings
    """

    def __init__(self):
        super().__init__()

        # Window title and initial size
        self.setWindowTitle("Local Transcriber (Offline)")
        self.resize(980, 720)

        # Initialize database tables
        init_db()

        # Load saved settings
        self.settings = load_settings()

        # Track currently selected file
        self.current_file: Path | None = None

        # Worker thread instance (created per job)
        self.worker: TranscribeSummarizeWorker | None = None

        # Paths to most recent outputs
        self.last_transcript_json: Path | None = None
        self.last_summary_md: Path | None = None

        # Root container widget
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # ----------------------------
        # Top control bar
        # ----------------------------

        top = QHBoxLayout()
        self.file_label = QLabel("No file selected.")
        pick_btn = QPushButton("Choose File")
        pick_btn.clicked.connect(self.pick_file)

        # Main action button
        self.run_btn = QPushButton("Transcribe + Summarize")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.start_job)

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self.open_settings)

        top.addWidget(self.file_label, 1)
        top.addWidget(pick_btn)
        top.addWidget(self.run_btn)
        top.addWidget(settings_btn)
        layout.addLayout(top)

        # Status label showing progress messages
        self.status = QLabel("Ready.")
        layout.addWidget(self.status)

        # ----------------------------
        # Main display area
        # ----------------------------

        body = QHBoxLayout()

        # Transcript viewer
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setPlaceholderText("Transcript will appear here...")

        # Summary viewer
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("Summary will appear here...")

        body.addWidget(self.transcript_box, 2)
        body.addWidget(self.summary_box, 1)
        layout.addLayout(body)

        # ----------------------------
        # Export buttons
        # ----------------------------

        export_row = QHBoxLayout()
        self.export_txt_btn = QPushButton("Export Transcript TXT")
        self.export_md_btn = QPushButton("Export Summary MD")
        self.export_txt_btn.setEnabled(False)
        self.export_md_btn.setEnabled(False)
        self.export_txt_btn.clicked.connect(self.export_transcript_txt)
        self.export_md_btn.clicked.connect(self.export_summary_md)
        export_row.addWidget(self.export_txt_btn)
        export_row.addWidget(self.export_md_btn)
        export_row.addStretch(1)
        layout.addLayout(export_row)

    def pick_file(self):
        """
        Open a file picker for audio/video input.
        """

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio/video file",
            "",
            "Media Files (*.wav *.mp3 *.m4a *.mp4 *.mov *.aac *.flac *.ogg);;All Files (*.*)"
        )
        if not path:
            return
        self.current_file = Path(path)
        self.file_label.setText(str(self.current_file))

        # Enable transcription button
        self.run_btn.setEnabled(True)

    def open_settings(self):
        """
        Open the settings dialog.
        """

        dlg = SettingsDialog(self, self.settings)
        if dlg.exec():
            # Reload settings after save
            self.settings = load_settings()
            QMessageBox.information(self, "Saved", "Settings saved.")

    def start_job(self):
        """
        Start the transcription + summarization worker thread.
        """

        if not self.current_file:
            return

        # Prevent starting multiple jobs simultaneously
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return

        # Reset UI state
        self.run_btn.setEnabled(False)
        self.export_txt_btn.setEnabled(False)
        self.export_md_btn.setEnabled(False)
        self.transcript_box.clear()
        self.summary_box.clear()
        self.status.setText("Starting...")

        # Create worker thread
        self.worker = TranscribeSummarizeWorker(self.current_file, self.settings)

        # Connect worker signals to UI handlers
        self.worker.signals.progress.connect(self.status.setText)
        self.worker.signals.error.connect(self.on_error)
        self.worker.signals.finished.connect(self.on_finished)

        # Start the background job
        self.worker.start()

    def on_error(self, msg: str):
        """
        Display error message from worker thread.
        """

        self.status.setText("Error.")
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def on_finished(self, job_id: str, transcript_json_path: str, summary_md_path: str):
        """
        Handle completion of the worker job.
        """

        self.status.setText(f"Completed job {job_id}")
        self.run_btn.setEnabled(True)

        # Store paths for export buttons
        self.last_transcript_json = Path(transcript_json_path)
        self.last_summary_md = Path(summary_md_path)

        # ----------------------------
        # Display transcript
        # ----------------------------

        data = json.loads(self.last_transcript_json.read_text(encoding="utf-8"))
        lines = []
        for seg in data["segments"]:
            start = float(seg["start"])
            end = float(seg["end"])
            text = seg["text"]
            lines.append(f"[{start:7.2f} - {end:7.2f}] {text}")
        self.transcript_box.setPlainText("\n".join(lines))

        # ----------------------------
        # Display summary
        # ----------------------------

        self.summary_box.setPlainText(self.last_summary_md.read_text(encoding="utf-8"))

        # Enable export buttons
        self.export_txt_btn.setEnabled(True)
        self.export_md_btn.setEnabled(True)

    def export_transcript_txt(self):
        """
        Export transcript as plain text.
        """

        if not self.last_transcript_json:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save transcript as TXT", "", "Text Files (*.txt)")
        if not save_path:
            return
        data = json.loads(self.last_transcript_json.read_text(encoding="utf-8"))
        text = "\n".join([seg["text"] for seg in data["segments"]]).strip() + "\n"
        Path(save_path).write_text(text, encoding="utf-8")
        QMessageBox.information(self, "Exported", f"Saved: {save_path}")

    def export_summary_md(self):
        """
        Export summary Markdown file.
        """

        if not self.last_summary_md:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save summary as Markdown", "", "Markdown Files (*.md)")
        if not save_path:
            return
        Path(save_path).write_text(self.last_summary_md.read_text(encoding="utf-8"), encoding="utf-8")
        QMessageBox.information(self, "Exported", f"Saved: {save_path}")