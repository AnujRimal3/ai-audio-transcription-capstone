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
    QLineEdit, QComboBox, QDialog, QFormLayout, QMessageBox, QSpinBox, QCheckBox, QSlider
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QTextCursor, QTextCursor, QColor, QTextCharFormat
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from app.settings import load_settings, save_settings, AppSettings
from app.storage import init_db
from app.workers import TranscribeSummarizeWorker

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds → MM:SS or HH:MM:SS if needed.
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    if h > 0:
        return f"{h:02}:{m:02}:{s:02}"
    return f"{m:02}:{s:02}"

class ClickableTranscriptBox(QTextEdit):
    """
    QTextEdit that emits the clicked block number.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.block_clicked_cb = None

    def mousePressEvent(self, event):
        cursor = self.cursorForPosition(event.position().toPoint())
        block = cursor.block()
        if block.isValid() and self.block_clicked_cb is not None:
            self.block_clicked_cb(block.blockNumber())
        super().mousePressEvent(event)

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

        # Diarization
        self.diarization_check = QCheckBox("Enable speaker diarization")
        self.diarization_check.setChecked(settings.enable_diarization)

        self.diarization_python_edit = QLineEdit(settings.diarization_python_path)
        self.hf_token_edit = QLineEdit(settings.hf_token)
        self.hf_token_edit.setEchoMode(QLineEdit.Password)

        form.addRow("Enable diarization", self.diarization_check)
        form.addRow("Diarization Python path", self.diarization_python_edit)
        form.addRow("Hugging Face token", self.hf_token_edit)

        self.llama_gpu_check = QCheckBox("Enable llama.cpp GPU offload")
        self.llama_gpu_check.setChecked(settings.llama_use_gpu)

        self.llama_gpu_layers_spin = QSpinBox()
        self.llama_gpu_layers_spin.setRange(0, 999)
        self.llama_gpu_layers_spin.setValue(settings.llama_gpu_layers)

        form.addRow("LLM GPU offload", self.llama_gpu_check)
        form.addRow("LLM GPU layers", self.llama_gpu_layers_spin)



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
        s.enable_diarization = self.diarization_check.isChecked()
        s.diarization_python_path = self.diarization_python_edit.text().strip()
        s.hf_token = self.hf_token_edit.text().strip()
        s.llama_use_gpu = self.llama_gpu_check.isChecked()
        s.llama_gpu_layers = int(self.llama_gpu_layers_spin.value())

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

        self.current_audio_path: Path | None = None
        self.current_playing_segment_index: int = -1

        # Search state
        self.search_matches: list[tuple[int, int]] = []
        self.search_match_pos: int = -1
        self.last_search_query: str = ""

        # Audio playback
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)

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
        # Search row
        # ----------------------------
        search_row = QHBoxLayout()

        self.search_edit = QLineEdit()
        self.search_edit.textChanged.connect(self.on_search_text_changed)
        self.search_edit.setPlaceholderText("Search transcript...")
        self.search_edit.returnPressed.connect(self.search_next)

        self.search_prev_btn = QPushButton("Prev")
        self.search_next_btn = QPushButton("Next")

        self.search_prev_btn.setEnabled(False)
        self.search_next_btn.setEnabled(False)

        self.search_prev_btn.clicked.connect(self.search_prev)
        self.search_next_btn.clicked.connect(self.search_next)

        search_row.addWidget(QLabel("Search:"))
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(self.search_prev_btn)
        search_row.addWidget(self.search_next_btn)

        layout.addLayout(search_row)

        # ----------------------------
        # Main display area
        # ----------------------------

        body = QHBoxLayout()

        # Transcript viewer
        self.transcript_box = ClickableTranscriptBox()
        self.transcript_box.block_clicked_cb = self.on_transcript_block_clicked
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
        # Audio controls
        # ----------------------------
        audio_row = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")

        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self.play_btn.clicked.connect(self.play_audio)
        self.pause_btn.clicked.connect(self.pause_audio)
        self.stop_btn.clicked.connect(self.stop_audio)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.on_slider_moved)

        self.time_label = QLabel("00:00 / 00:00")

        audio_row.addWidget(self.play_btn)
        audio_row.addWidget(self.pause_btn)
        audio_row.addWidget(self.stop_btn)
        audio_row.addWidget(self.position_slider, 1)
        audio_row.addWidget(self.time_label)

        layout.addLayout(audio_row)

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

        self.current_segments = []
        lines = []

        for seg in data["segments"]:
            if seg.get("start") is None or seg.get("end") is None:
                continue

            start = float(seg["start"])
            end = float(seg["end"])
            text = seg.get("text", "")
            speaker = seg.get("speaker", "UNKNOWN")

            self.current_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker,
                }
            )

            lines.append(
                f"[{format_timestamp(start)} - {format_timestamp(end)}] {speaker}: {text}"
            )

        self.transcript_box.setPlainText("\n".join(lines))

        # ----------------------------
        # Display summary
        # ----------------------------

        self.summary_box.setPlainText(self.last_summary_md.read_text(encoding="utf-8"))

        # Enable export buttons
        self.export_txt_btn.setEnabled(True)
        self.export_md_btn.setEnabled(True)

        # Reset search state when new transcript loads
        self.search_matches = []
        self.search_match_pos = -1
        self.last_search_query = ""
        self._clear_search_highlighting()
        self.search_prev_btn.setEnabled(True)
        self.search_next_btn.setEnabled(True)

        # Resolve audio path from the job artifact directory
        self.current_audio_path = self.last_transcript_json.parent / "audio_16k.wav"

        if self.current_audio_path.exists():
            self.player.setSource(QUrl.fromLocalFile(str(self.current_audio_path)))
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        else:
            self.current_audio_path = None
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

        self.current_playing_segment_index = -1
        self._refresh_highlighting()

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
        text = "\n".join(
            [
                f"[{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}] "
                f"{seg.get('speaker', 'UNKNOWN')}: {seg['text']}"
                for seg in data["segments"]
            ]
        ).strip() + "\n"
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

    def _clear_search_highlighting(self) -> None:
        self.search_matches = []
        self.search_match_pos = -1
        self._refresh_highlighting()

    def _refresh_highlighting(self) -> None:
        """
        Apply combined transcript highlighting:
        - all search matches (light)
        - current search match (strong)
        - currently playing transcript segment (blue tint)
        """
        selections = []

        # Light highlight for all search matches
        all_fmt = QTextCharFormat()
        all_fmt.setBackground(QColor("#fff59d"))  # light yellow

        for start_pos, end_pos in self.search_matches:
            cursor = self.transcript_box.textCursor()
            cursor.setPosition(start_pos)
            cursor.setPosition(end_pos, QTextCursor.KeepAnchor)

            sel = QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = all_fmt
            selections.append(sel)

        # Stronger highlight for current search match
        if 0 <= self.search_match_pos < len(self.search_matches):
            start_pos, end_pos = self.search_matches[self.search_match_pos]

            current_fmt = QTextCharFormat()
            current_fmt.setBackground(QColor("#ffd54f"))  # darker yellow

            cursor = self.transcript_box.textCursor()
            cursor.setPosition(start_pos)
            cursor.setPosition(end_pos, QTextCursor.KeepAnchor)

            sel = QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = current_fmt
            selections.append(sel)

        # Blue highlight for current playing segment
        if 0 <= self.current_playing_segment_index < len(self.current_segments):
            doc = self.transcript_box.document()
            block = doc.findBlockByLineNumber(self.current_playing_segment_index)
            if block.isValid():
                cursor = QTextCursor(block)
                cursor.select(QTextCursor.BlockUnderCursor)

                play_fmt = QTextCharFormat()
                play_fmt.setBackground(QColor("#cfe8ff"))  # light blue

                sel = QTextEdit.ExtraSelection()
                sel.cursor = cursor
                sel.format = play_fmt
                selections.append(sel)

                # keep current playing block visible
                self.transcript_box.setTextCursor(cursor)
                self.transcript_box.ensureCursorVisible()

        self.transcript_box.setExtraSelections(selections)

    def _rebuild_search_matches(self, query: str) -> None:
        """
        Find every occurrence of the query in the transcript text box.
        Stores exact character ranges for navigation/highlighting.
        """
        self.search_matches = []
        self.search_match_pos = -1
        self.last_search_query = query

        if not query:
            self._clear_search_highlighting()
            return

        document = self.transcript_box.document()
        cursor = QTextCursor(document)

        while True:
            cursor = document.find(query, cursor)
            if cursor.isNull():
                break

            self.search_matches.append((cursor.selectionStart(), cursor.selectionEnd()))

        self._refresh_highlighting()

    def _ensure_search_matches(self) -> str:
        """
        Ensure matches are up to date for the current query.
        """
        query = self.search_edit.text().strip()

        if query != self.last_search_query:
            self._rebuild_search_matches(query)

        return query

    def search_next(self) -> None:
        """
        Jump to the next search occurrence.
        """
        query = self._ensure_search_matches()

        if not query:
            self.status.setText("Enter a search term.")
            self._clear_search_highlighting()
            return

        if not self.search_matches:
            self.status.setText(f'No matches for "{query}".')
            self._clear_search_highlighting()
            return

        if self.search_match_pos == -1:
            self.search_match_pos = 0
        else:
            self.search_match_pos = (self.search_match_pos + 1) % len(self.search_matches)

        self._refresh_highlighting()
        self.status.setText(
            f'Match {self.search_match_pos + 1}/{len(self.search_matches)} for "{query}"'
        )

    def search_prev(self) -> None:
        """
        Jump to the previous search occurrence.
        """
        query = self._ensure_search_matches()

        if not query:
            self.status.setText("Enter a search term.")
            self._clear_search_highlighting()
            return

        if not self.search_matches:
            self.status.setText(f'No matches for "{query}".')
            self._clear_search_highlighting()
            return

        if self.search_match_pos == -1:
            self.search_match_pos = len(self.search_matches) - 1
        else:
            self.search_match_pos = (self.search_match_pos - 1) % len(self.search_matches)

        self._refresh_highlighting()
        self.status.setText(
            f'Match {self.search_match_pos + 1}/{len(self.search_matches)} for "{query}"'
        )

    def on_search_text_changed(self, text: str) -> None:
        """
        Clear highlights immediately when the search box is emptied.
        """
        if not text.strip():
            self.search_matches = []
            self.search_match_pos = -1
            self.last_search_query = ""
            self._clear_search_highlighting()
            self.status.setText("Ready.")

    def play_audio(self):
        if self.current_audio_path:
            self.player.play()

    def pause_audio(self):
        self.player.pause()

    def stop_audio(self):
        self.player.stop()
        self.current_playing_segment_index = -1
        self._refresh_highlighting()

    def on_slider_moved(self, position: int):
        self.player.setPosition(position)

    def on_player_duration_changed(self, duration: int):
        self.position_slider.setRange(0, duration)
        self.time_label.setText(f"00:00 / {format_timestamp(duration / 1000.0)}")

    def on_transcript_block_clicked(self, block_number: int):
        """
        Seek audio to the clicked transcript segment.
        """
        if block_number < 0 or block_number >= len(self.current_segments):
            return

        seg = self.current_segments[block_number]
        start_ms = int(float(seg["start"]) * 1000)

        self.player.setPosition(start_ms)
        self.current_playing_segment_index = block_number
        self._refresh_highlighting()

        self.player.play()

    def on_player_position_changed(self, position: int):
        """
        Update slider, time label, and current transcript highlight as audio plays.
        """
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)

        current_sec = position / 1000.0
        total_sec = self.player.duration() / 1000.0 if self.player.duration() > 0 else 0.0
        self.time_label.setText(
            f"{format_timestamp(current_sec)} / {format_timestamp(total_sec)}"
        )

        seg_index = self._segment_index_for_time(current_sec)
        if seg_index != self.current_playing_segment_index:
            self.current_playing_segment_index = seg_index
            self._refresh_highlighting()

    def _segment_index_for_time(self, t: float) -> int:
        """
        Return the transcript segment index containing the given playback time.
        """
        for i, seg in enumerate(self.current_segments):
            start = float(seg["start"])
            end = float(seg["end"])
            if start <= t <= end:
                return i
        return -1