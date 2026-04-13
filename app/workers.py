"""
workers.py
----------

Background processing workers for LocalTranscriber.

This module implements the threaded pipeline that performs
the full transcription and summarization workflow.

Pipeline Steps
--------------
1. Register job in database
2. Convert input media to Whisper-compatible WAV
3. Transcribe audio using faster-whisper
4. Save transcript JSON artifact
5. Generate transcript summary using llama.cpp
6. Save summary Markdown
7. Update job record in database

Threading
---------
Processing runs inside a QThread so that long-running
tasks do not block the GUI.

Signals communicate progress updates, errors, and completion
events back to the UI.

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
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QThread

from app.settings import AppSettings
from app.paths import artifacts_dir, bundled_whisper_small_dir
from app.ffmpeg_util import find_ffmpeg, convert_to_wav_16k_mono
from app.whisper_util import transcribe_file, resolve_whisper_runtime, merge_segments_into_sentences
from app.llama_util import find_llama, summarize_general_chunked, ChunkedSummaryConfig
from app.storage import connect
from app.models import TranscriptResult
from app.diarization_util import run_diarization_subprocess, assign_speakers_to_segments, normalize_speaker_labels


class JobSignals(QObject):
    """
    Signals emitted by the background worker thread.

    progress: Sends human-readable status updates to the UI.
    error: Sends an error message if the job fails.
    finished: Sends final output paths when the job completes successfully.
    """

    progress = Signal(str)
    error = Signal(str)
    finished = Signal(str, str, str)  # job_id, transcript_json_path, summary_md_path


class TranscribeSummarizeWorker(QThread):
    """
    Background worker that performs the full offline pipeline:

    1) Register a job in the database
    2) Locate FFmpeg
    3) Convert input media to 16kHz mono WAV
    4) Transcribe audio with faster-whisper
    5) Save transcript.json
    6) Summarize transcript with llama.cpp (if GGUF configured)
    7) Save summary.md
    8) Update the job record in the database
    9) Send completion or error signals back to the UI
    """

    def __init__(self, input_path: Path, settings: AppSettings):
        super().__init__()
        # Original user-selected media file
        self.input_path = input_path

        # Snapshot of current application settings
        self.settings = settings

        # Signal bundle used to communicate with the main UI thread
        self.signals = JobSignals()

    def run(self) -> None:
        """
        Entry point for the worker thread.

        This method is executed in the background when `start()` is called on
        the QThread instance. It should never directly update the UI; instead,
        it communicates progress/results through signals. -Zack
        """

        # Create a unique ID for this processing job
        job_id = str(uuid.uuid4())

        # Store UTC creation time in ISO 8601 format
        created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # These will be filled in as the pipeline succeeds
        transcript_json = ""
        summary_md = ""

        # Insert an initial "running" record into the jobs table
        with connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, created_at, input_path, status) VALUES (?, ?, ?, ?)",
                (job_id, created_at, str(self.input_path), "running")
            )
            conn.commit()

        try:
            # ----------------------------
            # Step 1: Locate FFmpeg
            # ----------------------------
            self.signals.progress.emit("Locating FFmpeg...")

            # Use explicit FFmpeg path from settings if provided
            # Otherwise use bundled/system discovery
            ffmpeg = find_ffmpeg(self.settings.ffmpeg_path)

            # ----------------------------
            # Step 2: Convert input file to Whisper-friendly WAV
            # ----------------------------
            self.signals.progress.emit("Converting audio to 16kHz mono WAV...")

            # Output WAV path lives under artifacts/<job_id>/
            wav_out = artifacts_dir() / job_id / "audio_16k.wav"

            # Convert source media into mono 16kHz WAV for transcription
            convert_to_wav_16k_mono(ffmpeg, self.input_path, wav_out)

            # ----------------------------
            # Step 3: Resolve Whisper model reference
            # ----------------------------
            # Priority:
            # 1) explicit whisper_model_path from settings
            # 2) bundled whisper small model directory
            # 3) plain model name (may trigger download/cache in faster-whisper)
            model_ref = self.settings.whisper_model_path.strip()

            if not model_ref and bundled_whisper_small_dir().exists():
                model_ref = str(bundled_whisper_small_dir())

            if not model_ref:
                # This fallback may download/cache if not already present
                # So it is not guaranteed to remain fully offline
                # Under normal operating circumstances we should never trigger this
                # But just in case it comes up we'll have this contingency -Zack
                model_ref = self.settings.whisper_model

            # ----------------------------
            # Step 4: Transcribe audio with faster-whisper
            # ----------------------------
            self.signals.progress.emit("Preparing transcription runtime...")

            final_device, final_compute_type = resolve_whisper_runtime(
                self.settings.device,
                self.settings.compute_type,
            )

            self.signals.progress.emit(
                f"Transcribing with faster-whisper... "
                f"(device={final_device}, compute_type={final_compute_type})"
            )

            tr: TranscriptResult = transcribe_file(
                wav_out,
                model_name_or_path=model_ref,
                device=final_device,
                compute_type=final_compute_type,
                language=self.settings.language,
            )

            # ----------------------------
            # Step 5: Optional speaker diarization
            # ----------------------------
            if self.settings.enable_diarization:
                diar_py = self.settings.diarization_python_path.strip()
                if not diar_py:
                    raise RuntimeError(
                        "Speaker diarization is enabled, but no diarization Python path is configured in Settings."
                    )

                self.signals.progress.emit("Running speaker diarization..."
                                           f"(device={final_device}, compute_type={final_compute_type})")
                diarization_json_path = artifacts_dir() / job_id / "diarization.json"

                speaker_turns = run_diarization_subprocess(
                    wav_path=wav_out,
                    output_json_path=diarization_json_path,
                    diarization_python=diar_py,
                    hf_token=self.settings.hf_token,
                    device=self.settings.device,
                )

                speaker_turns = normalize_speaker_labels(speaker_turns)

                tr.segments = assign_speakers_to_segments(tr.segments, speaker_turns)

                tr.segments = merge_segments_into_sentences(tr.segments)

            # ----------------------------
            # Step 6: Save transcript JSON output
            # ----------------------------
            out_dir = artifacts_dir() / job_id
            out_dir.mkdir(parents=True, exist_ok=True)

            transcript_json_path = out_dir / "transcript.json"

            # Serialize transcript into a structured JSON payload
            payload = {
                "job_id": job_id,
                "input_path": str(self.input_path),
                "language": tr.language,
                "segments": [asdict(s) for s in tr.segments],
            }
            transcript_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            transcript_json = str(transcript_json_path)

            # Include speaker labels in the summary input
            full_text = "\n".join([f"{s.speaker}: {s.text}" for s in tr.segments]).strip()

            # ----------------------------
            # Step 7: Summarize transcript with llama.cpp if GGUF is configured
            # ----------------------------
            summary_md_path = out_dir / "summary.md"
            if self.settings.llama_model_path.strip():
                if self.settings.llama_use_gpu:
                    self.signals.progress.emit(
                        f"Summarizing with llama.cpp GPU offload "
                        f"(gpu_layers={self.settings.llama_gpu_layers})..."
                    )
                else:
                    self.signals.progress.emit("Summarizing locally with llama.cpp (chunked)...")

                # Resolve the llama.cpp executable
                llama_bin = find_llama(self.settings.llama_path)
                self.signals.progress.emit(f"Using llama binary: {llama_bin}")

                # Chunked summarization config; defaults are currently used -Zack
                cfg = ChunkedSummaryConfig()

                # Small adapter so summarize_general_chunked() can report
                # Progress back through the worker's Qt signal
                def progress(msg: str):
                    self.signals.progress.emit(msg)

                # Run map→reduce summarization over the transcript text
                summary = summarize_general_chunked(
                    llama_bin=llama_bin,
                    model_path=Path(self.settings.llama_model_path),
                    transcript_text=full_text,
                    ctx=self.settings.llama_ctx,
                    threads=self.settings.llama_threads,
                    config=cfg,
                    progress_cb=progress,
                    use_gpu=self.settings.llama_use_gpu,
                    gpu_layers=self.settings.llama_gpu_layers,
                )

                # Save final Markdown summary
                summary_md_path.write_text(summary, encoding="utf-8")
                summary_md = str(summary_md_path)
            else:
                # If no GGUF model is configured, still create a summary file
                # Explaining that summarization is disabled until configured
                summary_md_path.write_text(
                    "## Summary\n\n(Configure a GGUF model in Settings to enable local summarization.)\n",
                    encoding="utf-8"
                )
                summary_md = str(summary_md_path)

            # ----------------------------
            # Step 8: Mark job as complete in the database
            # ----------------------------
            with connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status=?, transcript_json_path=?, summary_md_path=? WHERE id=?",
                    ("done", transcript_json, summary_md, job_id)
                )
                conn.commit()

            # Notify UI that processing is complete
            self.signals.progress.emit("Done.")
            self.signals.finished.emit(job_id, transcript_json, summary_md)

        except Exception as e:
            # ----------------------------
            # Error handling
            # ----------------------------
            # Any failure in conversion, transcription, summarization, or file IO
            # ends up here. We store the error in the database and notify the UI.
            msg = str(e)
            with connect() as conn:
                conn.execute("UPDATE jobs SET status=?, error=? WHERE id=?", ("error", msg, job_id))
                conn.commit()
            self.signals.error.emit(msg)