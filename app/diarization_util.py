from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.models import TranscriptSegment


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_segments(
    segments: Iterable[TranscriptSegment],
    speaker_turns: Iterable[SpeakerTurn],
    unknown_label: str = "UNKNOWN",
) -> list[TranscriptSegment]:
    turns = list(speaker_turns)
    updated: list[TranscriptSegment] = []

    for seg in segments:
        best_speaker = unknown_label
        best_overlap = 0.0

        for turn in turns:
            ov = _overlap(seg.start, seg.end, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn.speaker

        updated.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=best_speaker,
            )
        )

    return updated


def run_diarization_subprocess(
    wav_path: Path,
    output_json_path: Path,
    diarization_python: str,
    hf_token: str = "",
    device: str = "auto",
    timeout_s: int = 1800,
) -> list[SpeakerTurn]:
    """
    Run diarization in a separate Python environment and return speaker turns.
    """
    if not wav_path.exists():
        raise RuntimeError(f"Diarization input file not found: {wav_path}")

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_diarization.py"
    if not script_path.exists():
        raise RuntimeError(f"Diarization runner script not found: {script_path}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        diarization_python,
        str(script_path),
        "--input", str(wav_path),
        "--output", str(output_json_path),
        "--device", device,
    ]

    if hf_token.strip():
        cmd += ["--hf-token", hf_token.strip()]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Diarization Python executable was not found.\n"
            f"Configured path: {diarization_python}"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            "Diarization subprocess failed.\n\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDERR:\n{(proc.stderr or '').strip()}\n\n"
            f"STDOUT:\n{(proc.stdout or '').strip()}"
        )

    if not output_json_path.exists():
        raise RuntimeError(
            "Diarization subprocess reported success but no output JSON was created.\n"
            f"Expected: {output_json_path}"
        )

    try:
        data = json.loads(output_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse diarization output JSON: {output_json_path}"
        ) from e

    turns: list[SpeakerTurn] = []
    for item in data.get("speaker_turns", []):
        turns.append(
            SpeakerTurn(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker=str(item["speaker"]),
            )
        )

    return turns

def normalize_speaker_labels(
    speaker_turns: Iterable[SpeakerTurn],
) -> list[SpeakerTurn]:
    """
    Convert raw diarization labels like SPEAKER_00 / SPEAKER_01
    into user-friendly labels like Speaker A / Speaker B.
    """
    turns = list(speaker_turns)

    raw_labels: list[str] = []
    for turn in turns:
        if turn.speaker not in raw_labels:
            raw_labels.append(turn.speaker)

    label_map: dict[str, str] = {}
    for i, raw_label in enumerate(raw_labels):
        friendly = f"Speaker {chr(ord('A') + i)}"
        label_map[raw_label] = friendly

    normalized: list[SpeakerTurn] = []
    for turn in turns:
        normalized.append(
            SpeakerTurn(
                start=turn.start,
                end=turn.end,
                speaker=label_map.get(turn.speaker, turn.speaker),
            )
        )

    return normalized