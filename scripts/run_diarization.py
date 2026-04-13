from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["PYANNOTE_AUDIO_BACKEND"] = "soundfile"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--device", default="auto")   # auto / cpu / cuda
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise RuntimeError(f"Input WAV not found: {input_path}")

    try:
        import torch
        import soundfile as sf
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies in diarization environment. "
            "Install: torch, pyannote.audio, soundfile"
        ) from e

    requested_device = (args.device or "auto").strip().lower()
    has_cuda = torch.cuda.is_available()

    if requested_device == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA was requested for diarization, but torch.cuda.is_available() is False.")
        final_device = "cuda"
    elif requested_device == "cpu":
        final_device = "cpu"
    else:
        final_device = "cuda" if has_cuda else "cpu"

    kwargs = {}
    if args.hf_token.strip():
        kwargs["token"] = args.hf_token.strip()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        **kwargs,
    )

    if final_device == "cuda":
        pipeline.to(torch.device("cuda"))

    audio, sample_rate = sf.read(str(input_path), dtype="float32")

    if audio.ndim == 1:
        waveform = torch.from_numpy(audio).unsqueeze(0)
    else:
        waveform = torch.from_numpy(audio.T)

    if final_device == "cuda":
        waveform = waveform.to("cuda")

    diarization = pipeline(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
    )

    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    elif hasattr(diarization, "annotation"):
        annotation = diarization.annotation
    else:
        annotation = diarization

    speaker_turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    payload = {
        "device": final_device,
        "speaker_turns": speaker_turns,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())