from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Fill these in with your actual repository values
GITHUB_OWNER = "AnujRimal3"
GITHUB_REPO = "ai-audio-transcription-capstone"
RELEASE_TAG = "assets-v1"

# Direct download URL for the large GGUF summarization model
GGUF_MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf?download=true"
GGUF_MODEL_FILENAME = "summary-model.gguf"

# Repository root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]

ASSETS_DIR = REPO_ROOT / "assets"
BIN_DIR = ASSETS_DIR / "bin"
GRAMMAR_DIR = ASSETS_DIR / "grammars"
MODELS_DIR = ASSETS_DIR / "models"
LLM_DIR = MODELS_DIR / "llm"
WHISPER_DIR = MODELS_DIR / "whisper"
WHISPER_SMALL_DIR = WHISPER_DIR / "small"


# ---------------------------------------------------------
# Asset model
# ---------------------------------------------------------

@dataclass
class AssetSpec:
    name: str
    url: str
    dest: Path
    is_zip: bool = False
    sha256: str | None = None
    required: bool = True
    manual_hint: str | None = None


# ---------------------------------------------------------
# Asset list
# ---------------------------------------------------------

def github_release_url(filename: str) -> str:
    return (
        f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/"
        f"{RELEASE_TAG}/{filename}"
    )


ASSETS: list[AssetSpec] = [
    AssetSpec(
        name="Bundled binaries",
        url=github_release_url("bin.zip"),
        dest=BIN_DIR,
        is_zip=True,
    ),
    AssetSpec(
        name="Whisper small CTranslate2 model",
        url=github_release_url("whisper-small-ct2.zip"),
        dest=WHISPER_SMALL_DIR,
        is_zip=True,
    ),
    AssetSpec(
        name="GGUF summarization model",
        url=GGUF_MODEL_URL,
        dest=LLM_DIR / GGUF_MODEL_FILENAME,
        is_zip=False,
        required=True,
        manual_hint=(
            f"Download the GGUF model manually and place it here:\n"
            f"  {LLM_DIR / GGUF_MODEL_FILENAME}\n"
            f"Expected source URL:\n"
            f"  {GGUF_MODEL_URL}"
        ),
    ),
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_dirs() -> None:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_SMALL_DIR.mkdir(parents=True, exist_ok=True)


def dir_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def asset_present(spec: AssetSpec) -> bool:
    if spec.is_zip:
        return dir_has_files(spec.dest)
    return spec.dest.exists()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path.name}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "LocalTranscriberBootstrap/1.0"})
    with urlopen(req) as response, dest.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            if total_size:
                pct = downloaded * 100 // total_size
                print(
                    f"  {downloaded / (1024 * 1024):7.1f} MB / "
                    f"{total_size / (1024 * 1024):7.1f} MB ({pct:3d}%)",
                    end="\r",
                )

    if total_size:
        print(" " * 100, end="\r")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def flatten_single_nested_dir(dest_dir: Path) -> None:
    """
    If extraction produced exactly one child directory and no sibling files,
    move that directory's contents up one level.

    Example:
        assets/bin/bin/... -> assets/bin/...
        assets/models/whisper/small/small/... -> .../small/...
    """
    if not dest_dir.exists():
        return

    items = list(dest_dir.iterdir())
    if len(items) != 1:
        return

    only_item = items[0]
    if not only_item.is_dir():
        return

    nested_items = list(only_item.iterdir())
    for item in nested_items:
        target = dest_dir / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))

    try:
        only_item.rmdir()
    except Exception:
        pass


def install_asset(spec: AssetSpec) -> tuple[bool, str | None]:
    """
    Returns:
        (success, error_message_or_none)
    """
    if asset_present(spec):
        print(f"[skip] {spec.name} already present")
        return True, None

    print(f"[download] {spec.name}")
    print(f"  URL: {spec.url}")

    if spec.is_zip:
        temp_zip = REPO_ROOT / "scripts" / f"__tmp_{spec.dest.name or 'asset'}.zip"
        try:
            download_file(spec.url, temp_zip)
            verify_sha256(temp_zip, spec.sha256)

            print(f"[extract] {spec.name} -> {spec.dest}")
            extract_zip(temp_zip, spec.dest)
            flatten_single_nested_dir(spec.dest)

            print(f"[done] {spec.name}")
            return True, None

        except Exception as e:
            return False, str(e)

        finally:
            if temp_zip.exists():
                try:
                    temp_zip.unlink()
                except Exception:
                    pass

    try:
        download_file(spec.url, spec.dest)
        verify_sha256(spec.dest, spec.sha256)
        print(f"[done] {spec.name}")
        return True, None
    except Exception as e:
        return False, str(e)


def verify_required_files() -> None:
    grammar_file = GRAMMAR_DIR / "summary_json.gbnf"
    if not grammar_file.exists():
        print(
            "[warn] summary_json.gbnf is missing.\n"
            "       Commit this file to the repo under assets/grammars/summary_json.gbnf"
        )

    if not dir_has_files(BIN_DIR):
        raise RuntimeError("assets/bin is missing or empty after bootstrap.")

    ffmpeg_candidates = [
        BIN_DIR / "ffmpeg.exe",
        BIN_DIR / "ffmpeg",
    ]
    llama_candidates = [
        BIN_DIR / "llama-cli.exe",
        BIN_DIR / "llama-cli",
        BIN_DIR / "llama.exe",
        BIN_DIR / "llama",
        BIN_DIR / "main.exe",
        BIN_DIR / "main",
        BIN_DIR / "llama-completion.exe",
        BIN_DIR / "llama-completion",
    ]

    if not any(p.exists() for p in ffmpeg_candidates):
        raise RuntimeError(
            "No FFmpeg executable found in assets/bin after extracting bin.zip."
        )

    if not any(p.exists() for p in llama_candidates):
        raise RuntimeError(
            "No llama.cpp executable found in assets/bin after extracting bin.zip."
        )

    if not any(LLM_DIR.glob("*.gguf")):
        raise RuntimeError(
            "No GGUF model found in assets/models/llm.\n"
            f"Expected at least one .gguf file in:\n  {LLM_DIR}"
        )

    if not dir_has_files(WHISPER_SMALL_DIR):
        raise RuntimeError(
            "Whisper model folder assets/models/whisper/small is missing or empty."
        )


def print_manual_model_instructions() -> None:
    model_dest = LLM_DIR / GGUF_MODEL_FILENAME
    print()
    print("[manual step required] GGUF summarization model")
    print("The model could not be downloaded automatically.")
    print("Download it manually and place it here:")
    print(f"  {model_dest}")
    print("Suggested source URL:")
    print(f"  {GGUF_MODEL_URL}")


def main() -> int:
    print("LocalTranscriber bootstrap")
    print("==========================")
    print(f"Repo root: {REPO_ROOT}")
    print()

    ensure_dirs()

    gguf_failed = False
    gguf_error: str | None = None

    try:
        for spec in ASSETS:
            ok, err = install_asset(spec)

            if ok:
                continue

            print()
            print(f"[error] Failed to install: {spec.name}")
            print(err or "Unknown error")

            is_gguf = spec.dest == (LLM_DIR / GGUF_MODEL_FILENAME)
            if is_gguf:
                gguf_failed = True
                gguf_error = err
                print_manual_model_instructions()
                continue

            if spec.required:
                return 1

        verify_required_files()

    except Exception as e:
        print()
        print("[error] Bootstrap failed")
        print(str(e))
        return 1

    print()
    if gguf_failed:
        print("[partial success] Bootstrap completed except for the GGUF model.")
        print("Everything else is in place.")
        print_manual_model_instructions()
        return 1

    print("[success] Bootstrap complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())