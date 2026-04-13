"""
llama_util.py
-------------

Utilities for running local LLM summarization using llama.cpp.

This module implements the map → reduce summarization pipeline
used by the LocalTranscriber application to generate structured
summaries of Whisper transcripts.

Responsibilities
----------------
- Locate a working llama.cpp executable
- Execute llama.cpp with safe runtime configuration
- Clean noisy CLI output produced by different llama builds
- Enforce structured JSON output using grammar-constrained decoding
- Chunk large transcripts to fit within model context windows
- Merge chunk summaries into a final structured summary

Summarization Pipeline
----------------------

The summarization process follows a two-stage architecture:

MAP STEP
    Each transcript chunk is summarized independently into
    a short bullet list.

REDUCE STEP
    All chunk summaries are merged into a final summary that
    returns structured JSON with the schema:

        {
          "bullets": ["..."],
          "paragraph": "..."
        }

Grammar-Constrained Decoding
----------------------------
The reduce step optionally uses a GBNF grammar file
(`assets/grammars/summary_json.gbnf`) to force the model
to produce valid JSON.

Without this constraint, LLMs sometimes produce malformed JSON,
extra text, or truncated output which breaks parsing.

Robust Execution
----------------
The llama runner includes safeguards for common llama.cpp issues:

- automatic flag detection across different binaries
- prompt file fallback for Windows CLI limits
- stdout cleanup to remove CLI noise
- JSON recovery via brace matching
- timeout protection for long generations

Supported Binaries
------------------
This module can run several llama.cpp variants including:

- llama-cli
- llama
- main
- llama-completion

The code dynamically detects supported flags to avoid runtime errors.

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
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ----------------------------
# llama binary discovery
# ----------------------------

def find_llama(explicit_path: str = "") -> str:
    """
    Locate the llama.cpp executable.

    Search order:
      1) explicit path (Settings)
      2) bundled assets/bin inside the repo
      3) system PATH

    We run `--help` to verify the executable launches. This is more of a convenience thing
    while I was troubleshooting the initial prototype build, but I'll leave it for now
    since checking if '--help' runs doesn't have any performance impact - Zack
    """

    # Import to avoid circular imports
    from app.paths import bundled_bin_dir

    # Helper function to verify that a candidate binary actually works
    def _works(p: str) -> bool:
        try:
            # Attempt to run --help and if it succeeds the binary is valid
            r = subprocess.run([p, "--help"], capture_output=True, text=True)
            return r.returncode == 0
        except Exception:
            return False

    # If the user provided an explicit path via Settings try that first
    if explicit_path:
        if _works(explicit_path):
            return explicit_path
        raise RuntimeError(f"Configured llama path does not work: {explicit_path}")

    # Directory containing bundled binaries
    bin_dir = bundled_bin_dir()

    # Prefer llama-cli (most consistent flags), but allow llama.exe/main.exe fallbacks.
    bundled_candidates = [
        bin_dir / "llama-cli.exe",
        bin_dir / "llama-cli",
        bin_dir / "llama.exe",
        bin_dir / "llama",
        bin_dir / "main.exe",
        bin_dir / "main",
        bin_dir / "llama-completion.exe",
        bin_dir / "llama-completion",
    ]

    # Check bundled binaries first
    for c in bundled_candidates:
        if c.exists() and _works(str(c)):
            return str(c)

    # If not bundled try the system PATH
    for candidate in [
        "llama-cli.exe", "llama-cli",
        "llama.exe", "llama",
        "main.exe", "main",
        "llama-completion.exe", "llama-completion",
    ]:
        if _works(candidate):
            return candidate

    # If no suitable binaries are found raise an error
    raise RuntimeError("llama.cpp binary not found. Put it in assets/bin or set Llama path in Settings.")


# ----------------------------
# Paths
# ----------------------------

def _repo_root() -> Path:
    # Determine repository root relative to this file
    # llama_util.py lives in app/ so repo root is one level above
    return Path(__file__).resolve().parents[1]


def default_summary_grammar_path() -> Path:
    # Default location of the grammar used to enforce JSON output
    return _repo_root() / "assets" / "grammars" / "summary_json.gbnf"


# ----------------------------
# Output cleanup / JSON extraction
# ----------------------------

def _extract_generated_text(raw_stdout: str) -> str:
    """
    Remove common llama.cpp CLI noise and instruction leakage.

    Even when using non-conversation mode, I ran into:
    - model loading banners
    - CLI help commands
    - instruction echoes
    - ASCII art

    This function cleans the output before it reaches the UI or JSON parser.

    # ------------------------------------ WARNING DO NOT TOUCH ------------------------------------- #
    # This was a lot of trial and error to get the generated output cleaned without any leakage       #
    # while still preserving valid output so don't touch this unless we start to see bad output start #
    # leaking again. - Zack                                                                           #
    # ------------------------------------ WARNING DO NOT TOUCH ------------------------------------- #
    """

    lines = (raw_stdout or "").splitlines()
    cleaned: list[str] = []

    # Instruction prefixes that sometimes leaked from prompts, so I gathered them all here - Zack
    bad_prefixes = (
        "use only",
        "do not",
        "keep it concise",
        "return only",
        "return",
        "output:",
        "bullet lists:",
        "rules:",
        "transcript chunk:",
        "combine the bullet lists",
        "summarize the transcript chunk",
        "bullets:",
        "paragraph:",
    )

    for line in lines:
        s = line.strip()

        # Skip empty lines
        if not s:
            continue

        # Remove interactive prompt markers
        if s.startswith(">"):
            continue

        # Remove prompt wrapper tokens if they leak into the output
        if "<s>[INST]" in s or "[/INST]" in s:
            continue

        # Remove ASCII banners if they are printed by the model
        if "██" in s or "▄▄" in s or s == "▀▀    ▀▀":
            continue

        # Remove model metadata / performance output
        if s.startswith("Loading model"):
            continue
        if s.startswith("build"):
            continue
        if s.startswith("model"):
            continue
        if s.startswith("modalities"):
            continue
        if s.startswith("available commands"):
            continue
        if s.startswith("/exit") or s.startswith("/regen") or s.startswith("/clear") or s.startswith("/read"):
            continue
        if s.startswith("[ Prompt:"):
            continue
        if s.startswith("Exiting"):
            continue
        if s.startswith("load_backend:"):
            continue

        # Remove any prompt instructions that leak into the output
        if s.lower().startswith(bad_prefixes):
            continue

        # Keep valid output
        cleaned.append(line)

    # Return the cleaned output
    return "\n".join(cleaned).strip()


def extract_last_json_object(text: str) -> str:
    """
    Extract the last valid JSON object from a text blob.

    This protects against our model if it tries to:
    - prepend explanation text
    - append extra commentary
    - partially repeat prompts

    Uses brace matching to find the final complete JSON block.

    Tried a regex first, but ran into issues passing it to the summarization model.
    JSON ended up being an elegant solution. - Zack
    """

    if not text:
        return ""

    s = text

    # Find last opening brace
    last_open = s.rfind("{")
    if last_open == -1:
        return ""

    depth = 0
    in_str = False
    esc = False

    # Walk forward tracking nested braces
    for i in range(last_open, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[last_open:i + 1]

    # If our braces never balanced return empty
    return ""


# ----------------------------
# llama runner
# ----------------------------

def _run_llama(
    llama_bin: str,
    model_path: Path,
    prompt: str,
    ctx: int = 4096,
    threads: int = 0,
    max_tokens: int = 260,
    timeout_s: int = 240,
    grammar_file: Optional[Path] = None,
    use_gpu: bool = False,
    gpu_layers: int = 0,
) -> str:
    """
    Execute llama.cpp and return cleaned text output.

    Important design decisions after some trial and error with the generated output - Zack:
    - Use temporary files instead of CLI prompt strings to avoid Windows length limits
    - Detect supported flags dynamically
    - Clean stdout to remove CLI noise
    - Support grammar-constrained decoding for structured output
    """

    # Make sure the model file exists before trying to execute
    if not model_path.exists():
        raise RuntimeError(f"LLM model not found: {model_path}")

    # Cache flag detection per binary to avoid repeated --help calls
    cache = getattr(_run_llama, "_flag_cache", {})
    cache_key = str(Path(llama_bin).resolve())
    supported = cache.get(cache_key)

    # Detect supported flags by parsing the help output
    def _detect_supported_flags() -> str:
        # Return help text
        for help_arg in ("--help", "-h"):
            try:
                r = subprocess.run(
                    [llama_bin, help_arg],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                txt = (r.stdout or "") + "\n" + (r.stderr or "")
                if txt.strip():
                    return txt.lower()
            except Exception:
                pass
        return ""

    # Perform detection once per binary
    if supported is None:
        help_txt = _detect_supported_flags()
        supported = {
            "--simple-io": ("--simple-io" in help_txt),
            "-st": ("-st" in help_txt) or ("--single-turn" in help_txt),
            "-no-cnv": ("-no-cnv" in help_txt) or ("no-cnv" in help_txt),
            "--no-display-prompt": ("--no-display-prompt" in help_txt),
            "--log-disable": ("--log-disable" in help_txt),
            "--context-shift": ("--context-shift" in help_txt),
            "--grammar-file": ("--grammar-file" in help_txt),
            "--file": ("--file" in help_txt),
        }
        cache[cache_key] = supported
        setattr(_run_llama, "_flag_cache", cache)

    # Build our base command arguments
    cmd_base = [
        llama_bin,
        "-m", str(model_path),      # model file
        "-c", str(ctx),             # context size
        "-n", str(max_tokens),      # max tokens to generate
        "--temp", "0.2",            # temperature for stability
        "--top-p", "0.9",           # nucleus sampling
    ]

    # GPU offload for CUDA-enabled llama.cpp builds
    if use_gpu and gpu_layers > 0:
        cmd_base += ["-ngl", str(gpu_layers)]

    if threads and threads > 0:
        cmd_base += ["-t", str(threads)]

    # Optional runtime flags depending on support
    if supported.get("-st", False):
        cmd_base += ["-st"]
    if supported.get("--context-shift", False):
        cmd_base += ["--context-shift"]
    if supported.get("--no-display-prompt", False):
        cmd_base += ["--no-display-prompt"]
    if supported.get("--log-disable", False):
        cmd_base += ["--log-disable"]

    # Grammar constrained decoding
    if grammar_file is not None:
        grammar_file = Path(grammar_file)
        if not grammar_file.exists():
            raise RuntimeError(f"Grammar file not found: {grammar_file}")
        if supported.get("--grammar-file", False):
            cmd_base += ["--grammar-file", str(grammar_file)]
        # If not supported, we silently skip grammar (caller can still post-parse/validate, but the grammar forces strict JSON output)

    # Helper wrapper around subprocess.run
    def _run(cmd: list[str], *, stdin_text: Optional[str] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    # Debug helper for better error messages
    def _raise_with_debug(msg: str, cmd: list[str], p: subprocess.CompletedProcess) -> None:
        stdout = p.stdout or ""
        stderr = p.stderr or ""
        raise RuntimeError(
            f"{msg}\n\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDERR (tail):\n{stderr[-4000:]}\n\n"
            f"STDOUT (tail):\n{stdout[-4000:]}\n"
        )

    # ----------------------------
    # Attempt 1: prompt via temporary file
    # ----------------------------
    used_cmd = None
    proc = None
    prompt_path: Optional[Path] = None

    try:
        if supported.get("--file", True):

            # Write prompt to temp file
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8", newline="\n") as f:
                f.write(prompt)
                prompt_path = Path(f.name)

            cmd1 = cmd_base + ["--file", str(prompt_path)]
            used_cmd = cmd1
            proc = _run(cmd1)

            # Ran into an issue where some builds silently fail to read file → fallback - Zack
            if proc.returncode == 0 and (proc.stdout or "").strip():
                pass
            else:
                # Attempt 2: stdin fallback
                # This got CLI prompt based input working on builds that didn't support
                # Prompt via file - Zack
                cmd2 = cmd_base + ["-p", ""]
                used_cmd = cmd2
                proc = _run(cmd2, stdin_text=prompt)

    finally:
        # Clean the temp file
        if prompt_path is not None:
            try:
                prompt_path.unlink(missing_ok=True)
            except Exception:
                pass

    assert proc is not None and used_cmd is not None

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Check process exit code
    if proc.returncode != 0:
        _raise_with_debug(f"llama.cpp failed (exit {proc.returncode}).", used_cmd, proc)

    # Make sure we actually received output
    if not stdout.strip():
        # Helpful error: show stderr; stdout is empty
        _raise_with_debug("llama returned no completion on STDOUT.", used_cmd, proc)

    # Clean output
    completion = _extract_generated_text(stdout)

    if not completion:
        completion = stdout.strip()

    if not completion.strip():
        _raise_with_debug("Model ran but no completion text could be extracted.", used_cmd, proc)

    return completion.strip()


# ----------------------------
# Prompts
# ----------------------------

def build_map_prompt(chunk_text: str) -> str:
    """
    Prompt used during the MAP phase of chunked summarization.
    Each transcript chunk is summarized into bullet points.
    """

    user = (
        "Summarize the transcript chunk faithfully.\n"
        "Use ONLY the information in the transcript chunk.\n"
        "Do not add facts or speculation.\n"
        "If something is unclear, say it is unclear.\n"
        "Be concise.\n\n"
        "Return 5–10 bullet points.\n\n"
        f"{chunk_text}\n"
    )
    return f"<s>[INST] {user} [/INST]"


def build_reduce_prompt(bullets_text: str) -> str:
    """
    Prompt used during the REDUCE phase.
    Combines bullet lists from all chunks into a final structured summary.
    """

    return f"""You are merging chunk summaries into one final summary.

Return JSON only with exactly this shape:
{{"bullets":["..."],"paragraph":"..."}}

Rules:
- Output valid JSON only.
- No markdown.
- No extra keys.
- bullets: 4 to 8 short strings.
- paragraph: 1 concise paragraph, max 90 words.
- Remove repetition.
- Keep only the most important facts.

Chunk summaries:
{bullets_text}
"""


# ----------------------------
# chunking + summarization
# ----------------------------

def split_text_into_chunks(
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 150,
) -> List[str]:
    """
    Split large transcript text into overlapping chunks.
    Overlap helps preserve context between adjacent segments.
    """

    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap_chars)

    return chunks


def _keep_map_lines_only(text: str) -> str:
    """
    Clean map outputs so they mostly contain bullet lines.
    Removes prompt leakage and formatting noise.
    """

    out_lines: list[str] = []

    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue

        bad_prefixes = (
            "use only", "do not", "if something", "be concise",
            "rules:", "output:", "return", "transcript chunk:",
            "<s>[inst]", "[/inst]",
        )

        if s.lower().startswith(bad_prefixes):
            continue

        # Keep bullet style lines
        if s.startswith(("-", "*")) or (len(s) >= 2 and s[0].isdigit() and s[1] == "."):
            out_lines.append(line)

        # Keep short followup lines
        else:
            if out_lines and len(s) < 120:
                out_lines.append(line)

    return "\n".join(out_lines).strip()


@dataclass
class ChunkedSummaryConfig:
    """
    Summarization tuning knobs.
    We can change these as needed to tune performance/accuracy - Zack
    """

    max_chars_per_chunk: int = 1800             # Max char per chunk
    overlap_chars: int = 150                    # Overlap between chunks to preserve context
    include_debug_chunk_section: bool = False   # Whether to append chunk summaries for debugging


def summarize_general_chunked(
    llama_bin: str,
    model_path: Path,
    transcript_text: str,
    ctx: int = 4096,
    threads: int = 0,
    config: ChunkedSummaryConfig | None = None,
    progress_cb=None,
    use_gpu: bool = False,
    gpu_layers: int = 0,
) -> str:
    """
    Summarize transcript text using a chunked map→reduce pipeline.

    1. Split the transcript into overlapping chunks.
    2. Run a map prompt on each chunk to produce bullet summaries.
    3. Combine all chunk summaries into a reduce prompt.
    4. Parse the final JSON and render a Markdown summary.

    The reduce step optionally uses a grammar file if it exists, which helps
    constrain output to a known JSON structure. Before I constrained the JSON output from the
    summarization model with a grammar file it would occasionally spit out malformed JSON which
    caused the summarization to fail. -Zack

    We can also change the way the summary output looks here. This is just the initial iteration. - Zack
    """

    # If no config object was passed in create one using default settings
    if config is None:
        config = ChunkedSummaryConfig()

    # Normalize transcript input by trimming whitespace at the beginning/end
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return "## Summary\n\n(No transcript text to summarize.)\n"

    # Split the full transcript into slightly overlapping chunks so each chunk can fit
    # Comfortably into the model context window during the map step
    chunks = split_text_into_chunks(
        transcript_text,
        max_chars=config.max_chars_per_chunk,
        overlap_chars=config.overlap_chars,
    )

    # Report basic progress information to the UI
    if progress_cb:
        progress_cb(f"Transcript chars={len(transcript_text)}; using {len(chunks)} chunks")

    # ----------------------------
    # MAP step
    # ----------------------------
    # In the map step, each chunk is summarized independently into a smaller bullet list
    # These bullet lists are later merged in the reduce step
    chunk_blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        # Report per-chunk progress to the UI
        if progress_cb:
            progress_cb(f"Summarizing chunk {i}/{len(chunks)} (chars={len(chunk)})...")

        # Build the map prompt for this chunk
        prompt = build_map_prompt(chunk)

        # Run llama.cpp on the current chunk
        # We cap the map-step context to 2048 because chunk summaries do not
        # Need the full reduce-step context size - Zack
        raw = _run_llama(
            llama_bin=llama_bin,
            model_path=model_path,
            prompt=prompt,
            ctx=min(ctx, 2048),
            threads=threads,
            max_tokens=240,
            timeout_s=240,
            grammar_file=None,          # We don't need the grammar file for the map step
            use_gpu=use_gpu,            # Because it doesn't need to output in JSON just yet
            gpu_layers=gpu_layers,
        )

        # Clean the raw map output so we keep mostly bullet style lines
        # And filter out prompt leakage or CLI noise
        bullets = _keep_map_lines_only(raw)
        if not bullets:
            bullets = "- (No bullet output for this chunk.)"

        # If the model failed to return usable bullet lines, insert a fallback
        # Placeholder so the reduce step still has something to process
        chunk_blocks.append(f"### Chunk {i}\n{bullets}\n")

    # Combine all chunk summaries into one large text block that will be fed
    # Into the reduce prompt
    bullets_text = "\n".join(chunk_blocks).strip()

    # ----------------------------
    # REDUCE step
    # ----------------------------
    # In the reduce step, the model merges all chunk summaries into one final
    # Structured summary
    if progress_cb:
        progress_cb("Combining chunk summaries...")

    # Resolve the grammar file path used to constrain JSON output
    grammar_path = default_summary_grammar_path()

    # Build the reduce prompt using all intermediate chunk bullet summaries
    reduce_prompt = build_reduce_prompt(bullets_text)

    # Run llama.cpp again, this time for the final reduce pass
    # If the grammar file exists, pass it in so the model is constrained to
    # valid JSON output (This is important to stop malformed JSON output -Zack)
    # If it does not exist, skip grammar enforcement (This is less than ideal, but shouldn't happen in practice -Zack)
    raw_reduce = _run_llama(
        llama_bin=llama_bin,
        model_path=model_path,
        prompt=reduce_prompt,
        ctx=ctx,
        threads=threads,
        max_tokens=1400,
        timeout_s=420,
        grammar_file=grammar_path if grammar_path.exists() else None,
        use_gpu=use_gpu,
        gpu_layers=gpu_layers,
    ).strip()

    # ----------------------------
    # JSON parsing
    # ----------------------------
    # Ideally the grammar makes the reduce output valid JSON every time
    # We still guard against malformed output just in case something slips
    # through or grammar is unavailable
    obj = None
    try:
        # First try to parse the entire reduce output directly as JSON
        obj = json.loads(raw_reduce)
    except Exception:
        # If that fails, try to recover the last valid-looking JSON object from the text blob
        # This helps when the model adds extra text before or after the actual JSON
        json_str = extract_last_json_object(raw_reduce)

        # If no valid JSON object can be extracted, raise a detailed error
        # Showing the tail of the raw model output for debugging
        if not json_str:
            raise RuntimeError(
                "Reduce step did not return valid JSON.\n\n"
                f"RAW REDUCE OUTPUT (tail):\n{raw_reduce[-6000:]}"
            )

        # Parse the recovered JSON object
        obj = json.loads(json_str)

    # Pull the expected fields out of the parsed JSON
    # Use safe fallbacks in case the keys are missing or empty
    final_bullets = obj.get("bullets", []) or []
    final_paragraph = obj.get("paragraph", "") or ""

    # ----------------------------
    # Markdown rendering
    # ----------------------------
    # Build the final Markdown summary that will be shown in the UI and/or
    # Saved to summary.md
    md_lines: list[str] = ["## Summary", ""]

    # Add bullet points first, limiting the displayed count to at most 12
    if final_bullets:
        for b in final_bullets[:12]:
            md_lines.append(f"- {str(b).strip()}")
        md_lines.append("")

    # Add the summary paragraph below the bullets, if one exists
    if final_paragraph:
        md_lines.append(str(final_paragraph).strip())
        md_lines.append("")

    # Optionally append the full per-chunk debug section
    # This is useful when inspecting how the reduce step was constructed during
    # Development/debugging, but should stay off in normal user-facing summaries -Zack
    if config.include_debug_chunk_section:
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(bullets_text)

    # Join everything into a single Markdown string and ensure it ends
    # With a trailing newline.
    return "\n".join(md_lines).strip() + "\n"