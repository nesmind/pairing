"""Live progress for a running stable-diffusion.cpp generation, read from sd-server's own log (its API has no
progress endpoint). sd.cpp prints a `|===>   | 3/20 - 2.41s/it` bar per sampling step, a `|###| 294/686 - 2.3GB/s`
bar while loading weights, and a few stage lines; each job remembers the log size at its start (see
ImageGenerationJob.log_offset) so only its own lines are read.

Overall progress is split by stage: loading weights 0-15%, encoding the prompt 15-25%, sampling 25-85%
(proportional to steps), decoding 85-100%. sd.cpp loads the text encoder and the VAE lazily, so a weights bar
also shows up mid-job: it fills the band of the stage it appears in instead of resetting progress."""

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import DATA_DIR

LOG_PATH = DATA_DIR / "logs" / "sdcpp.log"
_MAX_READ_BYTES = 512 * 1024
_STEP_RE = re.compile(r"(\d+)/(\d+) - ([\d.]+)(s/it|it/s)")
_LOAD_RE = re.compile(r"(\d+)/(\d+) - [\d.]+\s*[KMG]?B/s")


@dataclass(frozen=True)
class Progress:
    progress: float  # 0-100
    stage: str
    eta_seconds: int | None = None
    phase: str = "load"  # load -> encode -> sample -> decode -> done; keeps a lazy weights bar in its own band


def current_offset(path: Path = LOG_PATH) -> int:
    """The log's size right now — a job's starting point."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_progress(offset: int, path: Path = LOG_PATH) -> Progress | None:
    """Progress of the job that started at `offset`, or None if the log has nothing from it yet."""
    try:
        size = path.stat().st_size
        start = max(offset if offset <= size else 0, size - _MAX_READ_BYTES)  # a truncated log restarts at 0
        with open(path, "rb") as f:
            f.seek(start)
            text = f.read().decode(errors="replace")
    except OSError:
        return None
    state: Progress | None = None
    for line in re.split(r"[\r\n]+", text):
        state = _advance(line, state)
    return state


def _advance(line: str, state: Progress | None) -> Progress | None:
    phase = state.phase if state else "load"
    if "|" in line and (m := _LOAD_RE.search(line)):
        frac = int(m[1]) / max(int(m[2]), 1)
        bands = {"load": (0, 15, "Loading the model"), "encode": (15, 25, "Encoding the prompt")}
        bands["decode"] = (85, 90, "Decoding the image")
        if phase not in bands:
            return state
        low, high, stage = bands[phase]
        return Progress(low + (high - low) * frac, stage, None, phase)
    if "|" in line and (m := _STEP_RE.search(line)):
        step, total = int(m[1]), max(int(m[2]), 1)
        rate = float(m[3])
        seconds_per_step = rate if m[4] == "s/it" else (1 / rate if rate else 0)
        return Progress(
            25 + 60 * step / total,
            f"Sampling · step {step} of {total}",
            round((total - step) * seconds_per_step),
            "sample",
        )
    if "generate_image completed" in line:
        return Progress(100, "Finishing", None, "done")
    if "decoding" in line and "latents" in line:
        return Progress(85, "Decoding the image", None, "decode")
    if "get_learned_condition completed" in line:
        return Progress(25, "Starting to sample", None, "sample")
    if "generate_image" in line:
        return Progress(15, "Encoding the prompt", None, "encode")
    return state
