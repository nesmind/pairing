"""
Local hardware detection, used to decide whether a given model in
app.model_catalog can realistically run on this machine before letting
someone select or download it (see app/routers/settings.py).

Deliberately checks *total* capacity rather than momentarily-free
memory: model card requirements ("needs 16GB RAM") are stated against
total system memory, and gating on free memory would make a model
flicker between available/blocked as unrelated processes use RAM.
"""

import logging
import platform
import re
import subprocess
from pathlib import Path

import psutil

logger = logging.getLogger("llama_chat")


def get_system_ram_gb() -> float:
    """Total physical RAM on this machine, in gigabytes (decimal GB —
    matches how RAM is marketed/specified, e.g. "16GB")."""
    return psutil.virtual_memory().total / 1_000_000_000


def get_gpu_vram_gb() -> float:
    """Total VRAM across any NVIDIA GPU(s), in gigabytes, or 0.0 if none
    is present/detectable. Best-effort: this app runs chat models
    through Ollama regardless of whether a GPU exists, so a missing or
    unparseable `nvidia-smi` just means "assume no GPU acceleration"
    rather than an error.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0.0

    total_mb = 0.0
    for line in output.strip().splitlines():
        match = re.match(r"\s*([\d.]+)", line)
        if match:
            total_mb += float(match.group(1))
    return total_mb / 1000  # nvidia-smi reports MiB-ish values; close enough for a gating check


def available_capacity_gb() -> float:
    """Total RAM plus any GPU VRAM — the pool of memory a model's
    weights need to fit into (with Ollama's usual CPU/GPU offloading)."""
    return get_system_ram_gb() + get_gpu_vram_gb()


def has_avx2() -> bool:
    """Whether this CPU supports AVX2 — the de facto baseline modern
    prebuilt PyPI wheels (numpy, torch, and anything depending on them,
    like ComfyUI's own kornia dependency) are compiled assuming. Missing
    it isn't a "runs slower" situation the way a missing GPU is: a CPU
    without AVX2 executing an AVX2 instruction hard-crashes with SIGILL
    ("Illegal instruction") the moment such code actually runs — confirmed
    live on a 2011-era CPU here (see app.services.comfyui_installer's own
    use of this, gating its install before wasting a multi-GB download/
    pip install on a machine that would just crash once ComfyUI starts).
    Checked via /proc/cpuinfo's own flags line — this app already only
    targets Linux (see scripts/start.sh). Fails open (True) if the file
    can't be read at all, rather than blocking an install over an
    inconclusive check."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return True
    return "avx2" in cpuinfo


def local_install_supported() -> bool:
    """Whether this app's own Ollama/ComfyUI auto-installers (see
    app.services.ollama_installer/comfyui_installer, both behind
    Settings > External servers' "Install from GitHub" button) can run
    on this machine at all. Both assume a Linux host — Ollama's pinned
    release asset is a Linux-only build (ollama-linux-{arch}.tar.zst,
    no macOS/Windows equivalent this app fetches), and this app's own
    process-management/deployment assumptions throughout (scripts/
    start.sh, scripts/install_on_fresh_server.sh's systemd unit) are
    Linux-only too. Exact, not a guess: platform.system() reports the
    real OS this process is running on. "Local" mode itself still works
    fine on any OS if Ollama/ComfyUI is installed by hand elsewhere and
    pointed at via "Enter its path manually" — only the auto-download
    is gated by this."""
    return platform.system() == "Linux"


def hardware_summary() -> dict:
    """A snapshot of this machine's capacity, for both the gating check
    and for display in Settings (e.g. "you have 16GB RAM, no GPU
    detected")."""
    ram_gb = get_system_ram_gb()
    vram_gb = get_gpu_vram_gb()
    return {
        "ram_gb": round(ram_gb, 1),
        "vram_gb": round(vram_gb, 1),
        "total_gb": round(ram_gb + vram_gb, 1),
    }
