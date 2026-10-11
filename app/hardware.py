"""Local hardware detection, to decide whether a catalog model can run here (app/routers/settings.py).

Checks *total* capacity, not free memory: model requirements are stated against total, and free
memory would make models flicker between available and blocked.
"""

import logging
import platform
import re
import subprocess
from pathlib import Path

import psutil

logger = logging.getLogger("llama_chat")


def get_system_ram_gb() -> float:
    """Total RAM in decimal GB."""
    return psutil.virtual_memory().total / 1_000_000_000


def get_gpu_vram_gb() -> float:
    """Total NVIDIA VRAM in GB, or 0.0 if none or undetectable (best-effort: models run without a GPU)."""
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
    return total_mb / 1000  # MiB-ish values; close enough for gating


def available_capacity_gb() -> float:
    """RAM plus any VRAM: the memory pool model weights must fit in."""
    return get_system_ram_gb() + get_gpu_vram_gb()


def has_avx2() -> bool:
    """Whether the CPU supports AVX2. Prebuilt wheels (numpy, torch, ComfyUI's kornia) assume it; without
    it they crash with SIGILL (seen on a 2011 CPU), so comfyui_installer gates on this before a big download.
    Reads /proc/cpuinfo (Linux only); fails open (True) if unreadable.
    """
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return True
    return "avx2" in cpuinfo


def local_install_supported() -> bool:
    """Whether the auto-installers (app.services.*_installer, "Install from GitHub") can run here.
    They assume Linux: a Linux-only pinned release asset, plus scripts/start.sh and the systemd unit.
    Local mode still works elsewhere with a hand-installed engine ("Enter its path manually").
    """
    return platform.system() == "Linux"


def hardware_summary() -> dict:
    """Snapshot of this machine's capacity, for gating and for display in Settings."""
    ram_gb = get_system_ram_gb()
    vram_gb = get_gpu_vram_gb()
    return {
        "ram_gb": round(ram_gb, 1),
        "vram_gb": round(vram_gb, 1),
        "total_gb": round(ram_gb + vram_gb, 1),
    }
