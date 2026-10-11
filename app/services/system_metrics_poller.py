"""Samples this machine's CPU/RAM/disk usage via psutil (already a dependency; app/hardware.py's static checks
read total RAM/VRAM once for model gating, this samples live usage on a timer), plus best-effort per-GPU
usage: `nvidia-smi` on Linux (as app.hardware.get_gpu_vram_gb, extended to utilization, memory and
temperature per poll) or `ioreg`'s IOAccelerator as an Apple Silicon fallback (_sample_apple_gpu_sync). Appends
one SystemMetricSnapshot plus zero or more GpuMetricSnapshot rows (one per detected GPU; zero is no error, as in
app.hardware); read by system_metrics_service for the Stats "System" view.

The second periodic loop (first: ollama_ps_poller, see its docstring for the detached-task/strong-ref
pattern). Primary-gated for the same reason: CPU/RAM/disk/GPU belong to the machine all instances share, so
polling from each would only duplicate rows.
"""

import asyncio
import logging
import platform
import re
import subprocess
from datetime import UTC, datetime

import psutil

from app.config import BASE_DIR
from app.database import AsyncSessionLocal
from app.models import GpuMetricSnapshot, SystemMetricSnapshot

logger = logging.getLogger("llama_chat")

POLL_INTERVAL_SECONDS = 15.0
_GPU_QUERY_FIELDS = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
_MIB_TO_BYTES = 1024 * 1024

_background_poller_tasks: set[asyncio.Task] = set()


def _sample() -> SystemMetricSnapshot:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(BASE_DIR))
    try:
        load_avg_1m = psutil.getloadavg()[0]
    except (AttributeError, OSError):
        # getloadavg is POSIX-only (this app targets Linux, app/hardware.py); failing open with "unknown" is cheap
        # insurance.
        load_avg_1m = None
    return SystemMetricSnapshot(
        cpu_percent=psutil.cpu_percent(interval=None),
        mem_used_bytes=mem.used,
        mem_total_bytes=mem.total,
        disk_used_bytes=disk.used,
        disk_total_bytes=disk.total,
        load_avg_1m=load_avg_1m,
        polled_at=datetime.now(UTC),
    )


def _parse_gpu_line(line: str, polled_at: datetime) -> GpuMetricSnapshot | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None
    index, name, util, mem_used_mib, mem_total_mib, temp = parts
    try:
        return GpuMetricSnapshot(
            gpu_index=int(index),
            name=name,
            utilization_percent=float(util),
            mem_used_bytes=int(float(mem_used_mib) * _MIB_TO_BYTES),
            mem_total_bytes=int(float(mem_total_mib) * _MIB_TO_BYTES),
            # A GPU whose temperature nvidia-smi can't read reports "N/A", not a number.
            temperature_c=float(temp) if temp.replace(".", "", 1).isdigit() else None,
            polled_at=polled_at,
        )
    except ValueError:
        # One malformed line (driver quirk, extra column) shouldn't drop every other GPU.
        return None


_IOREG_ACCELERATOR_HEADER_RE = re.compile(r"^\s*\+-o (\w+)")
_IOREG_UTILIZATION_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')


def _sample_apple_gpu_sync() -> list[GpuMetricSnapshot]:
    """Best-effort Apple Silicon GPU sampling via `ioreg`'s IOAccelerator entries, the closest non-root equivalent
    to nvidia-smi on macOS. The live per-poll number (`powermetrics` needs sudo) is each accelerator's
    "PerformanceStatistics" property, key "Device Utilization %". That isn't documented API: the key names come
    from reverse-engineering (asitop, Stats.app) and have shifted across macOS/chip generations, so parsing is
    defensive: non-matching lines are skipped, and absent or unparseable output returns [] like no GPU (fails
    open like _sample_gpus_sync's missing-nvidia-smi case), never a crash or wrong reading.

    Apple Silicon has no discrete VRAM (unified memory), so there is no GPU-specific mem_used_bytes/mem_total_bytes
    (both NOT NULL in this row, shaped for nvidia-smi's distinct VRAM). Overall system memory is used, the most
    honest number available; temperature_c stays None (no sensor reading without root).
    """
    try:
        output = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    mem = psutil.virtual_memory()
    polled_at = datetime.now(UTC)
    snapshots = []
    current_name: str | None = None
    for line in output.splitlines():
        header_match = _IOREG_ACCELERATOR_HEADER_RE.match(line)
        if header_match:
            current_name = header_match.group(1)
            continue
        util_match = _IOREG_UTILIZATION_RE.search(line)
        if util_match and current_name:
            snapshots.append(
                GpuMetricSnapshot(
                    gpu_index=len(snapshots),
                    name=current_name,
                    utilization_percent=float(util_match.group(1)),
                    mem_used_bytes=mem.used,
                    mem_total_bytes=mem.total,
                    temperature_c=None,
                    polled_at=polled_at,
                )
            )
            current_name = None
    return snapshots


def _sample_gpus_sync() -> list[GpuMetricSnapshot]:
    """Blocking (subprocess): run via asyncio.to_thread, never directly on the event loop. Unlike
    app.hardware.get_gpu_vram_gb's one-off per-request use, this runs on a tight timer for the whole process
    lifetime, and blocking the loop that serves chat streaming every poll is worth avoiding though nvidia-smi is
    normally fast.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_GPU_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        # No NVIDIA GPU/driver: not an error (app.hardware.get_gpu_vram_gb). macOS never has nvidia-smi, so this is its
        # normal
        # path: try the Apple Silicon fallback before giving up.
        if platform.system() == "Darwin":
            return _sample_apple_gpu_sync()
        return []

    polled_at = datetime.now(UTC)
    return [s for line in output.strip().splitlines() if (s := _parse_gpu_line(line, polled_at)) is not None]


async def _poll_once() -> None:
    # psutil.cpu_percent(interval=None) and the disk/mem calls are fast, non-blocking syscalls (unlike a blocking
    # interval=N cpu_percent), so no worker thread. The GPU sample does need one (_sample_gpus_sync).
    snapshot = _sample()
    gpu_snapshots = await asyncio.to_thread(_sample_gpus_sync)
    async with AsyncSessionLocal() as db:
        db.add(snapshot)
        db.add_all(gpu_snapshots)
        await db.commit()


async def _run_system_metrics_poller() -> None:
    while True:
        try:
            await _poll_once()
        except Exception:
            # A bad poll (e.g. a transient DB error) must never kill the loop for good.
            logger.exception("System metrics poll failed.")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start_system_metrics_poller() -> None:
    """Called once from startup_service.run_startup_tasks (primary only); see module docstring."""
    task = asyncio.create_task(_run_system_metrics_poller())
    _background_poller_tasks.add(task)
    task.add_done_callback(_background_poller_tasks.discard)
