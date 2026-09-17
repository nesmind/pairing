"""Periodically samples this machine's own CPU/RAM/disk usage via
psutil (already a dependency — see app/hardware.py's static-capacity
checks, which this is not: those check total RAM/VRAM once for model-
gating, this samples live usage on a timer) plus, best-effort, live
per-GPU usage: `nvidia-smi` on Linux (the same subprocess-based
detection app.hardware.get_gpu_vram_gb already uses for a one-off
total-VRAM check, extended here to a fuller per-poll query —
utilization, memory used, temperature — and run on a timer instead of
once), or `ioreg`'s IOAccelerator registry as a macOS/Apple Silicon
fallback (see _sample_apple_gpu_sync's own docstring — no nvidia-smi
equivalent exists there). Appends one SystemMetricSnapshot row plus
zero-or-more GpuMetricSnapshot rows (one per detected GPU, zero on any
machine neither path finds a GPU on — not an error, same as
app.hardware's own convention) — see app.services.system_metrics_service
for how the Stats page's "System" view reads these back.

Second periodic-loop precedent in this codebase (see
app.services.ollama_ps_poller for the first, and its own module
docstring for why this pattern — detached asyncio.Task, strong-ref set —
is new territory here). Primary-gated the same way and for the same
reason: CPU/RAM/disk/GPU are properties of the physical machine this
app's own primary and any local sibling instances (see
app.services.instance_service) all share, not something the notion of
"which instance" makes distinct — polling from every instance would
only duplicate rows.
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
        # getloadavg() is POSIX-only — this app only targets Linux (see app/hardware.py's own note on that), but
        # failing open with "unknown" here is cheap insurance against a platform that doesn't have it.
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
            # A GPU with no temperature sensor nvidia-smi can read reports "N/A" here rather than a number.
            temperature_c=float(temp) if temp.replace(".", "", 1).isdigit() else None,
            polled_at=polled_at,
        )
    except ValueError:
        # A single malformed line (a driver quirk, an unexpected extra column) shouldn't drop every other GPU.
        return None


_IOREG_ACCELERATOR_HEADER_RE = re.compile(r"^\s*\+-o (\w+)")
_IOREG_UTILIZATION_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')


def _sample_apple_gpu_sync() -> list[GpuMetricSnapshot]:
    """Best-effort Apple Silicon GPU sampling via `ioreg`'s IOAccelerator registry entries — the closest non-root
    equivalent to nvidia-smi macOS has. The actually-live per-poll number Apple exposes without root
    (`powermetrics` needs sudo; this doesn't) lives inside each accelerator's own "PerformanceStatistics" IOKit
    property, under a "Device Utilization %" key. This isn't documented/stable API — the key names are known only
    from reverse-engineering by tools like asitop/Stats.app, and have shifted across macOS/chip generations in
    the past — so this is parsed defensively: any line that doesn't match the expected shape is just skipped, and
    a totally absent/unparseable ioreg output returns [] the same as no GPU found at all (same "fails open"
    convention as _sample_gpus_sync's own missing-nvidia-smi case), never a crash or a stale/wrong reading.

    Apple Silicon has no discrete VRAM — the GPU shares the same physical pool as the CPU ("unified memory"), so
    there's no GPU-specific figure to report for mem_used_bytes/mem_total_bytes (this DB row's schema is shaped
    for nvidia-smi's real distinct-VRAM case and has a NOT NULL constraint on both). Overall system memory is
    used instead — the most honest number this architecture actually has — and temperature_c is left None, since
    no sensor reading is available without root either."""
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
    """Blocking (subprocess) by nature — always run this via
    asyncio.to_thread, never awaited/called directly from the event
    loop, unlike app.hardware.get_gpu_vram_gb's own one-off, per-request
    use of the same command: this runs on a tight timer for the
    process's entire lifetime, so blocking the loop that also serves
    live chat streaming every single poll is worth avoiding here even
    though nvidia-smi itself is normally fast."""
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_GPU_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        # No NVIDIA GPU/driver present — not an error, see app.hardware.get_gpu_vram_gb's own docstring. macOS
        # never has nvidia-smi at all, so this is also the normal path there — try the Apple Silicon fallback
        # before giving up entirely.
        if platform.system() == "Darwin":
            return _sample_apple_gpu_sync()
        return []

    polled_at = datetime.now(UTC)
    return [s for line in output.strip().splitlines() if (s := _parse_gpu_line(line, polled_at)) is not None]


async def _poll_once() -> None:
    # psutil.cpu_percent(interval=None) and the disk/mem calls are all fast, non-blocking syscalls — no need to
    # offload to a worker thread the way a blocking interval=N cpu_percent() call would require. The GPU sample
    # does need offloading — see _sample_gpus_sync's own docstring.
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
            # A single bad poll (e.g. a transient DB error) must never kill the loop permanently.
            logger.exception("System metrics poll failed.")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start_system_metrics_poller() -> None:
    """Called once, from startup_service.run_startup_tasks (primary
    only) — see this module's own docstring for why."""
    task = asyncio.create_task(_run_system_metrics_poller())
    _background_poller_tasks.add(task)
    task.add_done_callback(_background_poller_tasks.discard)
