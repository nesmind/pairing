"""Unit tests for app/services/system_metrics_poller.py — the second
periodic background-loop precedent in this codebase (see its own module
docstring). Uses the shared `db` fixture; conftest.py already patches
system_metrics_poller.AsyncSessionLocal at this module."""

import asyncio
import subprocess
from datetime import UTC, datetime

import psutil
import pytest
from sqlalchemy import select

from app.models import GpuMetricSnapshot, SystemMetricSnapshot
from app.services import system_metrics_poller


class _FakeVirtualMemory:
    total = 16_000_000_000
    used = 8_000_000_000


class _FakeDiskUsage:
    total = 100_000_000_000
    used = 40_000_000_000


@pytest.mark.asyncio
async def test_poll_once_writes_a_snapshot_row(monkeypatch, db):
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 12.5)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemory())
    monkeypatch.setattr(psutil, "disk_usage", lambda _path: _FakeDiskUsage())
    monkeypatch.setattr(psutil, "getloadavg", lambda: (0.5, 0.4, 0.3))

    await system_metrics_poller._poll_once()

    rows = (await db.execute(select(SystemMetricSnapshot))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.cpu_percent == 12.5
    assert row.mem_used_bytes == 8_000_000_000
    assert row.mem_total_bytes == 16_000_000_000
    assert row.disk_used_bytes == 40_000_000_000
    assert row.disk_total_bytes == 100_000_000_000
    assert row.load_avg_1m == 0.5


@pytest.mark.asyncio
async def test_poll_once_tolerates_missing_getloadavg(monkeypatch, db):
    """getloadavg() is POSIX-only — a platform without it must not break the whole sample."""
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 1.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemory())
    monkeypatch.setattr(psutil, "disk_usage", lambda _path: _FakeDiskUsage())

    def _raise():
        raise AttributeError("no getloadavg on this platform")

    monkeypatch.setattr(psutil, "getloadavg", _raise)

    await system_metrics_poller._poll_once()

    rows = (await db.execute(select(SystemMetricSnapshot))).scalars().all()
    assert rows[0].load_avg_1m is None


@pytest.mark.asyncio
async def test_run_system_metrics_poller_survives_a_poll_failure(monkeypatch):
    calls = 0

    async def fake_poll_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient failure")
        raise SystemExit  # stop the infinite loop cleanly once the second attempt has happened

    monkeypatch.setattr(system_metrics_poller, "_poll_once", fake_poll_once)
    monkeypatch.setattr(system_metrics_poller, "POLL_INTERVAL_SECONDS", 0)

    with pytest.raises(SystemExit):
        await system_metrics_poller._run_system_metrics_poller()

    assert calls == 2


class _FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_parse_gpu_line_valid():
    polled_at = datetime.now(UTC)
    snapshot = system_metrics_poller._parse_gpu_line("0, NVIDIA GeForce RTX 4090, 45, 8192, 24576, 62", polled_at)

    assert snapshot.gpu_index == 0
    assert snapshot.name == "NVIDIA GeForce RTX 4090"
    assert snapshot.utilization_percent == 45.0
    assert snapshot.mem_used_bytes == 8192 * 1024 * 1024
    assert snapshot.mem_total_bytes == 24576 * 1024 * 1024
    assert snapshot.temperature_c == 62.0
    assert snapshot.polled_at is polled_at


def test_parse_gpu_line_handles_missing_temperature_sensor():
    """nvidia-smi reports "[N/A]" (not a number) for a GPU with no readable temperature sensor."""
    snapshot = system_metrics_poller._parse_gpu_line("1, Tesla T4, 0, 0, 16384, [N/A]", datetime.now(UTC))
    assert snapshot.temperature_c is None


def test_parse_gpu_line_rejects_malformed_lines():
    assert system_metrics_poller._parse_gpu_line("not, enough, fields", datetime.now(UTC)) is None
    assert system_metrics_poller._parse_gpu_line("0, GPU, not-a-number, 1, 2, 3", datetime.now(UTC)) is None


def test_sample_gpus_sync_parses_multiple_gpus(monkeypatch):
    csv_output = "0, RTX 4090, 45, 8192, 24576, 62\n1, RTX 4090, 12, 2048, 24576, 50\n"
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: _FakeCompletedProcess(csv_output))

    snapshots = system_metrics_poller._sample_gpus_sync()

    assert len(snapshots) == 2
    assert [s.gpu_index for s in snapshots] == [0, 1]
    # Both rows from one sample must share the exact same polled_at — see system_metrics_service.py's own
    # docstring on why "latest GPU list" relies on this.
    assert snapshots[0].polled_at == snapshots[1].polled_at


def test_sample_gpus_sync_returns_empty_list_without_nvidia_smi(monkeypatch):
    def _raise(*_a, **_kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    monkeypatch.setattr(system_metrics_poller.platform, "system", lambda: "Linux")

    assert system_metrics_poller._sample_gpus_sync() == []


# Real, ioreg-confirmed shape of one AGXAccelerator entry's "PerformanceStatistics" IOKit property (see
# _sample_apple_gpu_sync's own docstring on where "Device Utilization %" comes from and why it's parsed
# defensively rather than assumed stable).
_IOREG_SAMPLE_OUTPUT = """\
+-o AGXAccelerator  <class AGXAccelerator, id 0x100000275, registered, matched, active, busy 0, retain 9>
    {
      "IOClass" = "AGXAccelerator"
      "PerformanceStatistics" = {"Device Utilization %"=13,"Alloc system memory"=929792,"In use system memory"=929792}
    }
"""


class _FakeVirtualMemoryForApple:
    total = 17_179_869_184
    used = 9_000_000_000


def test_sample_apple_gpu_sync_parses_utilization_from_ioreg(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: _FakeCompletedProcess(_IOREG_SAMPLE_OUTPUT))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemoryForApple())

    snapshots = system_metrics_poller._sample_apple_gpu_sync()

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.gpu_index == 0
    assert snap.name == "AGXAccelerator"
    assert snap.utilization_percent == 13.0
    # No discrete VRAM on Apple Silicon (unified memory) — overall system RAM stands in for both fields.
    assert snap.mem_used_bytes == 9_000_000_000
    assert snap.mem_total_bytes == 17_179_869_184
    assert snap.temperature_c is None  # no sensor reading available without root


def test_sample_apple_gpu_sync_parses_multiple_accelerator_entries(monkeypatch):
    output = _IOREG_SAMPLE_OUTPUT + _IOREG_SAMPLE_OUTPUT.replace("AGXAccelerator", "AGXAccelerator2")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: _FakeCompletedProcess(output))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemoryForApple())

    snapshots = system_metrics_poller._sample_apple_gpu_sync()

    assert [s.gpu_index for s in snapshots] == [0, 1]
    assert [s.name for s in snapshots] == ["AGXAccelerator", "AGXAccelerator2"]


def test_sample_apple_gpu_sync_returns_empty_list_without_ioreg(monkeypatch):
    def _raise(*_a, **_kw):
        raise FileNotFoundError("ioreg not found")

    monkeypatch.setattr(subprocess, "run", _raise)

    assert system_metrics_poller._sample_apple_gpu_sync() == []


def test_sample_apple_gpu_sync_returns_empty_list_for_unparseable_output(monkeypatch):
    """A future macOS/chip generation renaming or dropping "Device Utilization %" must fail open (no GPU rows),
    never crash the poller — this is reverse-engineered, not documented, API."""
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_kw: _FakeCompletedProcess("+-o AGXAccelerator <class AGXAccelerator>\n")
    )
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemoryForApple())

    assert system_metrics_poller._sample_apple_gpu_sync() == []


def test_sample_gpus_sync_falls_back_to_apple_gpu_sampling_on_macos(monkeypatch):
    """No nvidia-smi is ever present on macOS — _sample_gpus_sync should transparently try the ioreg-based
    fallback there instead of just giving up, unlike on Linux/Windows where no GPU really means no GPU."""

    def _raise_for_nvidia_smi(argv, **_kw):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi not found")
        return _FakeCompletedProcess(_IOREG_SAMPLE_OUTPUT)

    monkeypatch.setattr(subprocess, "run", _raise_for_nvidia_smi)
    monkeypatch.setattr(system_metrics_poller.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemoryForApple())

    snapshots = system_metrics_poller._sample_gpus_sync()

    assert len(snapshots) == 1
    assert snapshots[0].name == "AGXAccelerator"


def test_sample_gpus_sync_does_not_try_apple_fallback_on_linux(monkeypatch):
    called = []

    def _raise_for_nvidia_smi(argv, **_kw):
        called.append(argv[0])
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", _raise_for_nvidia_smi)
    monkeypatch.setattr(system_metrics_poller.platform, "system", lambda: "Linux")

    assert system_metrics_poller._sample_gpus_sync() == []
    assert called == ["nvidia-smi"]  # never tried ioreg on a platform that was never going to have it


@pytest.mark.asyncio
async def test_poll_once_writes_gpu_snapshots_alongside_the_system_snapshot(monkeypatch, db):
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 1.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemory())
    monkeypatch.setattr(psutil, "disk_usage", lambda _path: _FakeDiskUsage())
    monkeypatch.setattr(psutil, "getloadavg", lambda: (0.1, 0.1, 0.1))
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_kw: _FakeCompletedProcess("0, RTX 4090, 45, 8192, 24576, 62\n")
    )

    await system_metrics_poller._poll_once()

    system_rows = (await db.execute(select(SystemMetricSnapshot))).scalars().all()
    gpu_rows = (await db.execute(select(GpuMetricSnapshot))).scalars().all()
    assert len(system_rows) == 1
    assert len(gpu_rows) == 1
    assert gpu_rows[0].name == "RTX 4090"


@pytest.mark.asyncio
async def test_poll_once_writes_no_gpu_rows_when_no_gpu_is_present(monkeypatch, db):
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 1.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemory())
    monkeypatch.setattr(psutil, "disk_usage", lambda _path: _FakeDiskUsage())
    monkeypatch.setattr(psutil, "getloadavg", lambda: (0.1, 0.1, 0.1))

    def _raise(*_a, **_kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _raise)

    await system_metrics_poller._poll_once()  # must not raise

    system_rows = (await db.execute(select(SystemMetricSnapshot))).scalars().all()
    gpu_rows = (await db.execute(select(GpuMetricSnapshot))).scalars().all()
    assert len(system_rows) == 1
    assert gpu_rows == []


@pytest.mark.asyncio
async def test_start_system_metrics_poller_tracks_its_own_task_with_a_strong_reference(monkeypatch):
    async def fake_run_forever():
        await asyncio.sleep(999)

    monkeypatch.setattr(system_metrics_poller, "_run_system_metrics_poller", fake_run_forever)

    system_metrics_poller.start_system_metrics_poller()

    assert len(system_metrics_poller._background_poller_tasks) == 1
    # Cancel *and* await, not just fire-and-forget cancel() — an unawaited cancellation can keep resolving on
    # its own schedule after this test returns, bleeding stray event-loop activity into whatever test runs next
    # in the same process (see title_service's own background-task tests for this same convention).
    tasks = list(system_metrics_poller._background_poller_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task
