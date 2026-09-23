"""Real download progress for an installer's `pip install` step. pip prints no incremental progress when
its output isn't a terminal (an installer reads it through a pipe), so a multi-minute PyTorch download used to
show as one "Downloading torch-...whl (899.7 MB)" line and then nothing - a full-looking progress bar that read
as stuck (reported live, 2026-09-23). `pip install --progress-bar raw` (pip 24.1+) instead prints
`Progress <bytes> of <total>` about four times a second, which this turns into the same `completed`/`total`
events the installers' git-clone step already sends (see app/static/js/settings.js's installServer).
"""

import asyncio
import re
from pathlib import Path


class PipProgressTracker:
    _PROGRESS = re.compile(r"^Progress (\d+) of (\d+)$")
    # "Downloading" (a real download, followed by Progress lines) or "Using cached" (pip's own HTTP cache -
    # nothing to download, so no Progress lines at all; common on a reinstall).
    _DOWNLOADING = re.compile(r"^\s*(Downloading|Using cached) (\S+)(?: \(([^)]+)\))?")

    def __init__(self, base_label: str) -> None:
        self._base_label = base_label
        self._activity: str | None = None

    @staticmethod
    async def supports_raw(python: Path) -> bool:
        """Whether this venv's pip knows `--progress-bar raw` - an older bundled pip (e.g. Python 3.12's
        24.0) rejects it outright, which would fail the whole install."""
        process = await asyncio.create_subprocess_exec(
            str(python),
            "-m",
            "pip",
            "install",
            "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate()
        return b"raw" in output

    def _label(self) -> str:
        return f"{self._base_label} — {self._activity}" if self._activity else self._base_label

    def event_for(self, line: str) -> dict:
        """One installer event for one line of pip output. Progress lines carry no `status` - they'd flood
        the visible log at four lines a second."""
        progress = self._PROGRESS.match(line.strip())
        if progress:
            completed, total = int(progress.group(1)), int(progress.group(2))
            event: dict = {"step_label": self._label()}
            if total:
                event.update(completed=completed, total=total, unit="bytes")
            return event
        downloading = self._DOWNLOADING.match(line)
        # pip first fetches each wheel's small `.whl.metadata` (tens of kB) while resolving - log it, but don't
        # let it replace the label with a meaningless "(40 kB)" download.
        if downloading and not downloading.group(2).endswith(".metadata"):
            verb = "downloading" if downloading.group(1) == "Downloading" else "using cached"
            package = downloading.group(2).split("-")[0]
            size = f" ({downloading.group(3)})" if downloading.group(3) else ""
            self._activity = f"{verb} {package}{size}"
        elif line.startswith("Installing collected packages"):
            self._activity = "installing packages (no progress available, can take a minute)"
        return {"step_label": self._label(), "status": line}
