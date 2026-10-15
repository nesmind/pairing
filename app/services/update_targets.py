"""Admin settings > Update: pAIring and a local Matricxon as update targets, and how each restarts afterward."""

import re
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import (
    BASE_DIR,
    IS_PRIMARY,
    MATRICXON_DEFAULT_VERSION,
    MATRICXON_GITHUB_REPO,
    PAIRING_DEFAULT_BRANCH,
    PAIRING_GITHUB_REPO,
)
from app.schemas import MatricxonServerConfig
from app.schemas.update import PairingRestartInfo, SourceUpdateStatus
from app.services import matricxon_process
from app.services.source_updater import SourceUpdater, UpdateTarget

# Never overwritten by a download update: user data, environments, local settings.
_PAIRING_KEEP = frozenset(
    {"data", "models", "attachments", "avatars", "generated_images", "knowledge", "external", "run", ".venv", ".env"}
)
_MATRICXON_KEEP = frozenset({"data", "logs", "run", ".venv", ".env"})


class PairingUpdate:
    def __init__(self, proxy_url: str | None = None, base_dir: Path = BASE_DIR) -> None:
        self.base_dir = base_dir
        target = UpdateTarget(
            "pAIring", base_dir, PAIRING_GITHUB_REPO, PAIRING_DEFAULT_BRANCH, Path(sys.executable), _PAIRING_KEEP
        )
        self.updater = SourceUpdater(target, proxy_url)

    def in_container(self) -> bool:
        return Path("/.dockerenv").exists()

    async def status(self) -> SourceUpdateStatus:
        if self.in_container():
            return SourceUpdateStatus(
                name="pAIring", note="Running in a container: update by pulling a new image instead."
            )
        return await self.updater.status()

    @staticmethod
    def systemd_service() -> str | None:
        """The system service running us, from our cgroup (a desktop terminal sits under user.slice instead)."""
        try:
            cgroup = Path("/proc/self/cgroup").read_text()
        except OSError:
            return None
        match = re.search(r"/system\.slice/(?:.*/)?([^/]+)\.service$", cgroup, re.MULTILINE)
        return match[1] if match else None

    def restart_info(self) -> PairingRestartInfo:
        if self.in_container():
            return PairingRestartInfo(mode="docker", can_restart=False, hint="Restart the container.")
        if service := self.systemd_service():
            return PairingRestartInfo(mode="systemd", can_restart=False, hint=f"sudo systemctl restart {service}")
        if (self.base_dir / "scripts" / "restart.sh").is_file():
            return PairingRestartInfo(mode="script", can_restart=True, hint="bash scripts/restart.sh")
        return PairingRestartInfo(mode="manual", can_restart=False, hint="Restart pAIring the way you started it.")

    def restart(self) -> None:
        """Runs scripts/restart.sh detached, so it outlives this process."""
        if not self.restart_info().can_restart:
            raise ValueError("pAIring can't restart itself here; please restart it manually.")
        log_dir = self.base_dir / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "restart.log", "ab") as log:
            subprocess.Popen(
                ["bash", str(self.base_dir / "scripts" / "restart.sh")],
                cwd=self.base_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )


class MatricxonUpdate:
    def __init__(self, config: MatricxonServerConfig, proxy_url: str | None = None) -> None:
        self.config = config
        self.proxy_url = proxy_url
        self.project_dir = matricxon_process.resolve_project_dir(config.project_dir) if config.mode == "local" else None
        self.updater: SourceUpdater | None = None
        if self.project_dir is not None:
            target = UpdateTarget(
                "Matricxon",
                self.project_dir,
                config.install_repo or MATRICXON_GITHUB_REPO,
                config.install_version or MATRICXON_DEFAULT_VERSION,
                self.project_dir / ".venv" / "bin" / "python",
                _MATRICXON_KEEP,
            )
            self.updater = SourceUpdater(target, proxy_url)
            self.updater.total_steps = 4

    async def status(self) -> SourceUpdateStatus | None:
        return await self.updater.status() if self.updater else None

    async def update_stream(self) -> AsyncIterator[dict]:
        """The source update, then a restart if Matricxon was running."""
        if self.updater is None:
            yield {"error": "Matricxon is not installed locally."}
            return
        was_running = (await matricxon_process.get_status(self.config.project_dir)).running
        done = None
        async for event in self.updater.update_stream():
            if event.get("done"):
                done = event
                continue
            yield event
            if "error" in event:
                return
        meta = {"step": 4, "total_steps": 4, "step_label": "Restarting Matricxon"}
        if was_running and done and done.get("changed"):
            async for event in self._restart(meta):
                yield event
        else:
            yield {**meta, "status": "No restart needed."}
        yield {**(done or {"done": True}), "restarted": was_running}

    async def _restart(self, meta: dict) -> AsyncIterator[dict]:
        if not IS_PRIMARY:
            yield {**meta, "status": "Please restart Matricxon from the primary instance to load the update."}
            return
        yield {**meta, "status": "Restarting Matricxon..."}
        try:
            await matricxon_process.stop(self.config.project_dir)
            await matricxon_process.start(self.config, proxy_url=self.proxy_url)
            yield {**meta, "status": "Matricxon is running the new version."}
        except (RuntimeError, ValueError) as exc:
            yield {**meta, "status": f"Updated, but could not restart Matricxon: {exc}"}
