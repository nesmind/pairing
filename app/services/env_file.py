"""
Generic read/write for one KEY=value line in .env, leaving every other
line (comments, other settings, their order) untouched. Used by
app/services/db_config_url.py — bootstrap-level settings (DATABASE_URL,
AUTO_MIGRATE) live in .env rather than the AppSetting DB table because
they're needed before, or independently of, this app's own database
connection. (app/services/ollama_process.py used to be a second consumer
for OLLAMA_NUM_PARALLEL — that setting is now AppSetting-backed instead,
alongside Ollama's other local-mode parameters, once it became an
admin-editable-without-restart part of Settings > External servers.)
"""

from app.config import BASE_DIR

ENV_PATH = BASE_DIR / ".env"


def read_env_value(key: str) -> str | None:
    if not ENV_PATH.exists():
        return None
    prefix = f"{key}="
    for line in ENV_PATH.read_text().splitlines():
        if line.strip().startswith(prefix):
            return line.strip()[len(prefix) :]
    return None


def write_env_value(key: str, value: str) -> None:
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    prefix = f"{key}="
    written = False
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = f"{prefix}{value}"
            written = True
            break
    if not written:
        lines.append(f"{prefix}{value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
