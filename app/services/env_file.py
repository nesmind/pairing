"""Reads/writes KEY=value lines in a .env file, leaving other lines (comments, settings, order) untouched.
Used by db_config_url.py (bootstrap settings DATABASE_URL, AUTO_MIGRATE live in .env, not the AppSetting
table, since they're needed before or independent of the DB) and matricxon_process.py (MATRICXON_*
overrides in Matricxon's own .env, which must work however the process starts; see write_env_values).
"""

from pathlib import Path

from app.config import BASE_DIR

ENV_PATH = BASE_DIR / ".env"


def read_env_value(key: str, path: Path | None = None) -> str | None:
    path = path or ENV_PATH
    if not path.exists():
        return None
    prefix = f"{key}="
    for line in path.read_text().splitlines():
        if line.strip().startswith(prefix):
            return line.strip()[len(prefix) :]
    return None


def write_env_value(key: str, value: str, path: Path | None = None) -> None:
    write_env_values({key: value}, path)


def write_env_values(values: dict[str, str | None], path: Path | None = None) -> None:
    """Like write_env_value but several keys in one read/modify/write pass. A `None` value removes the line
    instead of writing `KEY=`: used when a field is cleared to the engine's default, since a stale `KEY=` would still
    override it.
    """
    path = path or ENV_PATH
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(values)
    result: list[str] = []
    for line in lines:
        matched_key = next((key for key in remaining if line.strip().startswith(f"{key}=")), None)
        if matched_key is None:
            result.append(line)
            continue
        value = remaining.pop(matched_key)
        if value is not None:
            result.append(f"{matched_key}={value}")
    for key, value in remaining.items():
        if value is not None:
            result.append(f"{key}={value}")
    path.write_text("\n".join(result) + "\n" if result else "")
