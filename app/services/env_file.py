"""
Generic read/write for one or more KEY=value lines in a .env file, leaving every other line
(comments, other settings, their order) untouched. Used by app/services/db_config_url.py —
bootstrap-level settings (DATABASE_URL, AUTO_MIGRATE) live in this app's own .env rather than the
AppSetting DB table because they're needed before, or independently of, this app's own database
connection — and by app/services/matricxon_process.py, to write MATRICXON_* overrides into
Matricxon's own .env file (a different path — see write_env_values' own docstring on why that one
needs to work regardless of which path started the Matricxon process).
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
    """Same single-key guarantee write_env_value already gave, extended to write several keys in one
    read/modify/write pass rather than one file rewrite per key. A `None` value removes that key's
    line entirely instead of writing an empty `KEY=` one — used when a config field is cleared back
    to "use the engine's own built-in default": leaving a stale `KEY=` line behind would keep
    overriding it even after the admin unset it."""
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
