"""DATABASE_URL <-> Settings > System > Database form fields, plus the .env persistence. Pure parsing and
file I/O, no connections (actions: db_config_service.py). Split from it for the file-size rule.
"""

from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from app import database
from app.config import BASE_DIR, DATA_DIR
from app.schemas import DbConfig, DbConfigUpdate, MysqlParams, SqliteParams
from app.services.document_upload import safe_filename
from app.services.env_file import read_env_value, write_env_value


def _default_url() -> str:
    return f"sqlite:///{DATA_DIR / 'pAIring.db'}"


def _read_env_database_url() -> str:
    """The DATABASE_URL saved in .env, which can differ from this process's frozen-at-startup one after a
    save. Falls back to app/config.py's default when unset.
    """
    return read_env_value("DATABASE_URL") or _default_url()


def _write_env_database_url(url: str) -> None:
    write_env_value("DATABASE_URL", url)


def _resolve_sqlite_path(params: SqliteParams) -> Path:
    directory = Path(params.directory).expanduser()
    if not directory.is_absolute():
        directory = BASE_DIR / directory
    # safe_filename strips directory components (../, absolute paths) from the admin-supplied filename: the
    # admin chooses `directory`, but `filename` is just a name; "../../etc/cron.d/evil" would escape it
    # (see document_upload.safe_filename).
    return directory / safe_filename(params.filename)


def _parse_url(url: str) -> DbConfig:
    """Raw parse with the real password. Internal reuse only (e.g. _build_url keeping an unchanged password);
    never return it from a router (unlike get_current_config).
    """
    if url.startswith("sqlite:///"):
        path = Path(url[len("sqlite:///") :])
        return DbConfig(db_type="sqlite", sqlite=SqliteParams(directory=str(path.parent), filename=path.name))

    parts = urlsplit(url)
    return DbConfig(
        db_type="mysql",
        mysql=MysqlParams(
            host=unquote(parts.hostname or ""),
            port=parts.port or 3306,
            user=unquote(parts.username or ""),
            password=unquote(parts.password or ""),
            database=unquote(parts.path.lstrip("/")),
        ),
    )


def _build_url(update: DbConfigUpdate, current_url: str) -> str:
    if update.db_type == "sqlite":
        return f"sqlite:///{_resolve_sqlite_path(update.sqlite)}"

    params = update.mysql
    password = params.password
    if not password:
        # A blank password keeps the saved one: the UI never shows it (get_current_config).
        current = _parse_url(current_url)
        if current.db_type == "mysql" and current.mysql and current.mysql.password:
            password = current.mysql.password
        else:
            raise ValueError("A password is required.")
    # All fields are admin-supplied and end up in this DSN and in a .env line (write_env_value). host/database
    # must be quoted like user/password: "x\nSECRET_KEY=deadbeef" would survive as an embedded newline and parse
    # as a second env var. quote's safe set leaves ordinary names (letters, digits, `_.-~`) alone.
    user = quote(params.user, safe="")
    password_enc = quote(password, safe="")
    host = quote(params.host, safe="")
    database_name = quote(params.database, safe="")
    return f"mysql+pymysql://{user}:{password_enc}@{host}:{params.port}/{database_name}"


def _resolve_effective_url(update: DbConfigUpdate) -> str:
    return _build_url(update, _read_env_database_url())


def _short_error(exc: Exception) -> str:
    """SQLAlchemy wraps the DBAPI error (e.g. pymysql's "Can't connect to MySQL server...") in a verbose
    exception; `.orig` is the readable original.
    """
    orig = getattr(exc, "orig", None)
    message = str(orig) if orig else str(exc)
    return message.strip().splitlines()[0][:300]


def get_current_config() -> DbConfig:
    saved_url = _read_env_database_url()
    cfg = _parse_url(saved_url)
    if cfg.mysql is not None:
        cfg.has_password = bool(cfg.mysql.password)
        cfg.mysql.password = ""
    # Saving here applies live (db_config_service.save_and_apply), so.env and the running engine differ only
    # after a manual.env edit (e.g. over SSH before a restart), which this catches. Compared with
    # app.database.live_database_url (kept current by switch_database), not the frozen app.config.DATABASE_URL,
    # which would leave this flag stuck True after the first live switch.
    cfg.restart_required = saved_url != database.live_database_url
    return cfg
