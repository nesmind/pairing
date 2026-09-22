"""
DATABASE_URL <-> Settings > System > Database form fields, plus the .env
file it's persisted in. Pure parsing/file I/O, no network or DB
connections of any kind — see app/services/db_config_service.py for the
actions (test/build-schema/save) built on top of this.

Split out from db_config_service.py purely to stay under CLAUDE.md's
file-size rule — this is still one feature, just two files.
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
    """The DATABASE_URL currently saved on disk in .env — which can
    differ from this process's own (frozen-at-startup) DATABASE_URL if a
    save happened since this process booted. Falls back to the same
    default app/config.py itself uses when the key isn't set there."""
    return read_env_value("DATABASE_URL") or _default_url()


def _write_env_database_url(url: str) -> None:
    write_env_value("DATABASE_URL", url)


def _resolve_sqlite_path(params: SqliteParams) -> Path:
    directory = Path(params.directory).expanduser()
    if not directory.is_absolute():
        directory = BASE_DIR / directory
    # safe_filename strips any directory component (../, absolute paths)
    # from the admin-supplied filename — `directory` above is themselves
    # choosing where the DB file goes, which is fine, but `filename` is
    # meant to be just a name; without this, a value like
    # "../../etc/cron.d/evil" would let it escape that chosen directory
    # entirely (see app.services.document_upload.safe_filename's own use
    # for regular uploads, same reasoning here).
    return directory / safe_filename(params.filename)


def _parse_url(url: str) -> DbConfig:
    """Raw parse — unlike get_current_config below, the password here is
    the real one. Only for internal reuse (e.g. _build_url filling in an
    unchanged password); never return this directly from a router."""
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
        # Blank password field means "keep whatever's already saved" —
        # the UI never shows a real saved password back (see
        # get_current_config), so re-entering it on every unrelated
        # change would be the only alternative.
        current = _parse_url(current_url)
        if current.db_type == "mysql" and current.mysql and current.mysql.password:
            password = current.mysql.password
        else:
            raise ValueError("A password is required.")
    # Every field here is admin-supplied and ends up interpolated
    # straight into both this DSN string and, from there, a literal line
    # in the .env file (see write_env_value) — host/database must be
    # quoted exactly like user/password already are, or a value like
    # "x\nSECRET_KEY=deadbeef" survives as a real embedded newline and
    # gets parsed back out as a second, attacker-controlled env var on
    # the next load. quote()'s default safe set already leaves ordinary
    # hostnames/db names (letters, digits, `_.-~`) untouched.
    user = quote(params.user, safe="")
    password_enc = quote(password, safe="")
    host = quote(params.host, safe="")
    database_name = quote(params.database, safe="")
    return f"mysql+pymysql://{user}:{password_enc}@{host}:{params.port}/{database_name}"


def _resolve_effective_url(update: DbConfigUpdate) -> str:
    return _build_url(update, _read_env_database_url())


def _short_error(exc: Exception) -> str:
    """SQLAlchemy wraps the real DBAPI error (e.g. pymysql's own
    "Can't connect to MySQL server...") in a verbose wrapper exception —
    `.orig` is the original, much more readable one."""
    orig = getattr(exc, "orig", None)
    message = str(orig) if orig else str(exc)
    return message.strip().splitlines()[0][:300]


def get_current_config() -> DbConfig:
    saved_url = _read_env_database_url()
    cfg = _parse_url(saved_url)
    if cfg.mysql is not None:
        cfg.has_password = bool(cfg.mysql.password)
        cfg.mysql.password = ""
    # Saving via this page always applies live (see
    # db_config_service.save_and_apply) — .env and the running engine
    # only ever drift apart if someone edits .env by hand outside the UI
    # (e.g. over SSH before a restart), which is exactly the case this
    # still catches. Compared against app.database.live_database_url
    # (kept current by switch_database on every successful live switch),
    # never app.config.DATABASE_URL — that constant is frozen at
    # whatever this process originally booted with, so comparing against
    # it would make this flag incorrectly stay stuck on True forever
    # after the very first live switch.
    cfg.restart_required = saved_url != database.live_database_url
    return cfg
