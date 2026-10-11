"""Admin-configurable database backend (Settings > System > Database): test a candidate connection, build its
schema (plus starter data), and save, switching the live app without a restart. URL parsing/.env:
db_config_url.py; schema compatibility: db_schema_check.py (split for the file-size rule); this file holds
the three actions.

Test/build-schema use a throwaway *sync* engine like app/database.py's `_run_migrations`. Saving ends with
the live engine on the new database (app.database.switch_database): schema/seed runs first on a throwaway
*async* session, then the same URL goes to switch_database.
"""

import asyncio
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import APP_NAME, BASE_DIR
from app.database import _async_url, _run_migrations, switch_database
from app.schemas import DbActionResult, DbConfigUpdate, SqliteParams
from app.services.auth_service import seed_default_users
from app.services.db_config_url import (
    _resolve_effective_url,
    _resolve_sqlite_path,
    _short_error,
    _write_env_database_url,
)
from app.services.db_schema_check import IncompatibleSchemaError, classify_schema, connection_result_with_schema_status
from app.services.instance_db_broadcast import broadcast_database_switch
from app.services.note_service import seed_default_notes

# Reused by every "a driver is missing" message.
_PIP_INSTALL_HINT = ".venv/bin/pip install -r requirements.txt"


async def _seed_defaults(url: str) -> None:
    """Brings a database to a fresh install's starting point: seed accounts (auth_service.seed_default_users)
    and each account's default persona/rules/skill notes (note_service.seed_default_notes).

    Reuses those functions rather than a SQL fixture so the note content has one definition and cannot drift.
    Runs through the ORM on a throwaway async engine, independent of the live engine and the sync one Alembic
    used for DDL in build_schema/save_and_apply.
    """
    seed_engine = create_async_engine(_async_url(url))
    try:
        session_factory = async_sessionmaker(seed_engine, expire_on_commit=False)
        async with session_factory() as db:
            admin_user = await seed_default_users(db)
            await seed_default_notes(db, admin_user)
            await db.commit()
    finally:
        await seed_engine.dispose()


async def _ensure_schema_and_seed(url: str, update: DbConfigUpdate) -> None:
    """The shared check, DDL and seed sequence of build_schema and save_and_apply: confirm existing tables look
    like pAIring's schema (IncompatibleSchemaError otherwise, before touching anything;
    db_schema_check.classify_schema),
    create/upgrade the schema (no-op if current), then seed the starter data (idempotent, see _seed_defaults).
    """
    if update.db_type == "sqlite":
        _resolve_sqlite_path(update.sqlite).parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(classify_schema, url)
    await asyncio.to_thread(_run_migrations, url)
    await _seed_defaults(url)


async def save_and_apply(update: DbConfigUpdate) -> DbActionResult:
    """Saves and switches the live app over, no restart. The schema/seed step and live switch must both succeed
    before .env is touched, so a failure leaves the app and .env as they were.

    Side effect: every logged-in session, including the admin's, stops working. The cookie holds only a user id
    (routers/auth.py `request.session["user_id"]`) from the *old* database; the new one has its own seeded
    admin with a different id, so the next request 401s and app.js's `api` redirects to /login. Not a bug:
    the caller must log in again (see the message below).
    """
    try:
        url = _resolve_effective_url(update)
    except ValueError as exc:
        return DbActionResult(ok=False, message=str(exc))

    try:
        await _ensure_schema_and_seed(url, update)
        await switch_database(url)
    except IncompatibleSchemaError as exc:
        return DbActionResult(ok=False, message=f"{exc} Nothing changed.")
    except ModuleNotFoundError as exc:
        return DbActionResult(
            ok=False,
            message=f"Driver not installed ({exc.name}). Run: {_PIP_INSTALL_HINT}, then try again. Nothing changed.",
        )
    except SQLAlchemyError as exc:
        return DbActionResult(ok=False, message=f"Couldn't switch databases: {_short_error(exc)}. Nothing changed.")
    except Exception as exc:
        # Deliberately broad, as a last resort: this action must report something clean to the admin; an unexpected
        # failure (e.g. an Alembic error on a not-quite-compatible schema) must not surface as a raw 500.
        return DbActionResult(ok=False, message=f"Couldn't switch databases: {exc}. Nothing changed.")

    _write_env_database_url(url)
    # This process is already switched; every *other* local instance (separate process, see
    # instance_db_broadcast) must be told or it keeps the old database. Best-effort: ok=True even if one didn't
    # answer, since the switch that matters (this process) succeeded.
    failures = await broadcast_database_switch(url)
    message = f"Saved and switched — {APP_NAME} is now running on this database. You'll need to log in again."
    if failures:
        message += " Some other local instances didn't switch (" + "; ".join(failures) + ") — restart them."
    return DbActionResult(ok=True, message=message)


def _test_sqlite(params: SqliteParams) -> DbActionResult:
    directory = Path(params.directory).expanduser()
    if not directory.is_absolute():
        directory = BASE_DIR / directory
    path = directory / params.filename
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return DbActionResult(ok=False, message=f"Can't create directory: {exc}")
    if not os.access(directory, os.W_OK):
        return DbActionResult(ok=False, message=f"Directory isn't writable: {directory}")
    if not path.exists():
        return DbActionResult(ok=True, message=f"Directory is writable — {path.name} will be created here.")
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as conn:
            # Not "SELECT 1": it never reads the file's page structure and "succeeds" on a non-SQLite file. Querying
            # sqlite_master forces SQLite to read the header/schema page, which fails on a bad file.
            conn.execute(text("SELECT count(*) FROM sqlite_master"))
    except SQLAlchemyError as exc:
        return DbActionResult(ok=False, message=f"File exists but isn't a valid SQLite database: {_short_error(exc)}")
    finally:
        engine.dispose()
    return connection_result_with_schema_status(f"sqlite:///{path}", f"Connected to the existing database at {path}.")


def _missing_async_driver_warning() -> str | None:
    """Save's live switch needs `aiomysql`, not just the `pymysql` used by Test connection / Build schema. Both are
    in requirements.txt, but this defensively catches venvs missing it (older or hand-resolved), where Test
    and Build would succeed and Save fail. A plain import check.
    """
    import importlib.util

    if importlib.util.find_spec("aiomysql") is None:
        return f"Save will still fail: the async MySQL driver isn't installed. Run: {_PIP_INSTALL_HINT}"
    return None


def _test_mysql(url: str) -> DbActionResult:
    engine = None
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except ModuleNotFoundError as exc:
        return DbActionResult(
            ok=False, message=f"MySQL driver not installed ({exc.name}). Run: {_PIP_INSTALL_HINT}, then restart."
        )
    except SQLAlchemyError as exc:
        return DbActionResult(ok=False, message=_short_error(exc))
    finally:
        if engine is not None:
            engine.dispose()

    result = connection_result_with_schema_status(url, "Connected successfully.")
    if result.ok:
        warning = _missing_async_driver_warning()
        if warning:
            return DbActionResult(ok=False, message=f"{result.message} {warning}")
    return result


def test_connection(update: DbConfigUpdate) -> DbActionResult:
    if update.db_type == "sqlite":
        return _test_sqlite(update.sqlite)
    try:
        url = _resolve_effective_url(update)
    except ValueError as exc:
        return DbActionResult(ok=False, message=str(exc))
    return _test_mysql(url)


async def build_schema(update: DbConfigUpdate) -> DbActionResult:
    """Runs the startup Alembic migration chain (app.database.init_db) against a candidate database, then seeds
    the starter data (_seed_defaults), so a fresh target is usable, not just structurally correct. Safe on an
    empty or already-current database (no-ops the second time). Doesn't touch the live app or .env
    (save_and_apply switches).
    """
    try:
        url = _resolve_effective_url(update)
    except ValueError as exc:
        return DbActionResult(ok=False, message=str(exc))
    try:
        await _ensure_schema_and_seed(url, update)
    except IncompatibleSchemaError as exc:
        return DbActionResult(ok=False, message=str(exc))
    except ModuleNotFoundError as exc:
        return DbActionResult(
            ok=False, message=f"Driver not installed ({exc.name}). Run: {_PIP_INSTALL_HINT}, then try again."
        )
    except SQLAlchemyError as exc:
        return DbActionResult(ok=False, message=f"Schema build failed: {_short_error(exc)}")
    except Exception as exc:
        # Deliberately broad, as in save_and_apply.
        return DbActionResult(ok=False, message=f"Schema build failed: {exc}")
    return DbActionResult(ok=True, message="Schema is up to date and seeded with default data.")
