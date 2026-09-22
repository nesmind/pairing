"""
Admin-configurable database backend (Settings > System > Database): test a candidate connection, build its
schema (plus the starter data a fresh install seeds itself with), and save — which switches the live app over
immediately, no restart needed. URL parsing/.env live in app/services/db_config_url.py, schema compatibility in
app/services/db_schema_check.py — split out to stay under CLAUDE.md's file-size rule; this file is the three
actions built on them.

Test/build-schema use a throwaway *sync* engine, same as app/database.py's own `_run_migrations` — one-off
actions against a *candidate* database. Saving ends with the live engine actually pointed at the new one (see
app.database.switch_database), so it runs schema/seed on a throwaway *async* session first, then hands that same
URL to switch_database.
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

# Reused across every "a driver is missing" message below.
_PIP_INSTALL_HINT = ".venv/bin/pip install -r requirements.txt"


async def _seed_defaults(url: str) -> None:
    """Brings a database up to the exact same starting point a brand-new
    install reaches on its very first boot: the seed accounts
    (app.services.auth_service.seed_default_users — admin/admin plus
    whatever else that function seeds) and each account's default
    persona/rules/skill notes (app.services.note_service.seed_default_notes).

    Deliberately reuses those functions instead of a hand-written SQL
    fixture: the note *content* especially is expected to keep growing,
    and a duplicated SQL copy of it would silently drift from the real
    defaults the moment either changes without the other being updated
    to match. Reusing the actual functions means there's only ever one
    place that defines what "fresh" means, for every database backend
    alike — no SQLite/MySQL split needed, since this runs through the
    ORM either way. Runs against a throwaway async engine, independent
    of both the live app engine and the sync one Alembic just used for
    the DDL half in build_schema/save_and_apply below.
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
    """The shared checks-then-DDL-then-seed sequence behind both
    build_schema and save_and_apply: confirm the target's existing
    tables (if any) actually look like pAIring's own schema — raises
    IncompatibleSchemaError otherwise, before touching anything (see
    app.services.db_schema_check.classify_schema) — then create/upgrade
    the schema (a no-op if it's already current), then seed the same
    starter data a fresh install gets on its first boot (also a no-op
    past the first time, since both seed functions are idempotent — see
    _seed_defaults)."""
    if update.db_type == "sqlite":
        _resolve_sqlite_path(update.sqlite).parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(classify_schema, url)
    await asyncio.to_thread(_run_migrations, url)
    await _seed_defaults(url)


async def save_and_apply(update: DbConfigUpdate) -> DbActionResult:
    """Saves and immediately switches the live app over — no restart.
    Order matters: the schema/seed step and the live switch both have to
    succeed *before* anything touches .env, so a failure here always
    leaves both the running app and .env exactly as they were, never a
    half-applied state where .env claims a database the app isn't
    actually using.

    One unavoidable side effect: every currently logged-in session (including whichever admin just clicked
    Save) stops working the moment this succeeds. A session cookie only holds a user id — see app/routers/auth.py's
    `request.session["user_id"] = user.id` — and that id belongs to a row in the *old* database; the new one has
    its own freshly seeded admin with a different id, so the very next request 401s and the frontend's `api()`
    helper (app.js) bounces it to /login automatically. Not a bug to work around, just something the caller needs
    to know to log back in right after — see the message below.
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
        # Broad on purpose, as a last resort: this action either fully
        # succeeds or must report *something* clean back to the admin —
        # an unexpected failure here (an Alembic error against a
        # not-quite-compatible schema, say) must never surface as a raw
        # 500 when the whole point of this feature is avoiding surprises.
        return DbActionResult(ok=False, message=f"Couldn't switch databases: {exc}. Nothing changed.")

    _write_env_database_url(url)
    # This process's own live engine is already switched above — every
    # *other* local instance (see app.services.instance_db_broadcast's
    # own docstring for why that's a real, separate process each) still
    # needs telling, or it silently keeps serving the old database
    # forever. Best-effort: still reports ok=True even if one didn't
    # respond, since the switch that matters most (this process, already
    # serving traffic) succeeded regardless.
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
            # Not "SELECT 1" — that never touches the file's actual page
            # structure, so it happily "succeeds" against a file that
            # isn't a SQLite database at all. Querying sqlite_master
            # forces SQLite to read the real header/schema page, which
            # is what actually fails on a bad file.
            conn.execute(text("SELECT count(*) FROM sqlite_master"))
    except SQLAlchemyError as exc:
        return DbActionResult(ok=False, message=f"File exists but isn't a valid SQLite database: {_short_error(exc)}")
    finally:
        engine.dispose()
    return connection_result_with_schema_status(f"sqlite:///{path}", f"Connected to the existing database at {path}.")


def _missing_async_driver_warning() -> str | None:
    """Save's live switch needs `aiomysql`, not just the `pymysql` that
    Test connection/Build schema check via their own sync engine. Both
    are in requirements.txt by default — a normal install already gets
    both together — but this stays as a defensive check for whatever
    venv wasn't set up that way (an older install, one resolved by
    hand): without it, Test connection and Build schema could both
    report success while Save was always going to fail on the next
    step. A plain import check — no need to open an async connection
    just to prove the package exists."""
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
    """Runs the same Alembic migration chain the app applies to itself
    on every startup (see app.database.init_db) against a candidate
    database instead, then seeds it with the same starter data a
    brand-new install gets on its first boot (see _seed_defaults) — so a
    fresh target is immediately usable, not just structurally correct.
    Safe to call on an empty database or an already-current/seeded one
    alike (both steps are no-ops the second time around). Doesn't touch
    the live app or .env at all — see save_and_apply for the action that
    actually switches pAIring over to this database."""
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
        # See save_and_apply's own matching except Exception for why
        # this is deliberately broad.
        return DbActionResult(ok=False, message=f"Schema build failed: {exc}")
    return DbActionResult(ok=True, message="Schema is up to date and seeded with default data.")
