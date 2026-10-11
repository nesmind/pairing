"""Database setup: async SQLAlchemy engine/session and the `get_db` FastAPI dependency (one session per request).

Alembic (alembic/env.py, `_run_migrations`) stays on its own sync engine built from `DATABASE_URL`.
"""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import AUTO_MIGRATE, DATABASE_URL


def _async_url(sync_url: str) -> str:
    """Rewrites a sync `sqlite://` / `mysql://` URL to its async-driver form (`mysql+aiomysql://`) for the
    app's engine. Alembic and the startup schema check keep the sync URL.
    """
    if sync_url.startswith("sqlite:///"):
        return sync_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    if sync_url.startswith("mysql+pymysql://"):
        return sync_url.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
    if sync_url.startswith("mysql://"):
        return sync_url.replace("mysql://", "mysql+aiomysql://", 1)
    raise ValueError(f"No known async driver mapping for DATABASE_URL={sync_url!r}")


engine = create_async_engine(_async_url(DATABASE_URL))

# The sync-form URL the live engine is bound to now, updated by switch_database. Separate from the
# frozen DATABASE_URL constant so db_config_url.get_current_config's "restart needed" flag clears after a live switch.
live_database_url = DATABASE_URL

# expire_on_commit=False: otherwise attributes go stale after commit and reading them triggers an
# implicit refresh that fails outside async ("MissingGreenlet").
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

# Base for every ORM model in app/models/; alembic/env.py uses its metadata.
Base = declarative_base()


async def get_db():
    """Yields one DB session per request and always closes it."""
    async with AsyncSessionLocal() as db:
        yield db


async def switch_database(database_url: str) -> None:
    """Rebinds the live engine to another database without restart (db_config_service.save_and_apply).

    The new connection is validated first; on failure the app keeps its current one.
    `AsyncSessionLocal.configure(bind=...)` mutates the same sessionmaker, so existing imports
    (e.g. title_service) keep working. Disposing the old engine closes only idle connections;
    checked-out ones finish normally.
    """
    global engine, live_database_url

    new_engine = create_async_engine(_async_url(database_url))
    try:
        async with new_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        await new_engine.dispose()
        raise

    old_engine = engine
    AsyncSessionLocal.configure(bind=new_engine)
    engine = new_engine
    live_database_url = database_url
    await old_engine.dispose()


# The baseline revision (see alembic/versions/*_baseline_schema.py), named explicitly since
# pre-Alembic databases are stamped at it (see _run_migrations).
_BASELINE_REVISION = "6d4212846f5c"


def _run_migrations(database_url: str = DATABASE_URL) -> None:
    """Brings the schema up to date; safe on every startup:
    - new database: runs every revision from the baseline;
    - pre-Alembic install: stamps the baseline, then applies later revisions;
    - Alembic-managed: applies new revisions.

    Sync code on a short-lived engine, called via asyncio.to_thread (see init_db). Takes `database_url`
    so db_config_service can build the schema on a candidate database.
    """
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from alembic import command
    from app.config import BASE_DIR

    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)

    sync_engine = create_engine(database_url)
    try:
        existing_tables = set(inspect(sync_engine).get_table_names())
    finally:
        sync_engine.dispose()

    if existing_tables and "alembic_version" not in existing_tables:
        command.stamp(cfg, _BASELINE_REVISION)

    command.upgrade(cfg, "head")


def schema_is_current(database_url: str = DATABASE_URL) -> bool:
    """Read-only: whether the schema is at Alembic's head; runs no migration. Used by init_db when
    AUTO_MIGRATE is off, to fail fast on a stale schema.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, inspect

    from app.config import BASE_DIR

    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "alembic"))
    head_revision = ScriptDirectory.from_config(cfg).get_current_head()

    sync_engine = create_engine(database_url)
    try:
        tables = set(inspect(sync_engine).get_table_names())
        if "alembic_version" not in tables:
            return False
        with sync_engine.connect() as conn:
            current_revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    finally:
        sync_engine.dispose()
    return current_revision == head_revision


class SchemaNotCurrentError(RuntimeError):
    """Raised by init_db when AUTO_MIGRATE is off and the schema is not at head."""


async def init_db():
    """Migrates via Alembic (AUTO_MIGRATE, default on); with it off, only verifies the schema is current
    and refuses to start otherwise (concurrent migrations on cold start are unsafe).
    """
    # Import models here so they register on `Base` before Alembic compares metadata.
    from app import models  # noqa: F401

    if not AUTO_MIGRATE:
        # Pass DATABASE_URL explicitly: a default parameter is bound at definition time and misses later changes.
        if not await asyncio.to_thread(schema_is_current, DATABASE_URL):
            raise SchemaNotCurrentError(
                "AUTO_MIGRATE is off and the schema isn't at the latest revision. "
                "Run the migration once, separately, before starting the app: "
                ".venv/bin/alembic upgrade head"
            )
        return

    await asyncio.to_thread(_run_migrations, DATABASE_URL)
