"""Database setup: creates the async SQLAlchemy engine/session and
exposes a FastAPI dependency (`get_db`) that routers use to get a
database session for the duration of a single request.

Async, per CLAUDE.md's "Async First" rule — every service function that
touches the database is `async def` and awaits its queries via
`AsyncSession`/`select()` rather than the old `Session.query()` style.
Alembic itself (see alembic/env.py and `_run_migrations` below) is
deliberately left on a plain *sync* engine of its own: Alembic's
migration runner has no async story worth adopting for a single-writer,
occasional-DDL job like this, and keeping it sync means alembic/env.py
needed zero changes for this migration — it already builds its own
engine straight from `DATABASE_URL`, completely independent of the
async engine below.
"""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import AUTO_MIGRATE, DATABASE_URL


def _async_url(sync_url: str) -> str:
    """Rewrites a plain `sqlite://`/`mysql://` URL (what DATABASE_URL
    holds, and what Alembic wants) to the async-driver variant the app's
    own engine needs. Kept as a URL rewrite rather than asking for two
    separate env vars — DATABASE_URL stays the single source of truth
    app/config.py's own docstring promises, this just picks the right
    driver for whichever backend it points at.

    MySQL is a verified, supported backend (see "Switching to MySQL" in
    README.md) — DATABASE_URL there is still the sync `mysql+pymysql://`
    form (Alembic and this module's own startup schema check both build
    a *sync* engine straight from that string), rewritten here to
    `mysql+aiomysql://` for the app's own request-serving engine only.
    Both drivers are plain dependencies in pyproject.toml, not an
    optional extra — always installed, whichever backend is actually
    configured.
    """
    if sync_url.startswith("sqlite:///"):
        return sync_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    if sync_url.startswith("mysql+pymysql://"):
        return sync_url.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
    if sync_url.startswith("mysql://"):
        return sync_url.replace("mysql://", "mysql+aiomysql://", 1)
    raise ValueError(f"No known async driver mapping for DATABASE_URL={sync_url!r}")


engine = create_async_engine(_async_url(DATABASE_URL))

# The sync-form URL (same shape as .env's DATABASE_URL / what Alembic
# uses) the live engine above is *actually* bound to right now — updated
# by switch_database on every successful live switch. Deliberately a
# separate name from the DATABASE_URL constant above, which stays frozen
# at whatever this process booted with: app.services.db_config_url.
# get_current_config compares .env against *this*, not that constant, or
# its "restart needed" flag would incorrectly stay stuck on True forever
# after the very first live switch (the constant never catches up, even
# though the live engine — the thing that actually matters — did).
live_database_url = DATABASE_URL

# expire_on_commit=False: without this, every attribute on an ORM object
# is marked stale immediately after `await session.commit()` and reading
# one afterwards (e.g. FastAPI serializing a returned row via
# response_model, or a service returning the same object it just
# committed) would trigger an implicit refresh — which needs an awaited
# query and can't happen from a non-async context, failing with
# "MissingGreenlet". This is the standard fix for async SQLAlchemy, not
# a workaround specific to this app.
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

# Base class that every ORM model in app/models/ inherits from. Plain
# declarative metadata, unaffected by sync vs. async — this is also what
# alembic/env.py imports to get at Base.metadata for autogenerate.
Base = declarative_base()


async def get_db():
    """FastAPI dependency: yields one DB session per request, and always
    closes it afterwards (even if the request raised an exception)."""
    async with AsyncSessionLocal() as db:
        yield db


async def switch_database(database_url: str) -> None:
    """Rebinds the app's live engine to a different database, without
    restarting the process — used by
    app.services.db_config_service.save_and_apply after Settings >
    System > Database is saved, so the switch takes effect immediately.

    Validates the new connection *before* touching anything live: if it
    fails, this raises and the app keeps running on whatever it was
    already using, completely untouched.

    `AsyncSessionLocal.configure(bind=...)` reconfigures the *same*
    sessionmaker object in place, rather than creating a new one and
    reassigning the module-level name — so code elsewhere that already
    did `from app.database import AsyncSessionLocal` (e.g.
    title_service.py's background title task) keeps working correctly:
    it's still the same object, just now handing out sessions against
    the new engine. `get_db()` above needs no such care either way,
    since it looks up the (module-level) name fresh on every call.

    Disposing the old engine only closes its *idle* pooled connections —
    a connection some other in-flight request already checked out keeps
    working until that request finishes; it simply won't be returned to
    the old (now-disposed) pool afterward.
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


# The single revision that creates the full schema from empty (see
# alembic/versions/*_baseline_schema.py). Named explicitly here — rather
# than looked up dynamically — because it's also the one revision this
# module ever needs to *stamp* an existing pre-Alembic database at (see
# _run_migrations below); every later revision is applied the normal way.
_BASELINE_REVISION = "6d4212846f5c"


def _run_migrations(database_url: str = DATABASE_URL) -> None:
    """Brings the schema up to date using Alembic. Safe to call on every
    startup, on either a brand-new database or one that already has data:

    - Brand new (no tables at all): Alembic runs every revision from
      scratch, starting with the baseline that creates all the tables.
    - Pre-Alembic install (tables already exist from the old
      create_all() + ad-hoc migration approach, no alembic_version table
      yet): stamping it at the baseline revision tells Alembic "this
      database is already at that point" without re-running its
      `CREATE TABLE` statements, then any revisions added after the
      baseline still apply normally on top.
    - Already-Alembic-managed: just runs whatever new revisions exist.

    Plain sync code, using its own short-lived sync engine — see this
    module's docstring for why Alembic isn't on the async engine above.
    Called via asyncio.to_thread (see init_db) so this brief blocking
    work doesn't block the event loop, even though in practice nothing
    else is being served yet at the point it runs.

    Takes an explicit `database_url` (defaulting to this module's own
    live DATABASE_URL) rather than always reading the module-level
    constant so app.services.db_config_service can reuse this same
    function to build the schema on a *candidate* database an admin is
    configuring from Settings > System > Database, before it's ever
    saved as the app's real DATABASE_URL.
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
    """Read-only: whether `database_url`'s schema is already at
    Alembic's newest revision — never runs any migration itself, safe
    to call against a database other instances might be using at the
    same time. Used by init_db when AUTO_MIGRATE is off (see that
    flag's own docstring in app/config.py) to fail fast with a clear
    message instead of starting the app against a stale or missing
    schema."""
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
    """Raised by init_db when AUTO_MIGRATE is off and the schema isn't
    already at Alembic's head — see AUTO_MIGRATE's own docstring in
    app/config.py for why this fails loudly rather than starting broken
    or silently auto-migrating."""


async def init_db():
    """Brings the schema up to date via Alembic (app.config.AUTO_MIGRATE,
    on by default) — or, with that off, just verifies it's already
    current and refuses to start otherwise. See AUTO_MIGRATE's own
    docstring for why: convenient to auto-migrate for a single instance,
    unsafe for several racing to migrate the same database concurrently
    on cold start."""
    # Import models here (not at module load time) so that all model
    # classes have registered themselves on `Base` before Alembic
    # compares the live database against `Base.metadata`.
    from app import models  # noqa: F401

    if not AUTO_MIGRATE:
        # DATABASE_URL passed explicitly, not relying on
        # schema_is_current's own default parameter value — a default
        # is bound once at function-definition time, not re-read per
        # call, so passing it here is what actually picks up
        # database.DATABASE_URL as it is right now.
        if not await asyncio.to_thread(schema_is_current, DATABASE_URL):
            raise SchemaNotCurrentError(
                "AUTO_MIGRATE is off and the schema isn't at the latest revision. "
                "Run the migration once, separately, before starting the app: "
                ".venv/bin/alembic upgrade head"
            )
        return

    await asyncio.to_thread(_run_migrations, DATABASE_URL)
