"""Unit tests for app/services/db_config_service.py: parsing/building
DATABASE_URL for both backends, the .env upsert that persists it, the
"blank password means keep the existing one" rule, and the
test-connection/build-schema actions themselves (against a real
throwaway SQLite file; MySQL's driver-missing branch is exercised via
monkeypatch rather than assumed, since whether pymysql happens to be
installed varies by environment — see pyproject.toml's optional `mysql`
extra)."""

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import database
from app.models import Note, User
from app.schemas import DbActionResult, DbConfigUpdate, MysqlParams, SqliteParams
from app.services import db_config_service, db_config_url, env_file, instance_process


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own throwaway .env — never touch the real
    project .env file just by running the test suite."""
    monkeypatch.setattr(env_file, "ENV_PATH", tmp_path / ".env")
    return tmp_path


@pytest.fixture(autouse=True)
def no_sibling_instances(monkeypatch):
    """save_and_apply's broadcast_database_switch (see
    app.services.instance_db_broadcast) reads this machine's *real*
    instances.json by default — without this, these tests would depend
    on whatever local instances happen to actually be tracked/alive on
    whichever machine runs the suite, exactly the incidental-environment-
    state trap this project has hit before. Every test in this file gets
    a clean "no other instances" world unless it explicitly overrides
    this (see test_save_and_apply_reports_a_sibling_that_failed_to_switch)."""
    monkeypatch.setattr(instance_process, "read_tracking", lambda: {})


def test_read_env_database_url_falls_back_to_default_when_env_missing():
    assert db_config_url._read_env_database_url() == db_config_url._default_url()


def test_write_then_read_env_database_url_round_trips():
    db_config_url._write_env_database_url("sqlite:///tmp/x.db")
    assert db_config_url._read_env_database_url() == "sqlite:///tmp/x.db"


def test_write_env_database_url_preserves_other_lines(isolated_env):
    (isolated_env / ".env").write_text("SECRET_KEY=abc123\nSOME_OTHER_SETTING=value\n")
    db_config_url._write_env_database_url("sqlite:///tmp/x.db")
    content = (isolated_env / ".env").read_text()
    assert "SECRET_KEY=abc123" in content
    assert "SOME_OTHER_SETTING=value" in content
    assert "DATABASE_URL=sqlite:///tmp/x.db" in content


def test_write_env_database_url_replaces_existing_line_in_place(isolated_env):
    (isolated_env / ".env").write_text("DATABASE_URL=sqlite:///old.db\nSECRET_KEY=abc123\n")
    db_config_url._write_env_database_url("sqlite:///new.db")
    lines = (isolated_env / ".env").read_text().splitlines()
    assert lines == ["DATABASE_URL=sqlite:///new.db", "SECRET_KEY=abc123"]


def test_parse_url_round_trips_sqlite():
    cfg = db_config_url._parse_url("sqlite:////data/pAIring.db")
    assert cfg.db_type == "sqlite"
    assert cfg.sqlite.directory == "/data"
    assert cfg.sqlite.filename == "pAIring.db"


def test_parse_url_round_trips_mysql():
    cfg = db_config_url._parse_url("mysql+pymysql://paichat:s3cret@db.local:3307/paichat")
    assert cfg.db_type == "mysql"
    assert cfg.mysql.host == "db.local"
    assert cfg.mysql.port == 3307
    assert cfg.mysql.user == "paichat"
    assert cfg.mysql.password == "s3cret"
    assert cfg.mysql.database == "paichat"


def test_build_url_sqlite_resolves_relative_directory_against_base_dir(monkeypatch):
    monkeypatch.setattr(db_config_url, "BASE_DIR", Path("/opt/pAIchat"))
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory="data", filename="x.db"))
    assert db_config_url._build_url(update, "sqlite:////irrelevant") == "sqlite:////opt/pAIchat/data/x.db"


def test_build_url_mysql_url_encodes_special_characters_in_credentials():
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="localhost", user="pa@ichat", password="p@ss/word", database="paichat"),
    )
    url = db_config_url._build_url(update, "sqlite:////irrelevant")
    assert url == "mysql+pymysql://pa%40ichat:p%40ss%2Fword@localhost:3306/paichat"


def test_build_url_mysql_reuses_existing_password_when_field_left_blank():
    current = "mysql+pymysql://paichat:s3cret@localhost:3306/paichat"
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="localhost", user="paichat", password="", database="paichat"),
    )
    url = db_config_url._build_url(update, current)
    assert url == "mysql+pymysql://paichat:s3cret@localhost:3306/paichat"


def test_build_url_mysql_raises_when_no_password_anywhere():
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="localhost", user="paichat", password="", database="paichat"),
    )
    with pytest.raises(ValueError, match="password is required"):
        db_config_url._build_url(update, db_config_url._default_url())


def test_build_url_mysql_escapes_a_newline_in_host_or_database():
    """Security regression test: host/database used to be interpolated
    unescaped, so a value containing a literal newline survived into the
    built DSN and, from there, into a real extra line in .env (see
    env_file.write_env_value) — letting a database-settings save inject
    arbitrary environment variables (e.g. overriding SECRET_KEY) that
    take effect on the next restart."""
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="x\nSECRET_KEY=deadbeef", user="paichat", password="s3cret", database="paichat"),
    )
    url = db_config_url._build_url(update, db_config_url._default_url())
    assert "\n" not in url
    assert "%0A" in url  # the newline survives only as a harmless, inert percent-escape


def test_build_url_mysql_round_trips_an_escaped_database_name():
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="localhost", user="paichat", password="s3cret", database="pa ichat"),
    )
    url = db_config_url._build_url(update, db_config_url._default_url())
    cfg = db_config_url._parse_url(url)
    assert cfg.mysql.database == "pa ichat"


def test_resolve_sqlite_path_strips_traversal_from_filename(monkeypatch):
    """Security regression test: `filename` used to be joined onto
    `directory` with no sanitization at all, so an admin-supplied value
    like "../../etc/cron.d/evil" escaped the chosen directory entirely —
    unlike every ordinary upload, which already goes through
    document_upload.safe_filename for exactly this reason."""
    monkeypatch.setattr(db_config_url, "BASE_DIR", Path("/opt/pAIchat"))
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory="data", filename="../../etc/cron.d/evil"))
    url = db_config_url._build_url(update, "sqlite:////irrelevant")
    assert ".." not in url
    assert url == "sqlite:////opt/pAIchat/data/evil"


def test_get_current_config_masks_password_and_reports_has_password():
    db_config_url._write_env_database_url("mysql+pymysql://paichat:s3cret@localhost:3306/paichat")
    cfg = db_config_url.get_current_config()
    assert cfg.mysql.password == ""
    assert cfg.has_password is True


def test_get_current_config_flags_restart_required_when_saved_differs_from_live(monkeypatch):
    """restart_required must compare .env against what the live engine
    is *actually* bound to right now (database.live_database_url, kept
    current by switch_database) — not app.config.DATABASE_URL, which is
    frozen at whatever this process originally booted with and would
    make this flag incorrectly stick at True forever after the first
    ever live switch (see database.switch_database's own docstring)."""
    monkeypatch.setattr(database, "live_database_url", "sqlite:////currently/running.db")
    db_config_url._write_env_database_url("sqlite:////not/yet/applied.db")
    assert db_config_url.get_current_config().restart_required is True


def test_get_current_config_no_restart_required_when_saved_matches_live(monkeypatch):
    monkeypatch.setattr(database, "live_database_url", "sqlite:////same.db")
    db_config_url._write_env_database_url("sqlite:////same.db")
    assert db_config_url.get_current_config().restart_required is False


@pytest_asyncio.fixture
async def restore_live_engine():
    """save_and_apply and switch_database mutate app.database's *global*
    engine/session factory for real — necessary to prove they actually
    work, but it has to be undone afterwards so it doesn't leak into
    whatever test (in this file or another) runs next in the same
    process. Disposes whatever engine ends up live at teardown if it
    isn't the original, so a temp sqlite file's connection doesn't stay
    open either. Also snapshots/restores live_database_url — the other
    piece of global state switch_database updates (see its docstring)."""
    original_engine = database.engine
    original_live_url = database.live_database_url
    yield
    if database.engine is not original_engine:
        await database.engine.dispose()
    database.AsyncSessionLocal.configure(bind=original_engine)
    database.engine = original_engine
    database.live_database_url = original_live_url


@pytest.mark.asyncio
async def test_seed_defaults_creates_the_admin_account_and_default_notes(tmp_path):
    url = f"sqlite:///{tmp_path / 'seed.db'}"
    await db_config_service._ensure_schema_and_seed(
        url, DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="seed.db"))
    )

    engine = create_async_engine(database._async_url(url))
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            admin = (await db.execute(select(User).where(User.username == "admin"))).scalar_one()
            assert admin.role == "admin"
            note_types = set(
                (await db.execute(select(Note.default_type).where(Note.owner_id == admin.id))).scalars().all()
            )
            assert note_types == {"persona", "rules", "skill"}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ensure_schema_and_seed_is_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path / 'seed.db'}"
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="seed.db"))
    await db_config_service._ensure_schema_and_seed(url, update)
    await db_config_service._ensure_schema_and_seed(url, update)  # must not duplicate or fail second time

    engine = create_async_engine(database._async_url(url))
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            admins = (await db.execute(select(User).where(User.username == "admin"))).scalars().all()
            assert len(admins) == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_apply_switches_the_live_engine_with_no_restart(tmp_path, restore_live_engine):
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="live.db"))
    result = await db_config_service.save_and_apply(update)

    assert result.ok is True
    assert db_config_url._read_env_database_url() == f"sqlite:///{tmp_path / 'live.db'}"

    # The live engine is now genuinely serving from the new file — no
    # restart, no re-import, just querying through the same
    # AsyncSessionLocal every request already uses.
    async with database.AsyncSessionLocal() as db:
        admin = (await db.execute(select(User).where(User.username == "admin"))).scalar_one()
        assert admin.role == "admin"

    # Regression: get_current_config used to compare .env against
    # app.config.DATABASE_URL, a constant frozen at process boot — which
    # never updates on a live switch, so this flag would incorrectly
    # read True forever after the very first successful save_and_apply,
    # even though .env and the live engine are now perfectly in sync.
    assert db_config_url.get_current_config().restart_required is False


@pytest.mark.asyncio
async def test_save_and_apply_reports_a_sibling_that_failed_to_switch(tmp_path, monkeypatch, restore_live_engine):
    """Security/reliability regression test for the real incident this
    fix addresses: save_and_apply must still report ok=True (this
    process's own switch — the one serving traffic right now — did
    succeed) while surfacing which *other* local instance(s) still need
    a restart, rather than silently leaving them stranded on the old
    database with no indication anything's wrong."""

    async def fake_broadcast(_url):
        return ["instance on port 8001: Connection refused"]

    monkeypatch.setattr(db_config_service, "broadcast_database_switch", fake_broadcast)

    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="live.db"))
    result = await db_config_service.save_and_apply(update)

    assert result.ok is True
    assert "instance on port 8001" in result.message
    assert "restart" in result.message.lower()


@pytest.mark.asyncio
async def test_save_and_apply_leaves_everything_untouched_on_failure(monkeypatch, restore_live_engine):
    original_engine = database.engine
    original_saved_url = db_config_url._read_env_database_url()

    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="127.0.0.1", port=1, user="paichat", password="s3cret", database="paichat"),
    )
    result = await db_config_service.save_and_apply(update)

    assert result.ok is False
    assert database.engine is original_engine  # live app untouched
    assert db_config_url._read_env_database_url() == original_saved_url  # .env untouched


def test_test_connection_sqlite_reports_ok_when_directory_writable_and_file_absent(tmp_path):
    target_dir = tmp_path / "fresh"
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(target_dir), filename="new.db"))
    result = db_config_service.test_connection(update)
    assert result.ok is True
    assert not (target_dir / "new.db").exists()  # a *test* must not create the database file itself


def test_test_connection_sqlite_connects_to_an_existing_valid_database(tmp_path):
    import sqlite3

    db_path = tmp_path / "existing.db"
    sqlite3.connect(str(db_path)).close()
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="existing.db"))
    result = db_config_service.test_connection(update)
    assert result.ok is True


def test_test_connection_sqlite_rejects_a_file_that_isnt_a_real_database(tmp_path):
    bad_path = tmp_path / "not-a-db.db"
    bad_path.write_text("this is plain text, not a SQLite file header")
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="not-a-db.db"))
    result = db_config_service.test_connection(update)
    assert result.ok is False


def test_test_connection_mysql_reports_missing_driver_clearly(monkeypatch):
    """pymysql is a plain dependency now (requirements.txt) — a normal
    install always gets it — but this stays as a defensive check for
    whatever venv wasn't set up that way, simulated here via monkeypatch
    rather than assumed."""

    def fake_create_engine(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'pymysql'", name="pymysql")

    monkeypatch.setattr(db_config_service, "create_engine", fake_create_engine)
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="localhost", user="paichat", password="s3cret", database="paichat"),
    )
    result = db_config_service.test_connection(update)
    assert result.ok is False
    assert "pip install" in result.message


def test_test_connection_mysql_reports_a_clean_message_on_connection_failure():
    """A real connection attempt against a host nothing listens on —
    exercises the actual failure path (not simulated) without depending
    on any real MySQL server being reachable from wherever tests run."""
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="127.0.0.1", port=1, user="paichat", password="s3cret", database="paichat"),
    )
    result = db_config_service.test_connection(update)
    assert result.ok is False
    assert result.message


class _FakeMysqlConn:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *_args, **_kwargs):
        pass


class _FakeMysqlEngine:
    def connect(self):
        return _FakeMysqlConn()

    def dispose(self):
        pass


def _stub_successful_mysql_connection(monkeypatch):
    """Stubs out the sync connect+schema-check steps of _test_mysql so
    these tests can isolate just the async-driver check that runs after
    them, without needing a real MySQL server reachable from wherever
    tests run."""
    monkeypatch.setattr(db_config_service, "create_engine", lambda *_a, **_k: _FakeMysqlEngine())
    monkeypatch.setattr(
        db_config_service,
        "connection_result_with_schema_status",
        lambda _url, msg: DbActionResult(ok=True, message=f"{msg} Empty database — ready for a fresh pAIring schema."),
    )


def test_test_connection_mysql_warns_when_async_driver_missing_even_though_sync_connection_works(monkeypatch):
    """Regression test: pymysql (sync, used by Test connection/Build
    schema) and aiomysql (async, only needed by Save's live switch) are
    both plain dependencies now (a normal install gets both), but
    nothing used to check them *together* — so an out-of-sync venv (an
    older install, one resolved by hand) could have only pymysql, see
    Test connection and Build schema both report success, and only
    discover the async gap when Save failed on the very next click. Test
    connection must catch it itself."""
    _stub_successful_mysql_connection(monkeypatch)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "aiomysql" else object())

    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="localhost", user="paichat", password="s3cret", database="paichat"),
    )
    result = db_config_service.test_connection(update)
    assert result.ok is False
    assert "Connected successfully" in result.message  # still says the connection itself worked
    assert "pip install" in result.message


def test_test_connection_mysql_succeeds_when_both_drivers_are_present(monkeypatch):
    _stub_successful_mysql_connection(monkeypatch)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())

    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="localhost", user="paichat", password="s3cret", database="paichat"),
    )
    result = db_config_service.test_connection(update)
    assert result.ok is True


@pytest.mark.asyncio
async def test_build_schema_creates_every_table_and_seeds_defaults_on_an_empty_sqlite_file(tmp_path):
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="schema.db"))
    result = await db_config_service.build_schema(update)
    assert result.ok is True

    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "schema.db"))
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        usernames = {row[0] for row in conn.execute("SELECT username FROM users")}
    finally:
        conn.close()
    assert "users" in tables
    assert "conversations" in tables
    assert "alembic_version" in tables
    assert "admin" in usernames  # build_schema seeds the same starter data a fresh install gets


@pytest.mark.asyncio
async def test_build_schema_is_a_safe_no_op_on_an_already_migrated_database(tmp_path):
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="schema.db"))
    first = await db_config_service.build_schema(update)
    second = await db_config_service.build_schema(update)
    assert first.ok is True
    assert second.ok is True


@pytest.mark.asyncio
async def test_build_schema_mysql_reports_a_clean_message_on_connection_failure():
    update = DbConfigUpdate(
        db_type="mysql",
        mysql=MysqlParams(host="127.0.0.1", port=1, user="paichat", password="s3cret", database="paichat"),
    )
    result = await db_config_service.build_schema(update)
    assert result.ok is False
    assert result.message


@pytest.mark.asyncio
async def test_build_schema_never_touches_the_env_file_or_the_live_engine(tmp_path, restore_live_engine):
    """build_schema is the non-committing preview action — only
    save_and_apply is allowed to touch .env or the live app (see its own
    tests above)."""
    original_saved_url = db_config_url._read_env_database_url()
    original_engine = database.engine

    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="preview.db"))
    result = await db_config_service.build_schema(update)

    assert result.ok is True
    assert db_config_url._read_env_database_url() == original_saved_url
    assert database.engine is original_engine


def _make_foreign_sqlite_db(path):
    """Creates a SQLite file with a table pAIring doesn't recognize —
    stands in for "someone pointed this at the wrong database" (another
    app's DB, a stale export, a typo'd name that happens to exist)."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE some_other_apps_table (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_build_schema_refuses_a_database_with_an_incompatible_schema(tmp_path):
    """Regression test for the admin-lockout scenario this whole check
    exists to prevent: without it, build_schema would hand a foreign
    database straight to Alembic, whose pre-Alembic-upgrade heuristic
    (see app.database._run_migrations) would wrongly *stamp* it as
    already migrated without ever creating pAIring's own tables — an app
    that starts fine and then 500s on every request."""
    db_path = tmp_path / "foreign.db"
    _make_foreign_sqlite_db(db_path)

    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="foreign.db"))
    result = await db_config_service.build_schema(update)

    assert result.ok is False
    assert "some_other_apps_table" in result.message

    # And critically: nothing was touched. No alembic_version table
    # planted, no pAIring tables created alongside the foreign one.
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert tables == {"some_other_apps_table"}


@pytest.mark.asyncio
async def test_save_and_apply_refuses_a_database_with_an_incompatible_schema(tmp_path, restore_live_engine):
    db_path = tmp_path / "foreign.db"
    _make_foreign_sqlite_db(db_path)

    original_engine = database.engine
    original_saved_url = db_config_url._read_env_database_url()

    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="foreign.db"))
    result = await db_config_service.save_and_apply(update)

    assert result.ok is False
    assert "some_other_apps_table" in result.message
    assert database.engine is original_engine  # live app never switched
    assert db_config_url._read_env_database_url() == original_saved_url  # .env never written


def test_test_connection_sqlite_reports_incompatible_schema_without_being_blocked_from_reporting_it(tmp_path):
    """test_connection is purely diagnostic — it must still *report* an
    incompatible schema (ok=False, a clear reason) rather than crash,
    even though (unlike build_schema/save_and_apply) it never touches
    the database at all."""
    db_path = tmp_path / "foreign.db"
    _make_foreign_sqlite_db(db_path)

    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="foreign.db"))
    result = db_config_service.test_connection(update)

    assert result.ok is False
    assert "some_other_apps_table" in result.message
    assert "Connected" in result.message  # still says the connection itself worked


@pytest.mark.asyncio
async def test_build_schema_accepts_a_database_with_an_existing_compatible_schema(tmp_path):
    """The flip side of the two tests above: a database that already has
    a real (even if not-yet-fully-current) pAIring schema must still be
    accepted, not treated as foreign just for being non-empty."""
    update = DbConfigUpdate(db_type="sqlite", sqlite=SqliteParams(directory=str(tmp_path), filename="schema.db"))
    first = await db_config_service.build_schema(update)
    assert first.ok is True

    second = await db_config_service.build_schema(update)  # already has pAIring's own schema now
    assert second.ok is True
