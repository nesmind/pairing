"""Unit tests for app/services/env_file.py — the generic .env upsert
helper both app/services/db_config_url.py and app/services/
ollama_process.py build on."""

from app.services import env_file


def test_read_env_value_falls_back_to_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(env_file, "ENV_PATH", tmp_path / ".env")
    assert env_file.read_env_value("SOME_KEY") is None


def test_write_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(env_file, "ENV_PATH", tmp_path / ".env")
    env_file.write_env_value("SOME_KEY", "hello")
    assert env_file.read_env_value("SOME_KEY") == "hello"


def test_write_env_value_preserves_other_lines(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("SECRET_KEY=abc\nOTHER=1\n")
    monkeypatch.setattr(env_file, "ENV_PATH", env_path)

    env_file.write_env_value("SOME_KEY", "new-value")

    lines = env_path.read_text().splitlines()
    assert "SECRET_KEY=abc" in lines
    assert "OTHER=1" in lines
    assert "SOME_KEY=new-value" in lines


def test_write_env_value_replaces_existing_line_in_place(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_KEY=old\nOTHER=1\n")
    monkeypatch.setattr(env_file, "ENV_PATH", env_path)

    env_file.write_env_value("SOME_KEY", "new")

    lines = env_path.read_text().splitlines()
    assert lines.count("SOME_KEY=new") == 1
    assert "SOME_KEY=old" not in lines
    assert "OTHER=1" in lines


def test_write_env_values_writes_several_keys_in_one_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(env_file, "ENV_PATH", tmp_path / ".env")
    env_file.write_env_values({"A": "1", "B": "2"})
    assert env_file.read_env_value("A") == "1"
    assert env_file.read_env_value("B") == "2"


def test_write_env_values_removes_a_key_whose_value_is_none(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("A=1\nB=2\n")
    monkeypatch.setattr(env_file, "ENV_PATH", env_path)

    env_file.write_env_values({"A": None})

    lines = env_path.read_text().splitlines()
    assert "A=1" not in lines
    assert not any(line.startswith("A=") for line in lines)
    assert "B=2" in lines


def test_write_env_values_never_writes_a_bare_key_for_a_none_value_not_already_present(tmp_path, monkeypatch):
    monkeypatch.setattr(env_file, "ENV_PATH", tmp_path / ".env")
    env_file.write_env_values({"A": None, "B": "2"})
    assert env_file.read_env_value("A") is None
    assert env_file.read_env_value("B") == "2"


def test_write_env_value_and_read_env_value_accept_an_explicit_path(tmp_path):
    """The path parameter — not just the module-level ENV_PATH default — is what
    app.services.matricxon_process needs to write into a *different* project's own .env file."""
    other_path = tmp_path / "other" / ".env"
    other_path.parent.mkdir()
    env_file.write_env_value("KEY", "value", path=other_path)
    assert env_file.read_env_value("KEY", path=other_path) == "value"
    assert not (tmp_path / ".env").exists()  # never touched the default ENV_PATH
