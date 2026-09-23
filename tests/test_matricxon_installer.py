"""Unit tests for app/services/matricxon_installer.py — the Matricxon-flavored twin of
tests/test_comfyui_installer.py (see that file's own docstring for the shared testing approach: every subprocess
call (git/venv/pip) is monkeypatched via a fake asyncio.create_subprocess_exec, no real network access, clone,
or pip install happens here). Differs from ComfyUI's own installer/tests in the ways Matricxon's own project
shape differs — a `.venv` (not `venv`) directory name, a saved project_dir (not python_path/main_py_path) in
the completion event, no GPU/--cpu extra_args concept (Matricxon is CPU-only unconditionally), no AVX2 gate
(see install_stream's own docstring for the real, confirmed-live evidence that one doesn't apply here), and an
is_installed() that delegates to app.services.matricxon_process rather than a dedicated completion marker."""

import shutil

import pytest

from app.services import matricxon_installer, matricxon_process


@pytest.fixture(autouse=True)
def _assume_linux_by_default(monkeypatch):
    monkeypatch.setattr(matricxon_installer, "local_install_supported", lambda: True)


@pytest.fixture(autouse=True)
def _assume_a_real_pinned_default(monkeypatch):
    """install_stream refuses outright with no repo/version at all (see
    test_install_stream_reports_error_when_no_repo_or_version_is_configured below for that behavior itself) —
    pinned to real-looking defaults here so the rest of this file's tests exercise the clone/venv/pip flow
    without needing to pass repo=/version= explicitly every time."""
    monkeypatch.setattr(matricxon_installer, "MATRICXON_GITHUB_REPO", "nesmind/matricxon")
    monkeypatch.setattr(matricxon_installer, "MATRICXON_DEFAULT_VERSION", "v0.1")


def test_is_installed_delegates_to_matricxon_process(monkeypatch):
    monkeypatch.setattr(matricxon_process, "is_installed", lambda: True)
    assert matricxon_installer.is_installed() is True
    monkeypatch.setattr(matricxon_process, "is_installed", lambda: False)
    assert matricxon_installer.is_installed() is False


@pytest.mark.asyncio
async def test_install_stream_reports_error_when_no_repo_or_version_is_configured(monkeypatch):
    monkeypatch.setattr(matricxon_installer, "MATRICXON_GITHUB_REPO", "")
    monkeypatch.setattr(matricxon_installer, "MATRICXON_DEFAULT_VERSION", "")

    events = [event async for event in matricxon_installer.install_stream()]

    assert len(events) == 1
    assert "error" in events[0]


@pytest.mark.asyncio
async def test_install_stream_reports_error_when_git_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    events = [event async for event in matricxon_installer.install_stream()]

    assert len(events) == 1
    assert "git" in events[0]["error"]


@pytest.mark.asyncio
async def test_install_stream_blocks_on_an_unsupported_os(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(matricxon_installer, "local_install_supported", lambda: False)
    monkeypatch.setattr(matricxon_installer.platform, "system", lambda: "Darwin")

    called = []
    monkeypatch.setattr(shutil, "which", lambda _name: called.append(True) or "/usr/bin/git")

    events = [event async for event in matricxon_installer.install_stream()]

    assert len(events) == 1
    assert "Darwin" in events[0]["error"]
    assert called == []  # refused before even checking for git


def test_install_stream_has_no_avx2_gate_unlike_comfyui_installer():
    """The real bug this covers: an earlier version of this module copied comfyui_installer's AVX2 gate
    verbatim, which refused every reinstall attempt on a real, confirmed-working machine (a Sandy Bridge CPU
    with no AVX2 at all) — see install_stream's own docstring for the live evidence that Matricxon's actual
    dependency versions don't need it (unlike ComfyUI's own gate, deliberately left untouched). Asserts the
    absence of the import directly rather than trying to prove a negative through install_stream's behavior."""
    assert not hasattr(matricxon_installer, "has_avx2")


class _AsyncLineIterator:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self._lines:
            yield line


class _FakeSubprocess:
    def __init__(self, lines: list[bytes], returncode: int):
        self.stdout = _AsyncLineIterator(lines)
        self._returncode_to_report = returncode
        self.returncode = None

    async def wait(self):
        self.returncode = self._returncode_to_report
        return self.returncode

    async def communicate(self):
        # Answers PipProgressTracker.supports_raw's `pip install --help` probe like a modern pip.
        return b"--progress-bar [on, off, raw]", b""


def _fake_clone_and_venv(tmp_path):
    """Shared fake create_subprocess_exec: simulates git actually creating the target directory +
    requirements.txt, and venv creation actually producing the interpreter path install_stream's own pip step
    invokes — same idiom as test_comfyui_installer's identical helper, inlined per-test there."""

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        if argv[0] == "git":
            target_dir = tmp_path / "matricxon"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "requirements.txt").write_text("some-package\n")
        if "venv" in argv:
            venv_python = tmp_path / "matricxon" / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("")
        return _FakeSubprocess([b"ok\n"], 0)

    return fake_create_subprocess_exec


@pytest.mark.asyncio
async def test_install_stream_runs_clone_venv_and_pip_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    calls = []

    async def fake_create_subprocess_exec(*argv, **kwargs):
        calls.append(argv)
        return await _fake_clone_and_venv(tmp_path)(*argv, **kwargs)

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in matricxon_installer.install_stream()]

    assert events[-1]["done"] is True
    assert events[-1]["step"] == 3
    assert events[-1]["total_steps"] == 3
    assert events[-1]["project_dir"] == str(tmp_path / "matricxon")
    assert calls[0][0] == "git"
    assert calls[1][1:3] == ("-m", "venv")
    assert str(tmp_path / "matricxon" / ".venv") in calls[1]
    assert calls[2][-2:] == ("install", "--help")  # PipProgressTracker.supports_raw's probe
    assert "pip" in calls[3]
    assert str(tmp_path / "matricxon" / ".venv" / "bin" / "python") == calls[3][0]
    assert calls[3][4:6] == ("--progress-bar", "raw")

    clone_events = [e for e in events if e.get("step") == 1]
    venv_events = [e for e in events if e.get("step") == 2]
    pip_events = [e for e in events if e.get("step") == 3 and not e.get("done")]
    assert all(e["step_label"] == "Cloning" for e in clone_events)
    assert all(e["step_label"] == "Creating a Python environment" for e in venv_events)
    assert all(e["step_label"].startswith("Installing dependencies") for e in pip_events)


def test_preserve_and_restore_data_dir_round_trips_real_files(tmp_path):
    target_dir = tmp_path / "matricxon"
    models_dir = target_dir / "data" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ministral-3b.gguf").write_text("fake weights")

    preserved = matricxon_installer._preserve_data_dir(target_dir)

    assert preserved is not None
    assert not (target_dir / "data").exists()  # moved out, not copied
    assert (preserved / "models" / "ministral-3b.gguf").read_text() == "fake weights"

    matricxon_installer._restore_data_dir(preserved, target_dir)

    assert (target_dir / "data" / "models" / "ministral-3b.gguf").read_text() == "fake weights"
    assert not preserved.parent.exists()  # the tempfile.mkdtemp() holding dir was cleaned up


def test_preserve_data_dir_is_a_noop_when_theres_nothing_there(tmp_path):
    target_dir = tmp_path / "matricxon"
    target_dir.mkdir()
    assert matricxon_installer._preserve_data_dir(target_dir) is None


@pytest.mark.asyncio
async def test_install_stream_preserves_previously_downloaded_models_across_a_reinstall(monkeypatch, tmp_path):
    """The real bug this covers: reinstalling used to `shutil.rmtree` the whole target_dir outright, silently
    destroying every already-pulled model living at data/models (Matricxon's real Settings.models_dir default)
    — pAIring's own managed model library, not a disposable install artifact, unlike Ollama's own installer,
    whose models live in a completely separate location this app never touches."""
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    existing_target = tmp_path / "matricxon"
    existing_models = existing_target / "data" / "models"
    existing_models.mkdir(parents=True)
    (existing_models / "already-pulled.gguf").write_text("real weights, several GB in spirit")

    async def fake_create_subprocess_exec(*argv, **kwargs):
        return await _fake_clone_and_venv(tmp_path)(*argv, **kwargs)

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in matricxon_installer.install_stream()]

    assert events[-1]["done"] is True
    restored = existing_target / "data" / "models" / "already-pulled.gguf"
    assert restored.exists()
    assert restored.read_text() == "real weights, several GB in spirit"


@pytest.mark.asyncio
async def test_install_stream_restores_preserved_models_even_when_the_clone_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    existing_target = tmp_path / "matricxon"
    existing_models = existing_target / "data" / "models"
    existing_models.mkdir(parents=True)
    (existing_models / "already-pulled.gguf").write_text("must survive a failed reinstall too")

    async def fake_create_subprocess_exec(*_argv, **_kwargs):
        return _FakeSubprocess([b"fatal: could not resolve host\n"], 1)

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in matricxon_installer.install_stream()]

    assert "error" in events[-1]
    restored = existing_target / "data" / "models" / "already-pulled.gguf"
    assert restored.exists()
    assert restored.read_text() == "must survive a failed reinstall too"


@pytest.mark.asyncio
async def test_install_stream_sets_proxy_env_for_clone_and_pip_but_not_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    calls = []

    async def fake_create_subprocess_exec(*argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        return await _fake_clone_and_venv(tmp_path)(*argv, **kwargs)

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    [event async for event in matricxon_installer.install_stream(proxy_url="http://10.0.0.5:8080")]

    # calls[2] is PipProgressTracker.supports_raw's local `pip install --help` probe - no network, no proxy.
    git_env, venv_env, pip_env = calls[0][1], calls[1][1], calls[3][1]
    for env in (git_env, pip_env):
        assert env["HTTP_PROXY"] == "http://10.0.0.5:8080"
        assert env["http_proxy"] == "http://10.0.0.5:8080"
    assert venv_env is None


@pytest.mark.asyncio
async def test_install_stream_uses_a_given_repo_and_version_override(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    calls = []

    async def fake_create_subprocess_exec(*argv, **kwargs):
        calls.append(argv)
        return await _fake_clone_and_venv(tmp_path)(*argv, **kwargs)

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in matricxon_installer.install_stream(repo="me/matricxon-fork", version="v9.9.9")]

    assert calls[0] == (
        "git",
        "clone",
        "--branch",
        "v9.9.9",
        "--depth",
        "1",
        "https://github.com/me/matricxon-fork.git",
        str(tmp_path / "matricxon"),
    )
    assert "v9.9.9" in events[0]["status"]


@pytest.mark.asyncio
async def test_install_stream_removes_a_partial_clone_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")

    async def fake_create_subprocess_exec(*argv, **_kwargs):
        target_dir = tmp_path / "matricxon"
        target_dir.mkdir(parents=True, exist_ok=True)
        return _FakeSubprocess([b"error\n"], 1)

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    events = [event async for event in matricxon_installer.install_stream()]

    assert "error" in events[-1]
    assert not (tmp_path / "matricxon").exists()


@pytest.mark.asyncio
async def test_install_stream_removes_a_pre_existing_directory_before_cloning(monkeypatch, tmp_path):
    monkeypatch.setattr(matricxon_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/git")
    stale_dir = matricxon_installer.install_dir()
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale-file").write_text("leftover from a failed attempt")

    async def fake_create_subprocess_exec(*_argv, **_kwargs):
        return _FakeSubprocess([], 1)

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    [event async for event in matricxon_installer.install_stream()]

    assert not (stale_dir / "stale-file").exists()


def test_proxied_env_returns_none_when_no_proxy_given():
    assert matricxon_installer._proxied_env(None) is None


def test_proxied_env_sets_both_cases_when_a_proxy_is_given():
    env = matricxon_installer._proxied_env("http://1.2.3.4:3128")
    assert env["HTTP_PROXY"] == "http://1.2.3.4:3128"
    assert env["http_proxy"] == "http://1.2.3.4:3128"


@pytest.mark.asyncio
async def test_run_streamed_kills_the_subprocess_if_torn_down_early(monkeypatch):
    class _FakeInfiniteSubprocess:
        def __init__(self):
            self.killed = False
            self.returncode = None
            self.stdout = self

        def __aiter__(self):
            return self

        async def __anext__(self):
            return b"still running...\n"

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    fake_process = _FakeInfiniteSubprocess()

    async def fake_create_subprocess_exec(*_argv, **_kwargs):
        return fake_process

    monkeypatch.setattr(matricxon_installer.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    gen = matricxon_installer._run_streamed(["some-long-running-cmd"])
    line = await gen.__anext__()
    assert line == "still running..."
    assert fake_process.killed is False

    await gen.aclose()

    assert fake_process.killed is True
