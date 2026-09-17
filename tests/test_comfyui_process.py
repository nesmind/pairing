"""Unit tests for app/services/comfyui_process.py — the subprocess/PID/
file primitives behind ComfyUI process supervision. No real ComfyUI
process is ever spawned; subprocess.Popen and /proc reads are
monkeypatched or redirected to tmp_path, matching
tests/test_instance_service.py's own conventions for the equivalent
sibling-process primitives."""

import subprocess

import pytest

from app.services import comfyui_process


@pytest.fixture(autouse=True)
def isolated_tracking_file(tmp_path, monkeypatch):
    monkeypatch.setattr(comfyui_process, "_TRACKING_FILE", tmp_path / "comfyui.json")


def test_read_tracking_returns_none_when_never_started():
    assert comfyui_process.read_tracking() is None


def test_write_and_read_tracking_round_trip():
    comfyui_process.write_tracking({"pid": 1234})
    assert comfyui_process.read_tracking() == {"pid": 1234}


def test_write_tracking_none_removes_the_file():
    comfyui_process.write_tracking({"pid": 1234})
    comfyui_process.write_tracking(None)
    assert comfyui_process.read_tracking() is None


def test_read_tracking_tolerates_a_corrupt_file(tmp_path):
    (tmp_path / "comfyui.json").write_text("not json")
    assert comfyui_process.read_tracking() is None


def test_spawn_builds_argv_from_python_and_main_path_and_extra_args(tmp_path, monkeypatch):
    captured = {}

    class _FakeProc:
        pid = 4242

        def poll(self):
            return None  # still running — a successful launch

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(comfyui_process.time, "sleep", lambda _seconds: None)
    main_py = tmp_path / "ComfyUI" / "main.py"
    main_py.parent.mkdir()

    pid = comfyui_process.spawn("/opt/venv/bin/python", str(main_py), "--listen 0.0.0.0 --port 8188")

    assert pid == 4242
    assert captured["argv"] == ["/opt/venv/bin/python", str(main_py), "--listen", "0.0.0.0", "--port", "8188"]
    assert captured["kwargs"]["cwd"] == str(main_py.parent)
    assert captured["kwargs"]["start_new_session"] is True


def test_spawn_returns_none_when_extra_args_is_none(tmp_path, monkeypatch):
    class _FakeProc:
        pid = 1
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: _FakeProc())
    monkeypatch.setattr(comfyui_process.time, "sleep", lambda _seconds: None)
    main_py = tmp_path / "main.py"
    main_py.touch()

    pid = comfyui_process.spawn("/opt/venv/bin/python", str(main_py), None)

    assert pid == 1


def test_spawn_returns_none_on_a_bad_path_instead_of_raising(tmp_path, monkeypatch):
    """A nonexistent python_path/main_py_path (a real, likely admin
    typo) makes subprocess.Popen itself raise OSError before there's
    ever a process to poll() — this must degrade to None, not crash the
    whole request with a raw 500 (a real bug found via live QA)."""

    def fake_popen(_argv, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "/opt/does-not-exist")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert comfyui_process.spawn("/opt/does-not-exist/python", "/opt/does-not-exist/main.py", None) is None


def test_spawn_returns_none_on_immediate_exit(tmp_path, monkeypatch):
    class _FakeProc:
        pid = 999
        returncode = 1

        def poll(self):
            return 1  # already exited

    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: _FakeProc())
    monkeypatch.setattr(comfyui_process.time, "sleep", lambda _seconds: None)
    main_py = tmp_path / "main.py"
    main_py.touch()

    assert comfyui_process.spawn("/opt/venv/bin/python", str(main_py), None) is None


def test_is_alive_false_for_a_pid_with_no_proc_entry():
    # A pid this test process definitely isn't the parent of, and that
    # (almost certainly) doesn't exist on the box at all.
    assert comfyui_process.is_alive(2**30, "/opt/ComfyUI/main.py") is False


def test_terminate_is_a_noop_for_an_already_gone_pid():
    # Must not raise — ProcessLookupError from os.kill is caught.
    comfyui_process.terminate(2**30, "/opt/ComfyUI/main.py")
