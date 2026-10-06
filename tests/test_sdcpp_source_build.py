"""sdcpp_source_build: tool checks, command streaming (real tiny subprocesses) and the whole build flow with the
git/cmake commands scripted — nothing is compiled here."""

import asyncio

import pytest

from app.services import sdcpp_installer, sdcpp_source_build
from app.services.sdcpp_source_build import BuildError, _run, _stream_command, build_stream


async def _collect(stream) -> list[dict]:
    return [event async for event in stream]


def test_missing_tools_lists_whatever_is_not_on_the_path(monkeypatch):
    monkeypatch.setattr(
        sdcpp_source_build.shutil, "which", lambda tool: None if tool in ("cmake", "g++", "c++") else "/bin/x"
    )
    assert sdcpp_source_build.missing_tools() == ["cmake", "a C++ compiler (g++)"]
    monkeypatch.setattr(sdcpp_source_build.shutil, "which", lambda tool: "/bin/x")
    assert sdcpp_source_build.missing_tools() == []


def test_jobs_is_half_the_cores_but_at_least_one(monkeypatch):
    monkeypatch.setattr(sdcpp_source_build.os, "cpu_count", lambda: 8)
    assert sdcpp_source_build._jobs() == 4
    monkeypatch.setattr(sdcpp_source_build.os, "cpu_count", lambda: 1)
    assert sdcpp_source_build._jobs() == 1


@pytest.mark.asyncio
async def test_stream_command_yields_lines_and_raises_with_the_output_tail_on_failure():
    lines = [line async for line in _stream_command(["sh", "-c", "echo a; echo b >&2"], None, {})]
    assert lines == ["a", "b"]
    with pytest.raises(BuildError, match="(?s)exit 3.*boom"):
        await _run(["sh", "-c", "echo boom; exit 3"], None, {})


@pytest.mark.asyncio
async def test_a_missing_tool_is_reported_without_touching_the_install_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(sdcpp_source_build, "missing_tools", lambda: ["cmake"])
    events = await _collect(build_stream(None, None, None, tmp_path / "sd"))
    assert len(events) == 1 and "needs cmake" in events[0]["error"] and "apt install" in events[0]["error"]
    assert not (tmp_path / "sd").exists()


def _script_commands(monkeypatch, fail_on: str | None = None, produce_binary: bool = True, dirty: str = ""):
    """Replaces the git/cmake runs: clone makes the source folder, the build prints percent lines and (optionally)
    leaves the binaries in build/bin."""
    calls = []

    async def fake_stream(argv, cwd, env):
        calls.append(argv)
        if fail_on and fail_on in argv:
            raise BuildError("compiler exploded")
        if argv[:2] == ["git", "clone"]:
            from pathlib import Path

            (Path(argv[-1]) / ".git").mkdir(parents=True)
        elif argv[:2] == ["git", "status"]:
            if dirty:
                yield dirty
        elif argv[:2] == ["git", "rev-parse"]:
            yield "abc1234"
        elif "--build" in argv:
            from pathlib import Path

            bindir = Path(argv[argv.index("--build") + 1]) / "bin"
            for line in ("[  5%] Building C object a.o", "[  5%] Building C object b.o", "[ 90%] Linking sd-server"):
                yield line
            if produce_binary:
                bindir.mkdir(parents=True, exist_ok=True)
                (bindir / "sd-server").write_text("NEW-BINARY")
        if False:
            yield ""

    monkeypatch.setattr(sdcpp_source_build, "_stream_command", fake_stream)
    monkeypatch.setattr(sdcpp_source_build, "missing_tools", lambda: [])
    return calls


@pytest.mark.asyncio
async def test_build_installs_the_binary_keeps_models_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "sd"
    (target / "models").mkdir(parents=True)
    (target / "models" / "keep.gguf").write_text("model")
    (target / "sd-server").write_text("OLD-BINARY")
    (target / "libggml.so.0").write_text("old-lib")
    calls = _script_commands(monkeypatch)

    events = await _collect(build_stream("org/repo", "v1", None, target))

    assert [e.get("step_label") for e in events if "step" in e and "completed" not in e][:4] == [
        "Cloning",
        "Configuring",
        "Compiling",
        "Installing",
    ]
    percents = [e["completed"] for e in events if "completed" in e]
    assert percents == [5, 90]  # a repeated percentage is not re-sent
    done = events[-1]
    assert done["done"] and done["version"] == "abc1234" and done["binary_path"] == str(target / "sd-server")
    assert (target / "sd-server").read_text() == "NEW-BINARY"
    assert not (target / "libggml.so.0").exists()  # old engine files are replaced
    assert (target / "models" / "keep.gguf").read_text() == "model"
    assert (target / ".install_complete").read_text() == "abc1234"
    assert (target / "src" / ".git").is_dir() and not (target / "src.partial").exists()  # the source is kept
    assert (target / "build" / "bin" / "sd-server").exists()  # and so is the build tree
    assert done["source_dir"] == str(target / "src")
    clone = calls[0]
    assert clone[:2] == ["git", "clone"] and "--branch" in clone and "v1" in clone
    assert "https://github.com/org/repo" in clone


@pytest.mark.asyncio
async def test_default_branch_is_cloned_when_no_ref_is_given(tmp_path, monkeypatch):
    calls = _script_commands(monkeypatch)
    await _collect(build_stream(None, None, None, tmp_path / "sd"))
    assert "--branch" not in calls[0]
    assert calls[0][-1].endswith("src.partial")  # cloned beside the real folder, then renamed


@pytest.mark.asyncio
async def test_a_failed_build_keeps_the_existing_install_and_the_source(tmp_path, monkeypatch):
    target = tmp_path / "sd"
    target.mkdir()
    (target / "sd-server").write_text("OLD-BINARY")
    _script_commands(monkeypatch, fail_on="--build")

    events = await _collect(build_stream(None, None, None, target))

    assert "Building stable-diffusion.cpp failed: compiler exploded" in events[-1]["error"]
    assert (target / "sd-server").read_text() == "OLD-BINARY"
    assert (target / "src" / ".git").is_dir()  # a retry resumes from the kept clone and build tree


@pytest.mark.asyncio
async def test_a_build_without_the_binary_is_an_error(tmp_path, monkeypatch):
    _script_commands(monkeypatch, produce_binary=False)
    events = await _collect(build_stream(None, None, None, tmp_path / "sd"))
    assert "no sd-server binary" in events[-1]["error"]


@pytest.mark.asyncio
async def test_install_stream_delegates_the_source_choice(tmp_path, monkeypatch):
    seen = {}

    async def fake_build(repo, ref, proxy_url, target_dir):
        seen.update(repo=repo, ref=ref, proxy=proxy_url)
        yield {"done": True}

    monkeypatch.setattr(sdcpp_installer.sdcpp_source_build, "build_stream", fake_build)
    monkeypatch.setattr(sdcpp_installer, "local_install_supported", lambda: True)
    events = [e async for e in sdcpp_installer.install_stream(None, None, proxy_url="http://p", build="source")]
    assert events == [{"done": True}]
    assert seen == {"repo": "leejet/stable-diffusion.cpp", "ref": None, "proxy": "http://p"}  # no pinned tag forced


@pytest.mark.asyncio
async def test_a_second_build_while_one_runs_is_refused_and_leaves_the_first_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(sdcpp_source_build, "missing_tools", lambda: [])
    gate_reached, release = asyncio.Event(), asyncio.Event()

    async def slow_stream(argv, cwd, env):
        if argv[:2] == ["git", "clone"]:
            gate_reached.set()
            await release.wait()
        raise BuildError("stop here")
        yield ""

    monkeypatch.setattr(sdcpp_source_build, "_stream_command", slow_stream)
    first = asyncio.create_task(_collect(build_stream(None, None, None, tmp_path / "sd")))
    await gate_reached.wait()

    second = await _collect(build_stream(None, None, None, tmp_path / "sd"))
    assert "already running" in second[0]["error"]

    release.set()
    assert "failed" in (await first)[-1]["error"]
    assert (await _collect(build_stream(None, None, None, tmp_path / "sd")))[-1]["error"]  # the flag was released
    assert sdcpp_source_build._building is False


@pytest.mark.asyncio
async def test_a_second_build_updates_the_kept_clone_instead_of_cloning_again(tmp_path, monkeypatch):
    target = tmp_path / "sd"
    first_calls = _script_commands(monkeypatch)
    await _collect(build_stream(None, None, None, target))
    assert first_calls[0][:2] == ["git", "clone"]

    calls = _script_commands(monkeypatch)
    events = await _collect(build_stream(None, "v2", None, target))

    kinds = [argv[:2] for argv in calls]
    assert ["git", "clone"] not in kinds
    assert ["git", "fetch"] in kinds and ["git", "reset"] in kinds and ["git", "submodule"] in kinds
    assert next(a for a in calls if a[:2] == ["git", "fetch"])[-1] == "v2"
    assert events[0]["step_label"] == "Updating" and events[-1]["done"]


@pytest.mark.asyncio
async def test_an_update_is_refused_when_tracked_source_files_were_edited(tmp_path, monkeypatch):
    target = tmp_path / "sd"
    _script_commands(monkeypatch)
    await _collect(build_stream(None, None, None, target))

    calls = _script_commands(monkeypatch, dirty=" M src/stable-diffusion.cpp")
    events = await _collect(build_stream(None, None, None, target))

    assert "local changes" in events[-1]["error"]
    assert ["git", "reset"] not in [argv[:2] for argv in calls]  # nothing was overwritten
    assert (target / "src" / ".git").is_dir()
