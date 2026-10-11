"""Tests for app/hardware.py's has_avx2. /proc/cpuinfo is patched via app.hardware's Path import, so results don't
depend on the real CPU."""

from app import hardware


class _FakePath:
    def __init__(self, text: str | None = None, raises: bool = False):
        self._text = text
        self._raises = raises

    def read_text(self) -> str:
        if self._raises:
            raise OSError("no such file")
        return self._text


def test_has_avx2_true_when_the_flag_is_present(monkeypatch):
    monkeypatch.setattr(hardware, "Path", lambda _p: _FakePath("flags : fpu vme avx avx2 sse4_2"))
    assert hardware.has_avx2() is True


def test_has_avx2_false_when_the_flag_is_absent(monkeypatch):
    # Confirmed live: a 2011-era CPU with AVX but not AVX2. Modern prebuilt numpy/torch wheels assume AVX2 and
    # SIGILL without it: the case comfyui_installer's AVX2 gate catches up front.
    monkeypatch.setattr(hardware, "Path", lambda _p: _FakePath("flags : fpu vme avx sse4_2"))
    assert hardware.has_avx2() is False


def test_has_avx2_fails_open_when_cpuinfo_is_unreadable(monkeypatch):
    monkeypatch.setattr(hardware, "Path", lambda _p: _FakePath(raises=True))
    assert hardware.has_avx2() is True


def test_local_install_supported_true_on_linux(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    assert hardware.local_install_supported() is True


def test_local_install_supported_false_on_macos(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    assert hardware.local_install_supported() is False


def test_local_install_supported_false_on_windows(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    assert hardware.local_install_supported() is False
