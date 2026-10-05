"""Unit tests for app/services/sdcpp_progress.py — real sd.cpp log excerpts (progress bars are \\r-separated)."""

from app.models import ImageGenerationJob
from app.services import image_generation_service, sdcpp_progress

_LOAD = "  |######################                            | 294/686 - 2.33GB/s\x1b[K\r"
_GEN = "[INFO   ] image.cpp:814  - generate_image 128x128\n"
_COND = "[INFO   ] image.cpp:529  - get_learned_condition completed, taking 5.03s\n"


def _step(n, total=4, rate="24.81s/it"):
    return f"  |============>                                     | {n}/{total} - {rate}\x1b[K\r"


def _log(tmp_path, text):
    path = tmp_path / "sdcpp.log"
    path.write_text(text)
    return path


def test_nothing_new_in_the_log_means_no_progress(tmp_path):
    path = _log(tmp_path, "old stuff\n")
    assert sdcpp_progress.read_progress(sdcpp_progress.current_offset(path), path) is None
    assert sdcpp_progress.read_progress(0, tmp_path / "missing.log") is None


def test_stages_in_order(tmp_path):
    path = _log(tmp_path, "")
    expected = [
        (_LOAD, "Loading the model", 15 * 294 / 686),
        (_GEN, "Encoding the prompt", 15),
        (_COND, "Starting to sample", 25),
        (_step(1), "Sampling · step 1 of 4", 25 + 65 / 4),
        (_step(4), "Decoding the image", 90),
    ]
    text = ""
    for chunk, stage, pct in expected:
        text += chunk
        path.write_text(text)
        found = sdcpp_progress.read_progress(0, path)
        assert found.stage == stage and abs(found.progress - pct) < 0.01


def test_sampling_reports_an_eta_from_the_step_rate(tmp_path):
    path = _log(tmp_path, _GEN + _COND + _step(1) + _step(2, rate="10.0s/it"))
    found = sdcpp_progress.read_progress(0, path)
    assert found.stage == "Sampling · step 2 of 4" and found.eta_seconds == 20  # 2 steps left at 10s each


def test_it_per_second_rates_are_converted(tmp_path):
    path = _log(tmp_path, _step(1, 10, "2.00it/s"))
    assert sdcpp_progress.read_progress(0, path).eta_seconds == 4  # 9 steps at 0.5s


def test_only_lines_after_the_jobs_own_offset_count(tmp_path):
    previous = _GEN + _COND + _step(4)
    path = _log(tmp_path, previous)
    offset = sdcpp_progress.current_offset(path)
    assert sdcpp_progress.read_progress(offset, path) is None  # the previous job's lines are ignored
    path.write_text(previous + _GEN)
    assert sdcpp_progress.read_progress(offset, path).stage == "Encoding the prompt"


def test_a_truncated_log_is_read_from_the_start(tmp_path):
    path = _log(tmp_path, _GEN)
    assert sdcpp_progress.read_progress(10_000, path).stage == "Encoding the prompt"


def _job(**kw):
    return ImageGenerationJob(status="running", log_offset=0, **kw)


def test_progress_for_only_applies_to_running_sdcpp_jobs(tmp_path, monkeypatch):
    path = _log(tmp_path, _GEN + _COND + _step(2))
    real = sdcpp_progress.read_progress
    monkeypatch.setattr(sdcpp_progress, "read_progress", lambda offset: real(offset, path))

    running = image_generation_service.progress_for(_job())
    assert running["stage"] == "Sampling · step 2 of 4" and 0 < running["progress"] < 100
    assert image_generation_service.progress_for(ImageGenerationJob(status="complete", log_offset=0)) == {}
    assert image_generation_service.progress_for(ImageGenerationJob(status="running", log_offset=None)) == {}  # ComfyUI
