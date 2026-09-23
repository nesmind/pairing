"""Unit tests for app/services/pip_progress.py - turning `pip install --progress-bar raw` output into
installer progress events."""

from app.services.pip_progress import PipProgressTracker


def test_progress_lines_become_completed_total_events_without_a_log_line():
    tracker = PipProgressTracker("Installing dependencies")

    event = tracker.event_for("Progress 450000000 of 899700000")

    assert event["completed"] == 450000000
    assert event["total"] == 899700000
    assert event["unit"] == "bytes"  # lets the UI show MB, unlike git-clone object counts
    assert "status" not in event  # four lines a second would flood the visible log


def test_downloading_line_names_the_package_in_the_step_label():
    tracker = PipProgressTracker("Installing dependencies")

    event = tracker.event_for("  Downloading torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl (899.7 MB)")

    assert event["step_label"] == "Installing dependencies — downloading torch (899.7 MB)"
    assert event["status"].strip().startswith("Downloading torch")
    # ...and the following progress lines keep that label.
    assert tracker.event_for("Progress 1 of 2")["step_label"].endswith("downloading torch (899.7 MB)")


def test_unknown_total_reports_no_percentage():
    tracker = PipProgressTracker("Installing dependencies")
    assert "total" not in tracker.event_for("Progress 1024 of 0")


def test_install_phase_says_there_is_no_progress_to_show():
    tracker = PipProgressTracker("Installing dependencies")
    event = tracker.event_for("Installing collected packages: torch, numpy")
    assert "installing packages" in event["step_label"]
    assert event["status"] == "Installing collected packages: torch, numpy"


def test_other_lines_are_logged_with_the_base_label():
    tracker = PipProgressTracker("Installing dependencies")
    assert tracker.event_for("Collecting numba>=0.61") == {
        "step_label": "Installing dependencies",
        "status": "Collecting numba>=0.61",
    }


def test_cached_wheel_says_so_instead_of_downloading():
    """A reinstall is served from pip's HTTP cache - no Progress lines follow, so the label must not
    promise a download percentage that will never come."""
    tracker = PipProgressTracker("Installing dependencies")
    event = tracker.event_for("  Using cached torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl (899.7 MB)")
    assert event["step_label"] == "Installing dependencies — using cached torch (899.7 MB)"


def test_metadata_fetches_do_not_replace_the_label():
    tracker = PipProgressTracker("Installing dependencies")
    event = tracker.event_for("  Downloading torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl.metadata (29 kB)")
    assert event["step_label"] == "Installing dependencies"
    assert "metadata" in event["status"]
