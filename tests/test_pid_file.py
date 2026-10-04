"""app/pid_file.py: the primary server's self-written PID file."""

import os

from app.pid_file import PidFile


def test_write_stores_this_processs_pid_and_remove_deletes_it(tmp_path):
    path = tmp_path / "run" / "pairing.pid"
    pid_file = PidFile(path)

    pid_file.write()
    assert path.read_text() == str(os.getpid())

    pid_file.remove()
    assert not path.exists()
    pid_file.remove()  # already gone: no error
