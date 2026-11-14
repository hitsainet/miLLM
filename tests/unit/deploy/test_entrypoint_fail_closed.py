"""P4 (041 FR-041.6, TID §3.4): a failed migration stops the container.

`docker-entrypoint.sh` runs `python -m alembic upgrade head` under `set -e` and then `exec "$@"`. A
migration that fails must stop there: a server started on a schema it does not match corrupts data or
fails later in a way that looks like something else. The script used to carry a "Migration failed, but
continuing startup" branch that `set -e` made unreachable; it is gone, and this test makes sure that
nothing brings the behaviour back.

The real script runs in bash with stub `python3` (the database wait passes at once) and `python` (logs its
arguments; fails when they name alembic, or succeeds in the second test) on `PATH`, and `APP_DIR`
pointing at a temporary directory. The command the entrypoint would `exec` writes a sentinel file.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.support.package_root import REPO_ROOT

ENTRYPOINT = REPO_ROOT / "docker-entrypoint.sh"


def _run(tmp_path: Path, migration_exit: int) -> tuple[subprocess.CompletedProcess, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "python.log"
    sentinel = tmp_path / "started"
    (bin_dir / "python3").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "python").write_text(
        "#!/bin/sh\n"
        f'echo "$PWD: $*" >> "{log}"\n'
        f'case "$*" in *alembic*) exit {migration_exit} ;; esac\n'
        "exit 0\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "APP_DIR": str(tmp_path),
        "DATABASE_URL": "postgresql+asyncpg://u:p@127.0.0.1:1/x",
        "SENTINEL": str(sentinel),
        "HOME": os.environ.get("HOME", str(tmp_path)),
    }
    proc = subprocess.run(["bash", str(ENTRYPOINT), "/bin/sh", "-c", 'touch "$SENTINEL"'],
                          env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    return proc, log, sentinel


def test_a_failed_migration_stops_the_container(tmp_path):
    proc, log, sentinel = _run(tmp_path, migration_exit=1)
    calls = log.read_text() if log.exists() else ""
    assert f"{tmp_path}: -m alembic upgrade head" in calls, (
        f"the entrypoint never ran the migration in APP_DIR; python calls: {calls!r}\n{proc.stdout}")
    assert proc.returncode != 0, f"the entrypoint exited 0 after a failed migration:\n{proc.stdout}"
    assert not sentinel.exists(), f"the server command ran after a failed migration:\n{proc.stdout}"


def test_a_successful_migration_execs_the_command(tmp_path):
    """The same harness with a migration that succeeds: the command runs, so the test above fails for
    the right reason and not because the harness never reaches the `exec`."""
    proc, log, sentinel = _run(tmp_path, migration_exit=0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"{tmp_path}: -m alembic upgrade head" in log.read_text()
    assert sentinel.exists(), proc.stdout
