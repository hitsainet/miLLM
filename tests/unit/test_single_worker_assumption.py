"""The arm lock's precondition lives with it (039 FTASKS 2.15, TD39-9; suite memory
`a-guards-precondition-must-live-with-it`).

`ProbeArmingService` serialises set commits with an IN-PROCESS lock. That is sufficient only while
ONE process serves the API: one replica, the `Recreate` strategy (never two pods during a rollout),
and one uvicorn worker. This reads the deployed manifest and the image's command; a change to
any of them turns this red, and the lock must then become a database-level one.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def test_one_replica_recreated_never_overlapping():
    docs = list(yaml.safe_load_all((REPO / "k8s" / "base" / "backend.yaml").read_text()))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"]["type"] == "Recreate"


def test_one_uvicorn_worker():
    dockerfile = (REPO / "Dockerfile").read_text()
    commands = [line for line in dockerfile.splitlines()
                if line.strip().startswith("CMD") and "uvicorn" in line]
    assert commands, "no uvicorn CMD in the Dockerfile — this guard has stopped looking"
    for command in commands:  # every stage (production and development)
        workers = re.findall(r"--workers[\"',\s=]+(\d+)", command)
        assert all(int(n) == 1 for n in workers), command
