"""P9 (041 FR-041.8, TID §3.7; R13): the image's OS user is `miserver`, uid and gid 1000.

The OS user is a decision of its own (R13), not a consequence of the package name, so it is a constant
here, not `package_name()`. What must not move is the NUMBER: the manifests run the backend as 1000 and
the rule-runner sidecar as 1001 in group 1000, the volumes are owned by 1000, and the hostIPC component
pairs CUDA IPC by uid. A rename that changed a uid would fail only at run time, on a permission.

The first three tests were `xfail(strict=True)` from C5 until P8's rename commit (R), which renamed the
user and removed the marks.
"""

from __future__ import annotations

import re
import shlex

import yaml

from tests.support.package_root import LEGACY_PACKAGES, REPO_ROOT

IMAGE_USER = "miserver"
IMAGE_UID = IMAGE_GID = 1000
SIDECAR_UID = 1001


def _instructions() -> list[tuple[str, str]]:
    """(INSTRUCTION, argument text) of every Dockerfile instruction, continuations joined."""
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", " ", text)
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, _, rest = line.partition(" ")
        out.append((word.upper(), rest.strip()))
    assert len(out) > 20, "the Dockerfile parsed to almost nothing"
    return out


def _commands() -> list[list[str]]:
    """Every shell command of every RUN, split on `&&`, `||` and `;`."""
    out = []
    for word, rest in _instructions():
        if word == "RUN":
            for part in re.split(r"&&|\|\||;", rest):
                if part.strip():
                    try:
                        out.append(shlex.split(part))
                    except ValueError:  # a quote opened in one `&&` part and closed in another
                        out.append(part.split())
    return out


def _user_commands(name: str) -> list[list[str]]:
    return [c for c in _commands() if c and c[0] == name]


def test_user_and_group_are_miserver_uid_gid_1000():
    groups = _user_commands("groupadd")
    users = _user_commands("useradd")
    assert groups == [["groupadd", "--gid", str(IMAGE_GID), IMAGE_USER]], groups
    assert len(users) == 1, users
    user = users[0]
    assert user[-1] == IMAGE_USER, user
    assert user[user.index("--uid") + 1] == str(IMAGE_UID), user
    assert user[user.index("--gid") + 1] == str(IMAGE_GID), user


def test_every_user_directive_is_miserver():
    """Every stage that switches user ends as `miserver`; a stage may step to `root` in between (the
    development stage installs tools), never to any other user."""
    stages: list[list[str]] = []
    for word, rest in _instructions():
        if word == "FROM":
            stages.append([])
        elif word == "USER":
            stages[-1].append(rest)
    users = [u for stage in stages for u in stage]
    assert len(users) >= 2, users  # the runtime stage and the development stage
    assert set(users) <= {IMAGE_USER, "root"}, users
    assert all(stage[-1] == IMAGE_USER for stage in stages if stage), stages


def test_no_user_token_names_millm():
    found = [" ".join(cmd) for cmd in _commands()
             if cmd and cmd[0] in ("useradd", "groupadd", "chown")
             and any(part in LEGACY_PACKAGES for tok in cmd[1:] for part in tok.split(":"))]
    found += [f"USER {rest}" for word, rest in _instructions() if word == "USER" and rest in LEGACY_PACKAGES]
    assert not found, found


def _security_ids() -> list[tuple[str, str, int]]:
    """(file, key, value) of every runAsUser/runAsGroup in the manifests, patches included."""
    out = []
    for path in sorted((REPO_ROOT / "k8s").rglob("*.yaml")):
        for m in re.finditer(r"\b(runAsUser|runAsGroup):\s*(\S+)", path.read_text(encoding="utf-8")):
            value = yaml.safe_load(m.group(2))
            out.append((path.relative_to(REPO_ROOT).as_posix(), m.group(1), value))
    return out


def test_manifest_uids_are_numeric_1000_or_1001():
    ids = _security_ids()
    users = [(f, v) for f, k, v in ids if k == "runAsUser"]
    groups = [(f, v) for f, k, v in ids if k == "runAsGroup"]
    assert len(users) >= 2 and len(groups) >= 2, ids
    assert all(isinstance(v, int) and v in (IMAGE_UID, SIDECAR_UID) for _, v in users), users
    assert all(isinstance(v, int) and v == IMAGE_GID for _, v in groups), groups
    assert ("k8s/base/backend.yaml", SIDECAR_UID) in users, "the sidecar runs as 1001"
    assert ("k8s/components/hostipc/kustomization.yaml", IMAGE_UID) in users, "hostIPC pairs uid 1000"
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    compose_users = {name: svc["user"] for name, svc in compose["services"].items() if "user" in svc}
    assert compose_users == {"rule-runner": f"{SIDECAR_UID}:{IMAGE_GID}"}, compose_users
