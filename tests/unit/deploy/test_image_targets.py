"""T17 (040 FR-040.19, FR-040.27, FR-040.40; DR-27; TPD-10).

From P1 to P7 every image was built once and tagged in two Docker Hub repositories, the new
`hitsai/miserver-*` and the old `hitsai/millm-*`. P7 stopped the dual push: an image publishes to
its new repository only. The old repositories stay on Docker Hub with every tag they hold (DR-27 is
`frozen`: deployed pins and the public mirror's users may pull them), they just no longer advance.

`test_every_image_still_publishes_to_its_old_repository` is DR-27's exercise; the name is census's
`EXERCISE` key, so it stays until S001 drops or renames it. Since P7 it is a register-only check
(FTASKS 15.3): the export still records the old repositories as frozen, and no workflow writes to
them.

The workflow's own shell runs: `repo_for` and the whole detect step, the build step's case statement and the
tag step, all extracted from the parsed YAML, so an edit to any of them is what is tested. NC-70: drop
`hitsai/miserver-backend` from `repo_for` and `test_each_image_maps_to_its_repository` goes red; put
`hitsai/millm-backend` back and the same test goes red.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.support import legacy_identifiers as li
from tests.support.package_root import package_name

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "docker-images.yml"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
IMAGES = {"backend": "hitsai/millm-backend", "admin-ui": "hitsai/millm-admin-ui"}
NEW_IMAGES = {"backend": "hitsai/miserver-backend", "admin-ui": "hitsai/miserver-admin-ui"}
SHA = "0123456789abcdef0123456789abcdef01234567"
BASE_PATH = "/usr/bin:/bin"


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def _steps() -> dict[str, dict]:
    return {s.get("id") or s.get("name"): s for j in _jobs().values() for s in j.get("steps", [])}


def _repo_for(image: str) -> list[str]:
    run = _steps()["detect"]["run"]
    fn = re.search(r"^ *repo_for\(\) \{.*?^ *\}$", run, flags=re.M | re.S)
    assert fn, "repo_for() not found in the detect step"
    out = subprocess.run(["bash", "-c", fn.group(0) + f'\nrepo_for "{image}"'],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _img_outputs(image: str, tmp_path: Path) -> dict[str, str]:
    run = _steps()["img"]["run"].replace("${{ matrix.name }}", image)
    out = tmp_path / f"img-{image}"
    subprocess.run(["bash", "-c", run], env={"GITHUB_OUTPUT": str(out), "PATH": BASE_PATH},
                   check=True, capture_output=True, text=True)
    return _outputs(out)


def _build_repos(image: str, tmp_path: Path) -> list[str]:
    return [v for k, v in _img_outputs(image, tmp_path).items() if k in ("repo", "repos", "legacy_repo")]


def _tags(image: str, ref: str, tmp_path: Path) -> list[str]:
    """Run the tag step with the build step's real outputs substituted, as Actions would."""
    run = _steps()["version"]["run"]
    for key, value in _img_outputs(image, tmp_path).items():
        run = run.replace("${{ steps.img.outputs.%s }}" % key, value)
    assert "${{" not in run, f"the tag step reads an expression this test does not resolve: {run}"
    out = tmp_path / f"version-{image}"
    subprocess.run(["bash", "-c", run], check=True, capture_output=True, text=True,
                   env={"GITHUB_OUTPUT": str(out), "GITHUB_REF": ref, "GITHUB_SHA": SHA, "PATH": BASE_PATH})
    return _outputs(out)["tags"].split(",")


def test_every_image_still_publishes_to_its_old_repository(tmp_path):
    """DR-27, register-only since P7: the old repositories are recorded as frozen (kept on Docker
    Hub), and nothing in the build writes to them any more."""
    export = li.load()
    (record,) = [r for r in export.entries() if r["id"] == "DR-27"]
    assert record["status"] == "frozen", record
    assert set(IMAGES.values()) <= set(record["literals"]), record["literals"]
    for image, old in IMAGES.items():
        assert old not in _repo_for(image), (image, _repo_for(image))
        assert old not in _build_repos(image, tmp_path), (image, _build_repos(image, tmp_path))
        assert not [t for t in _tags(image, "refs/heads/main", tmp_path) if t.startswith(f"{old}:")], image
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert len(workflows) >= 5, workflows
    for path in workflows:
        for old in IMAGES.values():
            assert old not in path.read_text(), f"{path.name} still names {old}"


def test_each_image_maps_to_its_repository(tmp_path):
    for image, new in NEW_IMAGES.items():
        # detect: the one repository, and nothing else.
        assert _repo_for(image) == [new], (image, _repo_for(image))
        # build: the same, so detect and build cannot disagree about where an image lives.
        outputs = _img_outputs(image, tmp_path)
        assert outputs.get("repo") == new and "legacy_repo" not in outputs, (image, outputs)
        # an exact set, not a subset.
        assert sorted(_tags(image, "refs/heads/main", tmp_path)) == sorted(
            [f"{new}:latest", f"{new}:{SHA[:8]}"]), image

    build = _jobs()["build"]["steps"]
    push = next(s for s in build if s.get("id") == "build")
    assert push["with"]["tags"] == "${{ steps.version.outputs.tags }}"
    assert push["with"]["push"] is True
    attests = [s for s in build if str(s.get("uses", "")).startswith("actions/attest-build-provenance@")]
    assert [a["with"]["subject-name"] for a in attests] == ["index.docker.io/${{ steps.img.outputs.repo }}"]
    for a in attests:
        assert a["with"]["subject-digest"] == "${{ steps.build.outputs.digest }}"
        assert a["with"]["push-to-registry"] is True


def test_release_names_the_new_images(tmp_path):
    body = yaml.safe_load(RELEASE.read_text())["jobs"]["release"]["steps"]
    text = next(s for s in body if s.get("uses", "").startswith("softprops/action-gh-release@"))["with"]["body"]
    pulls = set(re.findall(r"^docker pull (.+?)\s*$", text, flags=re.M))
    assert pulls == {f"{repo}:${{{{ github.ref_name }}}}" for repo in NEW_IMAGES.values()}, pulls
    # The images those lines name are the ones a tag push publishes.
    for image, new in NEW_IMAGES.items():
        assert sorted(_tags(image, "refs/tags/v0.6.0", tmp_path)) == sorted(
            [f"{new}:latest", f"{new}:v0.6.0"]), image


# ── detect reads the published revision ──────────────────────────────────────────────────────
# The detect step runs as written, against a stand-in `curl` that answers Docker Hub's three calls
# (token, manifest, config blob) from a table, and a two-commit git repository: BASE, then HEAD,
# which changes only backend paths. A repository missing from the table answers like a 404. Every
# old repository is in the table at HEAD, so a detect that still read one would change nothing:
# the cases below decide on the new repositories alone.

FAKE_CURL = r"""#!/bin/bash
url="${@: -1}"
case "$url" in
  *auth.docker.io*) repo="${url#*scope=repository:}"; repo="${repo%:pull}" ;;
  *) repo="${url#*/v2/}"; repo="${repo%/manifests/*}"; repo="${repo%/blobs/*}" ;;
esac
rev="$(awk -v r="$repo" '$1==r {print $2}' "$FAKE_REVS")"
[ -n "$rev" ] || exit 22
case "$url" in
  *auth.docker.io*) echo '{"token":"t"}' ;;
  */manifests/*) echo '{"mediaType":"application/vnd.oci.image.manifest.v1+json","config":{"digest":"sha256:c"}}' ;;
  */blobs/*) printf '{"config":{"Labels":{"org.opencontainers.image.revision":"%s"}}}\n' "$rev" ;;
  *) exit 22 ;;
esac
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def history(tmp_path):
    repo = tmp_path / "repo"
    (repo / package_name()).mkdir(parents=True)
    (repo / "admin-ui").mkdir()
    (repo / package_name() / "a.py").write_text("a = 1\n")
    (repo / "admin-ui" / "x.ts").write_text("x\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / package_name() / "a.py").write_text("a = 2\n")
    _git(repo, "commit", "-q", "-am", "backend change")
    return repo, base, _git(repo, "rev-parse", "HEAD")


def _detect(repo: Path, tmp_path: Path, revs: dict[str, str]) -> tuple[list[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "curl").write_text(FAKE_CURL)
    (bin_dir / "curl").chmod(0o755)
    table = tmp_path / "revs"
    table.write_text("".join(f"{r} {v}\n" for r, v in revs.items()))
    out = tmp_path / "detect-out"
    out.write_text("")
    proc = subprocess.run(["bash", "-c", _steps()["detect"]["run"]], cwd=repo, capture_output=True, text=True,
                          env={"PATH": f"{bin_dir}:{BASE_PATH}", "FAKE_REVS": str(table), "GITHUB_OUTPUT": str(out),
                               "REF": "refs/heads/main", "CREATED": "", "GITHUB_SHA": _git(repo, "rev-parse", "HEAD"),
                               "HOME": str(tmp_path)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = next(l for l in out.read_text().splitlines() if l.startswith("images="))
    return json.loads(line.removeprefix("images=")), proc.stdout


def _all(rev_new: str | None) -> dict[str, str]:
    """Every new repository's published revision (None: a 404); the old ones stale at "0" * 40."""
    revs = {r: rev_new for r in NEW_IMAGES.values() if rev_new}
    return {**{r: "0" * 40 for r in IMAGES.values()}, **revs}


@pytest.mark.parametrize("case, expected", [
    # Control: published from BASE; only backend paths changed since → only backend.
    ("at_base", ["backend"]),
    # Published from HEAD → nothing, whatever the old repositories hold.
    ("at_head", []),
    # The new repositories lost their images → every image builds.
    ("new_missing", ["backend", "admin-ui"]),
    # One new repository lost its image → it builds, even though unchanged.
    ("admin_missing", ["backend", "admin-ui"]),
])
def test_detect_reads_the_new_repository_only(history, tmp_path, case, expected):
    repo, base, head = history
    revs = {
        "at_base": _all(base),
        "at_head": _all(head),
        "new_missing": _all(None),
        "admin_missing": {k: v for k, v in _all(base).items() if k != "hitsai/miserver-admin-ui"},
    }[case]
    images, log = _detect(repo, tmp_path, revs)
    assert images == expected, log
