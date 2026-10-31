"""torch is pinned EXACTLY, to the build the serving image runs and miForge pins (2026-10-17).

miForge's hot-update attach (F11) refuses when the trainer's and the server's torch/CUDA builds
differ: a CUDA IPC handle is a contract between matching builds. miForge pins `torch==2.14.1`
(`backend/Dockerfile` `ARG TORCH_VERSION`). miLLM floated on `>=2.10.0,<3`, so the two matched
only by build date, and the next rebuild after a torch release would have broken every attach
with nothing in either repo changing.

2.14.1 is what the serving image runs: measured in the pod on 2026-10-16 (2.14.1+cu130,
`0xcc/reviews/attachable_weights_hardware_2026-10-16.md` §1), and the latest release
(2026-09-30), so every image built since resolved to it — the pin changes no torch at deploy.

Three places must agree, or something breaks quietly:
* the dependency in `pyproject.toml` (what the image installs);
* the CPU pre-install in CI (`backend-tests.yml`) — pinned below the dependency it is replaced
  and pip pulls the full CUDA stack onto the runner (the disk-exhaustion of an earlier CI run);
* miForge's `ARG TORCH_VERSION` (the sibling; skips loudly when absent, fails under
  `MILLM_REQUIRE_CROSS_REPO_CHECKS=1`).
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

from tests.support.sibling_repos import REPO_ROOT, sibling_repo

#: The serving image's measured torch (hardware record §1), without the local `+cu130` label.
SERVED = "2.14.1"


def _torch_requirements() -> list[str]:
    deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    return [d for d in deps if re.match(r"torch\s*[<>=!~]", d) or d.strip() == "torch"]


def test_the_dependency_is_one_exact_pin_on_the_served_build():
    assert _torch_requirements() == [f"torch=={SERVED}"], (
        "torch must be pinned EXACTLY: miForge's attach refuses on a torch mismatch, and a range "
        "lets the next image rebuild move it"
    )


def test_the_hardware_record_still_states_that_build():
    record = REPO_ROOT / "0xcc/reviews/attachable_weights_hardware_2026-10-16.md"
    if not record.is_file():
        pytest.skip("0xcc/ is stripped on the public mirror: the measured build is not here")
    measured = re.search(r"\|\s*torch / CUDA / cuDNN\s*\|\s*([0-9][^\s/|]*)", record.read_text())
    assert measured and measured.group(1).split("+")[0] == SERVED


def test_ci_pre_installs_the_same_torch():
    workflow = (REPO_ROOT / ".github/workflows/backend-tests.yml").read_text()
    installs = re.findall(r'pip install "(torch[^"]*)" --index-url', workflow)
    assert installs, "no CPU torch pre-install found in backend-tests.yml"
    assert set(installs) == {f"torch=={SERVED}"}, installs


def test_miforge_pins_the_same_torch():
    dockerfile = sibling_repo("miForge", "MIFORGE_REPO") / "backend" / "Dockerfile"
    if not dockerfile.is_file():
        message = f"miForge not checked out at {dockerfile.parents[1]}: torch pins NOT compared"
        if os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    pin = re.search(r"^ARG TORCH_VERSION=(\S+)$", dockerfile.read_text(), re.M)
    assert pin, "miForge's backend/Dockerfile no longer carries ARG TORCH_VERSION"
    assert pin.group(1) == SERVED, (
        f"miForge pins torch {pin.group(1)}, miLLM {SERVED}: hot-update attach would refuse"
    )


# ---------------------------------------------------------------------------------------------
# The whole serving stack (2026-10-17). Pinning torch alone still let any edit to pyproject.toml —
# which invalidates the image's cached dependency layer — re-resolve everything else unpinned:
# transformers would have moved the very next build. Every package below is `==` the version
# `pip list` reported in the live millm-backend pod on 2026-10-17 (coordinator reading).
# ---------------------------------------------------------------------------------------------

#: name -> version the live pod runs (`pip list`, deploy/millm-backend, 2026-10-17), except
#: huggingface-hub 1.33.0 -> 2.2.0 and the new hf-xet pin (operator directive 2026-10-18: "update ALL
#: of the huggingface software to the latest"), which the pod runs from the next image build, and
#: transformers 5.20.1 -> 5.20.2 (operator rule 2026-10-21: every package at its latest release;
#: deps_sync_2026-10-21.md). constraints.txt carries the same versions (test_dependency_lock.py).
SERVING_STACK = {
    "torch": "2.14.1",
    "transformers": "5.20.2",
    "accelerate": "1.15.0",
    "huggingface-hub": "2.2.0",
    "hf-xet": "1.7.0",
    "tokenizers": "0.23.3",
    "xgrammar": "0.2.8",
    "safetensors": "0.8.0",
    "bitsandbytes": "0.50.2",
}
#: The gguf extra (and the Dockerfile's separate CUDA-wheel install).
LLAMA_CPP = "0.3.36"


def _requirements(names: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {n: [] for n in names}
    for dep in _all_requirements():
        name = re.split(r"[<>=!~\[;\s]", dep, maxsplit=1)[0].strip().lower().replace("_", "-")
        if name in found:
            found[name].append(dep.replace(" ", ""))
    return found


def _all_requirements() -> list[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    deps = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        deps.extend(extra)
    return deps


@pytest.mark.parametrize("name", sorted(SERVING_STACK))
def test_each_serving_package_is_pinned_to_the_pods_version(name):
    found = _requirements([name])[name]
    assert found == [f"{name}=={SERVING_STACK[name]}"], (
        f"{name} must be pinned EXACTLY to {SERVING_STACK[name]} (the live pod's version); "
        f"pyproject.toml has {found or 'nothing'} — a range re-resolves on the next image build"
    )


def test_llama_cpp_is_pinned_in_the_extra_and_the_image_alike():
    assert _requirements(["llama-cpp-python"])["llama-cpp-python"] == [
        f"llama-cpp-python=={LLAMA_CPP}"
    ]
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    installs = re.findall(r'"llama-cpp-python([^"]*)"', dockerfile)
    assert installs == [f"=={LLAMA_CPP}"], installs


def test_no_dockerfile_comment_claims_a_torch_the_image_does_not_run():
    """The comments said "this image's torch (2.13.0+cu130 — verified in the running pod)"
    while the pod ran 2.14.1: a stale fact stated as verified is worse than none."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "this image's torch (2.13" not in dockerfile
    assert f"{SERVED}+cu130" in dockerfile
