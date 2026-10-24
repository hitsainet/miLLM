"""hostIPC is opt-in: the default deployment never has it, the component always adds it (D1).

Operator decision D1 (suite tracker, 2026-10-16): miForge's hot-update needs CUDA IPC between the
trainer pod and this backend pod, so both set `hostIPC: true` — shipped as an OPT-IN kustomize
Component (`k8s/components/hostipc`, applied by `k8s/overlays/hostipc`), never in what ArgoCD
deploys by default (`k8s/base`). Two ways this could go wrong silently, one test each:

* the default build gains hostIPC (someone references the component from `k8s/base`, or writes
  the field straight into `backend.yaml`) — every deploy then widens the pod's reach, opted in or
  not;
* the component stops setting it on the backend (a renamed Deployment makes the patch's target
  match nothing, and kustomize builds happily) — the operator opts in and gets nothing.

The rendered checks run `kubectl kustomize` (installed on GitHub's ubuntu runners and here). They
skip LOUDLY without it, so the structural checks below — which need no binary — still hold the
line: what ArgoCD's Application points at, and what the patch targets.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "k8s" / "base"
OVERLAY = ROOT / "k8s" / "overlays" / "hostipc"
COMPONENT = ROOT / "k8s" / "components" / "hostipc"
ARGOCD_APP = ROOT / "k8s" / "argocd" / "millm-app.yaml"
DOCKERFILE = ROOT / "Dockerfile"
#: The uid/gid the image's `millm` user is created with, and the one the component declares.
#: miForge's hostipc component runs its worker-train as this uid (cross-checked there).
MILLM_UID = 1000


def _render(path: Path) -> list[dict]:
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        pytest.skip("kubectl not installed: the RENDERED hostIPC checks did NOT run")
    out = subprocess.run(
        [kubectl, "kustomize", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def _host_ipc_by_workload(docs: list[dict]) -> dict[str, object]:
    """{kind/name: hostIPC value or None} for every pod-template workload."""
    found = {}
    for d in docs:
        spec = ((d.get("spec") or {}).get("template") or {}).get("spec")
        if spec is not None:
            found[f"{d['kind']}/{d['metadata']['name']}"] = spec.get("hostIPC")
    return found


class TestRendered:
    def test_the_default_build_has_no_hostipc_anywhere(self):
        workloads = _host_ipc_by_workload(_render(BASE))
        assert "Deployment/millm-backend" in workloads, workloads
        assert {k: v for k, v in workloads.items() if v} == {}, (
            "k8s/base — what ArgoCD deploys — sets hostIPC; it must stay opt-in (D1)"
        )

    def test_the_overlay_sets_hostipc_on_the_backend_and_nothing_else(self):
        workloads = _host_ipc_by_workload(_render(OVERLAY))
        assert workloads["Deployment/millm-backend"] is True
        others = {k: v for k, v in workloads.items() if v and k != "Deployment/millm-backend"}
        assert others == {}, f"hostIPC leaked onto other workloads: {others}"

    def test_the_overlay_states_millms_uid_on_the_backend_container_only(self):
        """The uid pairing (step 2): CUDA's /dev/shm IPC files are 0600, so miForge's
        worker must run as this uid; the component declares it so both sides can be compared."""
        docs = _render(OVERLAY)
        backend = next(d for d in docs if d.get("kind") == "Deployment"
                       and d["metadata"]["name"] == "millm-backend")
        containers = {c["name"]: c for c in backend["spec"]["template"]["spec"]["containers"]}
        assert containers["backend"]["securityContext"] == {
            "runAsUser": MILLM_UID, "runAsGroup": MILLM_UID,
        }
        assert "securityContext" not in backend["spec"]["template"]["spec"], (
            "pod-level: it would also apply to the root fix-permissions init container"
        )

    def test_the_default_build_declares_no_uid_override(self):
        for d in _render(BASE):
            for c in ((d.get("spec") or {}).get("template") or {}).get("spec", {}).get(
                "containers", []
            ):
                assert "runAsUser" not in (c.get("securityContext") or {}), c["name"]

    def test_the_overlay_changes_nothing_else(self):
        """The component is ONE field: everything else renders identically to the default."""
        base, overlay = _render(BASE), _render(OVERLAY)
        for d in overlay:
            spec = ((d.get("spec") or {}).get("template") or {}).get("spec")
            if spec is not None:
                spec.pop("hostIPC", None)
                for c in spec.get("containers", []):
                    if c.get("name") == "backend":
                        c.pop("securityContext", None)
        assert overlay == base


class TestStructure:
    """No binary needed: these hold even where kubectl is absent."""

    def test_argocd_deploys_the_base_not_the_overlay(self):
        apps = [d for d in yaml.safe_load_all(ARGOCD_APP.read_text())
                if d and d.get("kind") == "Application"]
        assert apps, "no ArgoCD Application in millm-app.yaml"
        assert [a["spec"]["source"]["path"] for a in apps] == ["k8s/base"]

    def test_the_base_references_no_component_and_sets_no_hostipc(self):
        kustomization = yaml.safe_load((BASE / "kustomization.yaml").read_text())
        assert "components" not in kustomization
        assert "hostIPC" not in (BASE / "backend.yaml").read_text()

    def test_the_component_patches_the_backend_deployment_that_exists(self):
        component = yaml.safe_load((COMPONENT / "kustomization.yaml").read_text())
        assert component["kind"] == "Component"
        (patch,) = component["patches"]
        assert patch["target"]["kind"] == "Deployment"
        names = [
            d["metadata"]["name"]
            for d in yaml.safe_load_all((BASE / "backend.yaml").read_text())
            if d and d.get("kind") == "Deployment"
        ]
        assert patch["target"]["name"] in names
        body = yaml.safe_load(patch["patch"])
        pod = body["spec"]["template"]["spec"]
        assert pod["hostIPC"] is True
        assert pod["containers"] == [
            {"name": "backend", "securityContext": {"runAsUser": MILLM_UID, "runAsGroup": MILLM_UID}}
        ]

    def test_the_overlay_is_base_plus_the_component(self):
        overlay = yaml.safe_load((OVERLAY / "kustomization.yaml").read_text())
        assert overlay["resources"] == ["../../base"]
        assert overlay["components"] == ["../../components/hostipc"]

    def test_the_image_user_is_pinned_to_the_uid_the_component_declares(self):
        """Not useradd's "first free id": a base-image change could move it and break the
        pairing silently - the CUDA IPC open would fail exactly as F11 14.3 did."""
        text = DOCKERFILE.read_text()
        assert f"groupadd --gid {MILLM_UID} millm" in text
        assert f"useradd --uid {MILLM_UID} --gid {MILLM_UID}" in text
