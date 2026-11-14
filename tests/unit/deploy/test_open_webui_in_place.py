"""T23 (040 FR-040.26; S001 FR-S001.26): Open WebUI is renamed nowhere and moves nowhere.

Its Application, namespace and volume claim keep their names and uids (S001 2.65 compares them with
P0), so its chats, users and settings stay. Only its pointer to the server changes, and that is a
separate config-only push, P2-O at window minute T62, after checks 1–4 (040 FTASKS 10.18–10.19): the
P2-W manifests leave `OPENAI_API_BASE_URL` on the old in-cluster address, which keeps answering
through `k8s/transition/`. Its saved connection in Open WebUI's own database is changed by hand
before P7 (rehearsal 2.16: the URL is SAVED, so the environment value alone does not reach it).

Control NC-81: rename the Application → red.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OWUI = ROOT / "k8s" / "open-webui"
NEW_BASE_URL = "http://miserver-backend.miserver.svc.cluster.local:8000/v1"


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if isinstance(d, dict)]


def test_name_and_namespace_unchanged():
    (app,) = _docs(OWUI / "argocd-application.yaml")
    assert app["kind"] == "Application"
    assert app["metadata"]["name"] == "open-webui"
    assert app["spec"]["destination"]["namespace"] == "open-webui-ns"
    assert app["spec"]["source"]["path"] == "k8s/open-webui"
    kustomization = yaml.safe_load((OWUI / "kustomization.yaml").read_text())
    assert kustomization["namespace"] == "open-webui-ns"
    (ns,) = _docs(OWUI / "namespace.yaml")
    assert ns["metadata"]["name"] == "open-webui-ns"
    (pvc,) = _docs(OWUI / "pvc.yaml")
    assert pvc["metadata"]["name"] == "open-webui-pvc"
    (deploy,) = _docs(OWUI / "deployment.yaml")
    assert deploy["metadata"]["name"] == "open-webui"
    claims = [v["persistentVolumeClaim"]["claimName"] for v in deploy["spec"]["template"]["spec"]["volumes"]]
    assert claims == ["open-webui-pvc"]


def test_base_url_names_miserver():
    (deploy,) = _docs(OWUI / "deployment.yaml")
    (container,) = deploy["spec"]["template"]["spec"]["containers"]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["OPENAI_API_BASE_URL"] == NEW_BASE_URL
