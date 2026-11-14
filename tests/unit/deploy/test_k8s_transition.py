"""T22 (040 FR-040.23; PD-7) after P7 (FR-040.40): the old in-cluster address is retired.

From P2 to P7, `k8s/transition/` published a Service `millm-backend` in namespace `millm` of type
`ExternalName` pointing at the new Service, deployed by the `miserver` Application (the base included
`../transition`, and the AppProject allowed the `millm` namespace). P7 removed all three, on S001's
post-sweep reading (every consumer calls `miserver-backend.miserver.svc.cluster.local`).

`test_externalname_fields` is DR-54's exercise (the name is census's `EXERCISE` key, so it stays): it
now asserts the old namespace and the ExternalName are gone from everything the `miserver`
Application renders. Controls: put `../transition` back in the base, or the `millm` destination back
in the AppProject → red.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from miserver.core.legacy_names import HOST_ALIASES

ROOT = Path(__file__).resolve().parents[3]
TRANSITION = ROOT / "k8s" / "transition"
BASE = ROOT / "k8s" / "base"
APP = ROOT / "k8s" / "argocd" / "miserver-app.yaml"
OLD_NAMESPACE, OLD_SERVICE = "millm", "millm-backend"


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if isinstance(d, dict)]


def _kustomization(directory: Path) -> dict:
    return yaml.safe_load((directory / "kustomization.yaml").read_text())


def _base_objects() -> list[dict]:
    """Every object the base lists, read from its resource files (no kubectl needed)."""
    out = []
    for rel in _kustomization(BASE)["resources"]:
        path = (BASE / rel).resolve()
        if path.is_dir():
            out += [d for sub in _kustomization(path)["resources"] for d in _docs(path / sub)]
        else:
            out += _docs(path)
    return out


def test_externalname_fields():
    objects = _base_objects()
    assert len(objects) >= 10, f"only {len(objects)} objects read from the base: failing open"
    externals = [d for d in objects if d.get("kind") == "Service" and d["spec"].get("type") == "ExternalName"]
    assert externals == [], f"an ExternalName Service is still deployed: {externals}"
    old = [(d["kind"], d["metadata"].get("namespace"), d["metadata"]["name"]) for d in objects
           if d["metadata"].get("namespace") == OLD_NAMESPACE or d["metadata"]["name"] == OLD_SERVICE]
    assert old == [], f"objects still in the old namespace or under the old Service name: {old}"
    (svc,) = [d for d in _docs(BASE / "backend.yaml") if d["kind"] == "Service"]
    new = f"{svc['metadata']['name']}.{svc['metadata']['namespace']}.svc.cluster.local"
    assert new == "miserver-backend.miserver.svc.cluster.local", new
    assert not [a for a in HOST_ALIASES if a.via == "in_cluster"], "the in-cluster aliases are retired"


def test_appproject_allows_only_miserver():
    docs = {(d["kind"], d["metadata"]["name"]): d for d in _docs(APP)}
    (app,) = [d for (kind, _n), d in docs.items() if kind == "Application"]
    project = docs[("AppProject", app["spec"]["project"])]
    destinations = {d["namespace"] for d in project["spec"]["destinations"]}
    assert destinations == {app["spec"]["destination"]["namespace"]} == {"miserver"}, destinations
    assert app["spec"]["source"]["path"] == BASE.relative_to(ROOT).as_posix()


def test_base_no_longer_includes_transition():
    resources = _kustomization(BASE)["resources"]
    assert not [r for r in resources if "transition" in r], resources
    assert not TRANSITION.exists(), "k8s/transition/ is deleted at P7"
