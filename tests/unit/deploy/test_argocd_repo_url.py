"""040 P1 (FTASKS 8.3; FR-040.26, FR-040.27; S001 §11.2 step 4.5): the Argo CD manifests in git name the
renamed repository.

Runbook 1.10/1.11 patch the live AppProjects and Applications to `Onegaishimas/miServer.git`; this commit
makes git say the same, so a later `kubectl apply` of these files does not put the old URL back. An
Application whose `repoURL` its AppProject does not allow is an `InvalidSpecError`, so the pair is checked
together. P2 renamed the files and objects themselves (FTDD §6.9, 040 FTASKS 10.3): the set below changed
then, on purpose.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
NEW_URL = "https://github.com/Onegaishimas/miServer.git"
MANIFESTS = [ROOT / "k8s" / "argocd" / "miserver-app.yaml", ROOT / "k8s" / "argocd" / "miserver-otel-app.yaml",
             ROOT / "k8s" / "open-webui" / "argocd-application.yaml"]


def _docs(kind: str) -> dict[str, dict]:
    return {d["metadata"]["name"]: d for path in MANIFESTS for d in yaml.safe_load_all(path.read_text())
            if d and d.get("kind") == kind and d.get("apiVersion", "").startswith("argoproj.io/")}


def test_these_are_every_argocd_manifest_in_the_repository():
    found = {p for p in (ROOT / "k8s").rglob("*.yaml")
             if any(d and d.get("kind") in ("Application", "AppProject")
                    for d in yaml.safe_load_all(p.read_text()))}
    assert found == set(MANIFESTS), sorted(map(str, found))


def test_every_application_sources_the_renamed_repository():
    apps = _docs("Application")
    assert set(apps) == {"miserver", "miserver-otel", "open-webui"}, sorted(apps)
    for name, app in apps.items():
        assert app["spec"]["source"]["repoURL"] == NEW_URL, name


def test_every_project_allows_exactly_the_renamed_repository():
    projects = _docs("AppProject")
    assert set(projects) == {"miserver", "miserver-otel"}, sorted(projects)
    for name, project in projects.items():
        assert project["spec"]["sourceRepos"] == [NEW_URL], name
    for name, app in _docs("Application").items():
        project = projects.get(app["spec"]["project"])
        if project is not None:  # `default` lives in the cluster (runbook 1.04 reads it)
            assert app["spec"]["source"]["repoURL"] in project["spec"]["sourceRepos"], name
