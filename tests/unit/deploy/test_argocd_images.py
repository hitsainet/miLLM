"""T24 (040 FR-040.27; FTDD §6.9): Argo CD Image Updater pins exactly the images the base deploys, and
the gitops sync never overwrites a pin file.

* Image Updater rewrites a kustomization image by NAME (`<alias>.kustomize.image-name`). An annotation
  naming an image the kustomization does not have pins nothing: the pod stays on `:latest` and no
  digest is ever written, silently. So the annotation set must equal the base kustomization's
  `images[].name` set, the `image-list` must track those repositories, and the `ImageUpdater`
  resource must select this Application.
* Image Updater writes its pins to `<source path>/.argocd-source-<application>.yaml` on gitops/pilot.
  The sync workflow resets every other file to main's content, so the pin file must be excluded:
  the Application's (`miserver`). From P2 to P7 the old Application's pin file was excluded too;
  P7 (FR-040.40) removed that exclusion, and it must not come back.

Control NC-82: leave `hitsai/millm-backend` in one annotation → red.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "k8s" / "argocd" / "miserver-app.yaml"
WORKFLOW = ROOT / ".github" / "workflows" / "sync-main-to-gitops-pilot.yml"
PREFIX = "argocd-image-updater.argoproj.io/"
#: The pin file the pre-P2 Application `millm` wrote; excluded from the sync until P7 (FTDD §6.9).
OLD_PIN = "k8s/base/.argocd-source-millm.yaml"


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if isinstance(d, dict)]


def _application() -> dict:
    (app,) = [d for d in _docs(APP) if d["kind"] == "Application"]
    return app


def _kustomization_images() -> set[str]:
    path = ROOT / _application()["spec"]["source"]["path"] / "kustomization.yaml"
    kustomization = yaml.safe_load(path.read_text())
    return {i["name"] for i in kustomization["images"]}


def test_image_updater_names_the_kustomization_images():
    app = _application()
    notes = app["metadata"]["annotations"]
    images = _kustomization_images()
    assert len(images) >= 2, images
    named = {v for k, v in notes.items() if k.startswith(PREFIX) and k.endswith(".kustomize.image-name")}
    assert named == images, {"annotations": named, "kustomization": images}
    aliases = {k[len(PREFIX):-len(".kustomize.image-name")] for k in notes
               if k.startswith(PREFIX) and k.endswith(".kustomize.image-name")}
    tracked = dict(item.strip().split("=", 1) for item in notes[PREFIX + "image-list"].split(","))
    assert set(tracked) == aliases, (tracked, aliases)
    for alias in aliases:
        repo = re.sub(r"^docker\.io/", "", tracked[alias]).rsplit(":", 1)[0]
        assert repo == notes[f"{PREFIX}{alias}.kustomize.image-name"], (alias, repo)
        assert notes[f"{PREFIX}{alias}.update-strategy"] == "digest", alias
    # The CRD-based Image Updater reconciles only the Applications an ImageUpdater selects.
    (updater,) = [d for d in _docs(APP) if d["kind"] == "ImageUpdater"]
    patterns = [r["namePattern"] for r in updater["spec"]["applicationRefs"]]
    assert any(fnmatch.fnmatchcase(app["metadata"]["name"], p) for p in patterns), patterns
    assert not any(fnmatch.fnmatchcase("millm", p) for p in patterns), patterns


def _excluded() -> set[str]:
    text = WORKFLOW.read_text()
    return set(re.findall(r"':\(exclude\)([^']+)'", text))


def test_gitops_pilot_excludes_the_pin_files():
    app = _application()
    new_pin = f"{app['spec']['source']['path']}/.argocd-source-{app['metadata']['name']}.yaml"
    assert new_pin == "k8s/base/.argocd-source-miserver.yaml", new_pin
    excluded = _excluded()
    assert excluded == {new_pin, "k8s/open-webui/.argocd-source-open-webui.yaml"}, excluded
    assert OLD_PIN not in excluded, "040 P7 removed the old Application's pin file exclusion"
    # The exclusions sit on the one converge command, not in a comment or another step.
    step = next(s for s in yaml.safe_load(WORKFLOW.read_text())["jobs"]["sync"]["steps"]
                if "git checkout origin/main -- ." in (s.get("run") or ""))
    assert f"':(exclude){new_pin}'" in step["run"], new_pin
