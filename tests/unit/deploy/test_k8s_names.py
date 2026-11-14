"""T18 (040 FR-040.21, FR-040.24; FTDD §6.9): after P2 no Kubernetes object under `k8s/` is named,
placed or labelled `millm`, and no image is `hitsai/millm-*`.

Read PARSED from every manifest under `k8s/` (the raw files ArgoCD renders); `k8s/transition/`, which
answered under the old in-cluster name from P2 (PD-7), was deleted at P7 (FR-040.40) and, with it, the
one allowance. What is checked, per document that has a `kind`:

* `metadata.name`, `metadata.namespace`, label values and selector values;
* every reference to another object by name: `secretKeyRef`/`configMapKeyRef` names, `claimName`,
  `volumeName`, `storageClassName`, Ingress backend services, the `websocket-services` annotation,
  pod volume names, an Application's destination namespace and project, an AppProject's destinations,
  an ImageUpdater's `namePattern`;
* kustomization `namespace`, `images[].name`/`newName`, generator namespaces and patch targets;
* any string anywhere that names a `hitsai/millm-*` image (container images, Image Updater
  annotations).

Frozen values that are not object names (DR-19's database and role `millm`, DR-40's
`/run/millm-rules` and `millm-rule-runner.json`, DR-42's `millm-*.dump`) are values of other keys
and are not read here. Since P7 no old name is allowed (`ALLOWED` is empty).

Control NC-74: leave `hitsai/millm-backend` in `k8s/base/kustomization.yaml` → red.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
K8S = ROOT / "k8s"
OLD = re.compile(r"millm", re.IGNORECASE)
OLD_IMAGE = re.compile(r"hitsai/millm-")
#: Keys whose string value names a Kubernetes object.
REF_KEYS = {"claimName", "volumeName", "storageClassName", "namePattern", "newName", "project",
            "serviceName", "externalName"}
NAME_ANNOTATIONS = {"nginx.ingress.kubernetes.io/websocket-services"}
#: (file, dotted path, value): allowed old names, each with its reason. Empty since 040 P7, which
#: removed the AppProject's `millm` destination with k8s/transition/ (PD-7).
ALLOWED: set[tuple[str, str, str]] = set()


def _manifests() -> list[tuple[str, dict]]:
    out = []
    for path in sorted(K8S.rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if isinstance(doc, dict) and "kind" in doc:
                out.append((path.relative_to(ROOT).as_posix(), doc))
    return out


def _names(doc: dict) -> list[tuple[str, str]]:
    """`(dotted path, value)` for every object name, namespace, label and reference in `doc`."""
    kind = doc["kind"]
    meta = doc.get("metadata") or {}
    where = f"{kind}/{meta.get('name', '')}"
    out = [(f"{where}.metadata.name", meta.get("name")), (f"{where}.metadata.namespace", meta.get("namespace"))]
    out += [(f"{where}.metadata.labels.{k}", v) for k, v in (meta.get("labels") or {}).items()]
    out += [(f"{where}.metadata.annotations.{k}", v) for k, v in (meta.get("annotations") or {}).items()
            if k in NAME_ANNOTATIONS]
    spec = doc.get("spec") or {}
    if kind == "Kustomization" or kind == "Component":
        out.append((f"{where}.namespace", doc.get("namespace")))
        for image in doc.get("images") or []:
            out += [(f"{where}.images[].name", image.get("name")),
                    (f"{where}.images[].newName", image.get("newName"))]
        for gen in doc.get("configMapGenerator") or []:
            out += [(f"{where}.configMapGenerator[].namespace", gen.get("namespace"))]
        for patch in doc.get("patches") or []:
            out.append((f"{where}.patches[].target.name", (patch.get("target") or {}).get("name")))
            if "patch" in patch:
                inner = yaml.safe_load(patch["patch"])
                if isinstance(inner, dict) and "kind" in inner:
                    out += _names(inner)
    if kind == "AppProject":
        out += [(f"{where}.spec.destinations[].namespace", d.get("namespace"))
                for d in spec.get("destinations") or []]
    if kind == "Application":
        out.append((f"{where}.spec.destination.namespace", (spec.get("destination") or {}).get("namespace")))

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}"
                if key in REF_KEYS and isinstance(value, str):
                    out.append((here, value))
                elif key in ("matchLabels", "selector", "labels") and isinstance(value, dict) and all(
                        isinstance(v, str) for v in value.values()):
                    out.extend((f"{here}.{k}", v) for k, v in value.items())
                elif key in ("secretKeyRef", "configMapKeyRef", "service", "configMap", "secret") \
                        and isinstance(value, dict):
                    out.append((f"{here}.name", value.get("name") or value.get("secretName")))
                elif key == "volumes" and isinstance(value, list):
                    out.extend((f"{here}[].name", v.get("name")) for v in value if isinstance(v, dict))
                walk(value, here)
        elif isinstance(node, list):
            for item in node:
                walk(item, f"{path}[]")

    walk(spec, f"{where}.spec")
    return [(p, v) for p, v in out if isinstance(v, str)]


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)


def test_no_object_namespace_or_image_names_millm():
    docs = _manifests()
    checked: list[tuple[str, str, str]] = []
    for rel, doc in docs:
        checked += [(rel, path, value) for path, value in _names(doc)]
    # Floors: the walk must see the objects it guards, or it would pass on nothing.
    assert len(docs) >= 30, f"only {len(docs)} manifests read: the walk is failing open"
    assert len(checked) >= 140, f"only {len(checked)} names read: the walk is failing open"
    seen = {value for _rel, _path, value in checked}
    assert {"miserver-backend", "miserver", "miserver-otel"} <= seen, sorted(seen)

    bad = [(rel, path, value) for rel, path, value in checked
           if OLD.search(value) and (rel, re.sub(r"\[\d+\]", "[]", path), value) not in ALLOWED]
    images = [(rel, s) for rel, doc in docs for s in _strings(doc) if OLD_IMAGE.search(s)]
    assert not bad, "objects still named, placed or labelled millm:\n  " + "\n  ".join(map(str, bad))
    assert not images, "images still named hitsai/millm-*:\n  " + "\n  ".join(map(str, images))


def test_the_allowed_old_names_are_still_there():
    """An allowance that matches nothing hides a renamed key path; keep the list exact."""
    found = {(rel, path, value) for rel, doc in _manifests() for path, value in _names(doc)}
    assert ALLOWED <= found, ALLOWED - found


def test_the_transition_directory_is_gone():
    """040 P7 (FR-040.40): k8s/transition/ is deleted, so no manifest under k8s/ is skipped."""
    assert not (K8S / "transition").exists()
