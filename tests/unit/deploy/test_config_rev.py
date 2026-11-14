"""T34 (040 CF-27): a changed ConfigMap rolls its consumers. Kubernetes does not restart a Deployment
when a ConfigMap it reads changes, so a config-only push would leave the old values running. Every
Deployment that reads an inline `kind: ConfigMap` must bump its pod-template annotation
`rebrand.hitsai/config-rev` in the same commit that changes the ConfigMap's `data`.

Kustomize `configMapGenerator` maps are exempt: their name carries a content hash, so a change
already rolls the consumers. Compared with the parent commit (`git show HEAD~1:<path>`; CI checks out
with `fetch-depth: 2`); the self-test on a fixture pair keeps it from passing vacuously on a tree with
no inline ConfigMap, which is miServer's state on 2026-11-08.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ANNOTATION = "rebrand.hitsai/config-rev"


def _docs(text: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]


def _configmaps(tree: dict[str, str]) -> dict[tuple[str, str], dict]:
    out = {}
    for text in tree.values():
        for doc in _docs(text):
            if doc.get("kind") == "ConfigMap":
                meta = doc.get("metadata") or {}
                out[(meta.get("namespace", ""), meta.get("name", ""))] = doc.get("data") or {}
    return out


def _consumers(tree: dict[str, str]) -> dict[tuple[str, str], tuple[set[str], str | None]]:
    """`{(namespace, deployment): (configmap names it reads, config-rev annotation)}`."""
    out = {}
    for text in tree.values():
        for doc in _docs(text):
            if doc.get("kind") not in ("Deployment", "StatefulSet", "DaemonSet"):
                continue
            template = (doc.get("spec") or {}).get("template") or {}
            pod = template.get("spec") or {}
            names = {(v.get("configMap") or {}).get("name") for v in pod.get("volumes") or []}
            for c in [*(pod.get("containers") or []), *(pod.get("initContainers") or [])]:
                names |= {(e.get("configMapRef") or {}).get("name") for e in c.get("envFrom") or []}
                names |= {((e.get("valueFrom") or {}).get("configMapKeyRef") or {}).get("name")
                          for e in c.get("env") or []}
            rev = ((template.get("metadata") or {}).get("annotations") or {}).get(ANNOTATION)
            meta = doc.get("metadata") or {}
            out[(meta.get("namespace", ""), meta.get("name", ""))] = (names - {None}, rev)
    return out


def findings(before: dict[str, str], after: dict[str, str]) -> list[str]:
    old_maps, new_maps = _configmaps(before), _configmaps(after)
    old_users, new_users = _consumers(before), _consumers(after)
    out = []
    for key, data in new_maps.items():
        if key in old_maps and old_maps[key] == data:
            continue
        for deploy, (reads, rev) in new_users.items():
            if deploy[0] == key[0] and key[1] in reads:
                if rev is None or rev == old_users.get(deploy, (set(), None))[1]:
                    out.append(f"ConfigMap {key[0]}/{key[1]} changed and {deploy[1]} kept "
                               f"{ANNOTATION}={rev!r}")
    return out


def _tree_at(ref: str | None) -> dict[str, str] | None:
    files = subprocess.run(["git", "ls-files", "--", "k8s"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    files = [f for f in files if f.endswith((".yaml", ".yml"))]
    if ref is None:
        return {f: (ROOT / f).read_text() for f in files if (ROOT / f).is_file()}
    out = {}
    for f in files:
        shown = subprocess.run(["git", "show", f"{ref}:{f}"], cwd=ROOT, capture_output=True, text=True)
        if shown.returncode == 0:
            out[f] = shown.stdout
    return out


def test_configmap_change_bumps_consumers():
    parent = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD~1"], cwd=ROOT,
                            capture_output=True, text=True)
    if parent.returncode != 0:
        pytest.skip("no parent commit in this checkout (the mirror's orphan snapshot, or a "
                    "fetch-depth 1 clone): the config-rev comparison did NOT run")
    after = _tree_at(None)
    assert after, "no manifest read: the comparison would pass vacuously"
    problems = findings(_tree_at("HEAD~1"), after)
    assert not problems, "\n".join(problems)


CM = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: owui-env\n  namespace: open-webui\ndata:\n  URL: {url}\n"
DEP = ("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: open-webui\n  namespace: open-webui\n"
       "spec:\n  template:\n    metadata:\n      annotations:\n        rebrand.hitsai/config-rev: \"{rev}\"\n"
       "    spec:\n      containers:\n        - name: app\n          envFrom:\n"
       "            - configMapRef:\n                name: owui-env\n")


def test_self_test_on_fixture_pair():
    before = {"a.yaml": CM.format(url="http://old") + "---\n" + DEP.format(rev="aaa")}
    unchanged_rev = {"a.yaml": CM.format(url="http://new") + "---\n" + DEP.format(rev="aaa")}
    bumped = {"a.yaml": CM.format(url="http://new") + "---\n" + DEP.format(rev="bbb")}
    assert len(findings(before, unchanged_rev)) == 1
    assert findings(before, bumped) == []
    assert findings(before, before) == []
