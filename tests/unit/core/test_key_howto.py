"""T25 (040 FR-040.28): the instructions a refused save quotes (`KEY_HOWTO`) name the namespace, Secret
and pod label the deployed manifests actually define.

`KEY_HOWTO` is the whole of what an operator gets when a Hugging Face token cannot be saved
(`SETTINGS_KEY_MISSING`): copied as written, it must patch the Secret the backend reads
`SETTINGS_ENCRYPTION_KEY` from and restart the pod that reads it. Every value is derived from
`k8s/base/backend.yaml` (parsed), so a rename on either side is red.

Control NC-83: revert `KEY_HOWTO`'s namespace to `-n millm` → red.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from miserver.core.secret_box import KEY_ENV, KEY_HOWTO

ROOT = Path(__file__).resolve().parents[3]


def _backend() -> dict:
    docs = [d for d in yaml.safe_load_all((ROOT / "k8s/base/backend.yaml").read_text()) if d]
    (deploy,) = [d for d in docs if d["kind"] == "Deployment"]
    return deploy


def test_key_howto_names_the_manifest_namespace_secret_and_label():
    deploy = _backend()
    namespace = deploy["metadata"]["namespace"]
    (container,) = [c for c in deploy["spec"]["template"]["spec"]["containers"] if c["name"] == "backend"]
    (ref,) = [e["valueFrom"]["secretKeyRef"] for e in container["env"] if e["name"] == KEY_ENV]
    secret = ref["name"]
    label = deploy["spec"]["template"]["metadata"]["labels"]["app"]

    namespaces = set(re.findall(r"kubectl -n (\S+)", KEY_HOWTO))
    assert namespaces == {namespace}, (namespaces, namespace)
    assert f"patch secret {secret} " in KEY_HOWTO, (secret, KEY_HOWTO)
    assert f"in the {secret} Secret" in KEY_HOWTO, (secret, KEY_HOWTO)
    assert f"delete pod -l app={label}." in KEY_HOWTO, (label, KEY_HOWTO)
    assert f'\\"{KEY_ENV}\\"' in KEY_HOWTO
    assert "millm" not in KEY_HOWTO.lower(), KEY_HOWTO
