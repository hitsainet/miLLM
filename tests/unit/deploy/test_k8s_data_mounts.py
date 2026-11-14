"""T19 (040 FR-040.22; DR-28). No rendered set of manifests mixes the old and the new data paths, and
every node path a pod mounts must already exist.

* **All or nothing (P0 half, the DR-28 exercise).** The P2 window moves every `/data/millm…` and
  `/mnt/hdd2/pv/millm` path to its `miserver` twin at once (S001 mover Job, decision MOVE); a
  half-moved tree would start one store on an empty directory beside its real data. The check reads
  `hostPath` volumes AND `local` PersistentVolumes (the backup disk `/mnt/hdd2/pv/…` is a `local` PV;
  until P2 the walk read `hostPath` only, so a mixed backup path passed).
* **`type: Directory` (P2 half).** `DirectoryOrCreate` turns a missing or misnamed path into an EMPTY
  directory and a store that starts healthy on nothing (the 2026-08-30 incident in `postgres.yaml`).
  `Directory` refuses to start instead (`FailedMount`; rehearsal C-P2-dir, 2026-11-10). The one
  exception is miForge's checkpoint root, mounted read-only so that miServer still starts where
  miForge is not installed (an empty root lists nothing; PD-11).

Controls NC-75 (`postgres.yaml` back to `DirectoryOrCreate`) and NC-76 (one path mixed) → red.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BRANDED = re.compile(r"^/(data|mnt/hdd2/pv)/(millm|miserver)(-otel)?(/|$)")
#: (file, volume path) allowed to stay `DirectoryOrCreate` (FTDD §6.9, PD-11).
CREATE_ALLOWED = {("k8s/base/backend.yaml", "/data/miforge/checkpoints")}


def _volumes(node, kinds=("hostPath",)):
    """Every `{path, type?}` mapping under one of `kinds` (`hostPath`, `local`)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in kinds and isinstance(value, dict) and "path" in value:
                yield value
            else:
                yield from _volumes(value, kinds)
    elif isinstance(node, list):
        for item in node:
            yield from _volumes(item, kinds)


def _walk(root: Path, kinds) -> list[tuple[str, dict]]:
    out = []
    for path in sorted((root / "k8s").rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            out += [(path.relative_to(root).as_posix(), v) for v in _volumes(doc, kinds)]
    return out


def host_paths(root: Path = ROOT) -> list[tuple[str, str]]:
    """`(file, path)` of every `hostPath` volume under `k8s/`."""
    return [(f, v["path"]) for f, v in _walk(root, ("hostPath",))]


def data_paths(root: Path = ROOT) -> list[tuple[str, str]]:
    """`(file, path)` of every node path: `hostPath` volumes and `local` PersistentVolumes."""
    return [(f, v["path"]) for f, v in _walk(root, ("hostPath", "local"))]


def test_data_paths_are_all_new_or_all_old():
    paths = data_paths()
    assert len(paths) >= 6, f"found only {paths}: the walk is failing open"
    assert any(p.startswith("/mnt/hdd2/pv/") for _f, p in paths), "the local backup volume is not read"
    brands = {m.group(2) for _f, p in paths if (m := BRANDED.match(p))}
    assert brands, "no branded data path at all: the pattern no longer matches the manifests"
    assert len(brands) == 1, (
        "old and new data paths are mixed; the P2 move is all or nothing (DR-28):\n  "
        + "\n  ".join(f"{f}: {p}" for f, p in paths if BRANDED.match(p)))


def test_every_hostpath_is_type_directory_except_miforge_checkpoints():
    volumes = _walk(ROOT, ("hostPath",))
    assert len(volumes) >= 5, f"found only {volumes}: the walk is failing open"
    wrong = [(f, v["path"], v.get("type")) for f, v in volumes
             if v.get("type") != "Directory" and (f, v["path"]) not in CREATE_ALLOWED]
    assert not wrong, ("a node path that does not exist must stop the pod, not become an empty "
                       "directory (type: Directory):\n  " + "\n  ".join(map(str, wrong)))
    exempt = {(f, v["path"]): v.get("type") for f, v in volumes if (f, v["path"]) in CREATE_ALLOWED}
    assert exempt == {k: "DirectoryOrCreate" for k in CREATE_ALLOWED}, exempt
