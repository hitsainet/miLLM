"""033's LOCAL-row base identity and 032's checkpoint identity are the SAME value (D-09).

`adapter_artifact.local_manifest_sha256` keeps its own guarded reader (one open, `O_NOFOLLOW`,
regular files only, size-capped), but the digest it produces must equal the one 032 records for
the same checkpoint (`checkpoint_store.sha256_file` of `manifest.json`), or an adapter naming a
checkpoint by its manifest hash would refuse against that checkpoint served as a LOCAL row.
Control (W3): hash the parsed-and-reserialised JSON instead of the bytes -> red.
"""

from __future__ import annotations

from millm.services.adapter_artifact import local_manifest_sha256
from millm.services.checkpoint_store import sha256_file


def test_the_local_row_identity_is_032s_manifest_hash(tmp_path):
    # Deliberately not canonical JSON (spacing, key order, trailing newline): only a BYTE hash
    # agrees with 032's.
    (tmp_path / "manifest.json").write_bytes(b'{"name":  "ckpt",\n "kind": "x"}\n')
    assert local_manifest_sha256(str(tmp_path)) == sha256_file(tmp_path / "manifest.json")


def test_no_manifest_is_none(tmp_path):
    assert local_manifest_sha256(str(tmp_path)) is None
    assert local_manifest_sha256(None) is None
