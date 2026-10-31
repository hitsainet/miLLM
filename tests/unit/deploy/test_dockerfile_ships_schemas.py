"""The image must carry the vendored schemas that production code validates against.

The adapter and checkpoint manifest validators resolve `docs/schemas/*.json` relative to
the package (`parents[2]`), which is `/app/docs/schemas/` in the image. Without the COPY
the file is absent in production while every unit test, which runs from a checkout, passes.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_dockerfile_copies_the_vendored_schemas_next_to_the_package():
    lines = [ln.split() for ln in (REPO / "Dockerfile").read_text().splitlines()]
    copies = {(ln[1], ln[2]) for ln in lines if len(ln) == 3 and ln[0] == "COPY"}
    assert ("millm/", "/app/millm/") in copies
    assert ("docs/schemas/", "/app/docs/schemas/") in copies


def test_the_runtime_schemas_exist_in_the_copied_directory():
    for name in ("adapter-manifest-v1.json", "checkpoint-manifest-v1.json"):
        assert (REPO / "docs" / "schemas" / name).is_file(), name
