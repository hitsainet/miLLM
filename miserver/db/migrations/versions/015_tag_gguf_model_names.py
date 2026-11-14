"""Give every GGUF model a `repo:QUANT` name so two quants stop colliding.

Migration 014 widened uniqueness to (repo_id, quantization, gguf_label) so
several quantizations of one repository could coexist — the entire point of the
picker it shipped with. It left `name` as `repo_id.split("/")[-1]`, so the
second download of a repo produced a second row with the SAME name.
`ModelRepository.find_by_name` then raised "Multiple rows were found when one or
none was required" and every OpenAI request naming that model returned 500. The
model was loaded and serving the whole time; it was simply unreachable by name.

Observed on this deployment:

    43 | gemma-4-31b-it-3MPER0RR-abliterated-GGUF | Q4_K_M | ready
    44 | gemma-4-31b-it-3MPER0RR-abliterated-GGUF | IQ4_XS | loaded

`name:tag` is the convention Ollama already taught everyone, it is unique by
construction, and it makes both quantizations separately selectable in an
OpenAI client instead of one shadowing the other in /v1/models.

Data-only. Idempotent: a row already carrying its tag is left alone, so a
re-run — or a deployment where new rows were written by the fixed service
before this ran — is a no-op rather than `foo:Q4_K_M:Q4_K_M`.

The downgrade strips the tags back off. It can reintroduce the collision, which
is what downgrading past the fix MEANS; it is not a reason to leave the tags on
a schema that no longer expects them.

Revision ID: 015
Revises: 014
"""

from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tag EXACTLY the rows the fixed service would have tagged: those whose name
    # is still the bare repo tail. A row carrying a custom_name is left alone,
    # because download_model does not tag one either — renaming "my-judge" to
    # "my-judge:Q4_K_M" here would override what the user typed and break every
    # config that names it.
    #
    # Idempotent by the same condition: once tagged, the name no longer equals
    # the bare tail, so a re-run — or a deployment where the fixed service wrote
    # new rows before this ran — is a no-op rather than `foo:Q4_K_M:Q4_K_M`.
    op.execute(
        """
        UPDATE models
        SET name = name || ':' || gguf_label
        WHERE gguf_label IS NOT NULL
          AND gguf_label <> ''
          AND repo_id IS NOT NULL
          AND name = regexp_replace(repo_id, '^.*/', '')
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE models
        SET name = regexp_replace(repo_id, '^.*/', '')
        WHERE gguf_label IS NOT NULL
          AND gguf_label <> ''
          AND repo_id IS NOT NULL
          AND name = regexp_replace(repo_id, '^.*/', '') || ':' || gguf_label
        """
    )
