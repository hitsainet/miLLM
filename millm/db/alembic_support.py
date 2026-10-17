"""What Alembic compares the database against — defined once.

``millm/db/migrations/env.py`` and ``tests/schema/test_schema_guards.py`` both import
this module, so the metadata and options behind ``alembic check`` are the ones the
schema guards test. env.py previously configured no ``compare_server_default``, so a
server default present in a migration and absent from a model was invisible.
"""

from sqlalchemy import MetaData

# Types and server defaults are part of the schema. Without these, autogenerate and
# `alembic check` report a matching schema over columns whose type or default differs.
COMPARE_OPTIONS = {"compare_type": True, "compare_server_default": True}


def target_metadata() -> MetaData:
    """``Base.metadata`` with every model registered.

    The import is inside the function so a fresh interpreter (which is what
    ``alembic`` is) registers the models no matter what else it has imported.
    """
    import millm.db.models  # noqa: F401  registers every model on Base.metadata
    from millm.db.base import Base

    return Base.metadata
