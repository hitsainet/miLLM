"""Feature 35 (ENH-003 §3.2): the runtime-settings registry, validation, apply and startup load.

Route-level behaviour (the live app, secrets, history over HTTP) is in
`tests/unit/api/test_environment_routes.py`; the five values that used to be captured at startup
are exercised through their real readers in `tests/unit/services/test_runtime_captured_keys.py`.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path

import pytest
import structlog
from sqlalchemy import select

from millm.core.config import Settings, settings
from millm.db.models.env_override import EnvOverride, EnvOverrideChange
from millm.services import runtime_settings as rs
from tests.unit.batch_fixtures import batch_db  # noqa: F401

REPO = Path(__file__).resolve().parents[3]

#: One valid new value per writable key, distinct from its default. A key added to RUNTIME_KEYS
#: without a row here fails `test_every_runtime_key_has_a_test_value`.
VALID_VALUES: dict[str, object] = {
    "LOG_LEVEL": "WARNING",
    "MAX_PENDING_REQUESTS": 3,
    "CLUSTER_HUB_CACHE_TTL_S": 60,
    "PROBE_HUB_CACHE_TTL_S": 61,
    "BATCH_RETENTION_INTERVAL_S": 600.0,
    "EMBEDDINGS_MAX_INPUTS": 64,
    "TRANSFORMERS_IDLE_CACHE_RELEASE_S": 12.5,
    "GRACEFUL_UNLOAD_TIMEOUT": 45.0,
    "LEASE_DEFAULT_TTL_SECONDS": 3600,
    "LEASE_MAX_TTL_SECONDS": 10800,
    "BATCH_MAX_ROWS": 1000,
    "BATCH_MAX_FILE_BYTES": 10_485_760,
    "BATCH_FILE_RETENTION_DAYS": 7,
    "BATCH_CHUNK_ROWS": 4,
    "PROBE_MAX_ARMED": 4,
    "PROBE_SLOWDOWN_BUDGET_PCT": 7.5,
    "PROBE_VERDICT_HEADER_MAX_BYTES": 4096,
    "PROBE_MAX_IMPORTED": 500,
    "PROBE_MAX_PER_BEHAVIOUR": 16,
    "PROBE_LIST_PAGE_SIZE": 50,
    "RULE_SIGNALS_ENABLED": False,
    "RULE_TIME_BUDGET_MS": 40.0,
    "RULE_SETS_MAX_IMPORTED": 200,
    "APPROVAL_TTL_HOURS": 48,
    "RULE_SET_MAX_SIGNALS": 32,
    "RULE_SET_MAX_BYTES": 65536,
    "RULE_CODE_CALL_WALL_MS": 500,
    "RULE_CODE_CALL_MEMORY_MIB": 128,
    "RULE_CODE_CALL_OUTPUT_BYTES": 8192,
    "RULE_CODE_CALLS_PER_REQUEST": 4,
    "RULE_CODE_MODULES_MAX_BYTES": 65536,
    "PROBE_MAX_AGE_DAYS": 14,
    "PROBE_MAX_EVENTS_PER_PROBE": 100,
    "SENSING_MAX_AGE_DAYS": 3,
    "CIRCUIT_SENSING_MAX_AGE_DAYS": 4,
    "SENSING_EPSILON": 0.2,
    "CIRCUIT_SENSING_EPSILON": 0.25,
    "CIRCUIT_SENSING_MAX_TOKEN_LAG": 4,
    "CLUSTER_INTENSITY_MIN": 0.25,
    "CLUSTER_INTENSITY_MAX": 2.0,
    "CIRCUIT_INTENSITY_MIN": 0.1,
    "CIRCUIT_INTENSITY_MAX": 3.0,
    "MULTISAE_VRAM_ENVELOPE_MB": 2048,
    "SCORING_FLOAT32_DEVICE": "cpu",
    "SCORING_FLOAT32_HEADROOM_MB": 512,
    "PROBE_SCORE_MAX_INPUTS": 128,
    "PROBE_MAX_OVERHEAD_MS_PER_PASS": 0.3,
    "PROBE_MAX_PREFILL_MS_PER_1K": 0.6,
    "PROBE_BUDGET_HEADROOM": 0.8,
    "PROBE_COST_SAMPLES": 10,
    "PROBE_CAPACITY_DECODE_STEPS": 16,
    "PROBE_TRANSCRIPTS": "on",
    "PROBE_TRANSCRIPT_MAX_AGE_DAYS": 3,
    "PROBE_TRANSCRIPT_MAX_TURNS": 500,
    "PROBE_SESSION_IDLE_MINUTES": 45,
    "PROBE_RETENTION_INTERVAL_S": 120,
    "PROBE_SESSION_MAX_AGE_DAYS": 14,
    # ENH-006 D75 (2026-10-28): every telemetry setting but the environment-only five.
    "TELEMETRY_TRACES_ENABLED": True,
    "TELEMETRY_METRICS_ENABLED": True,
    "TELEMETRY_LOGS_ENABLED": True,
    "TELEMETRY_ENDPOINT": "http://127.0.0.1:1",
    "TELEMETRY_PROTOCOL": "http/protobuf",
    "TELEMETRY_ALLOW_PLAINTEXT": True,
    "TELEMETRY_SERVICE_NAME": "millm-test",
    "TELEMETRY_SERVICE_INSTANCE_ID": "pod-7",
    "TELEMETRY_DEPLOYMENT_ENVIRONMENT": "staging",
    "TELEMETRY_QUEUE_SIZE": 1024,
    "TELEMETRY_BATCH_SIZE": 64,
    "TELEMETRY_EXPORT_TIMEOUT_S": 2.5,
    "TELEMETRY_RETRY_HORIZON_S": 30.0,
    "TELEMETRY_DRAIN_DEADLINE_S": 1.5,
    "TELEMETRY_METRIC_INTERVAL_S": 30.0,
    "TELEMETRY_TRACES_SAMPLER_RATIO": 0.25,
    "TELEMETRY_SERIES_CAP": 500,
    "TELEMETRY_STATELESS_EXPORT_PERMITTED": True,
    "TELEMETRY_RESULT_LOG_ORIGINS": "live",
    "TELEMETRY_HEALTH_DEGRADED_WINDOW_S": 30.0,
    "TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S": 600.0,
}

#: When each key takes effect, as the page states it. Pinned: a change here is a change to what
#: the operator is told, and must be made on purpose (spec §4 "effect timing is pinned").
EFFECTS: dict[str, str] = {
    "LOG_LEVEL": "now",
    "MAX_PENDING_REQUESTS": "next queue admission",
    "CLUSTER_HUB_CACHE_TTL_S": "next Hub lookup",
    "PROBE_HUB_CACHE_TTL_S": "next Hub lookup",
    "BATCH_RETENTION_INTERVAL_S": "within 30 seconds (next clean-up check)",
    "EMBEDDINGS_MAX_INPUTS": "next request",
    "TRANSFORMERS_IDLE_CACHE_RELEASE_S": "when the next request ends",
    "GRACEFUL_UNLOAD_TIMEOUT": "next unload",
    "LEASE_DEFAULT_TTL_SECONDS": "next lease grant",
    "LEASE_MAX_TTL_SECONDS": "next lease grant",
    "BATCH_MAX_ROWS": "next upload",
    "BATCH_MAX_FILE_BYTES": "next upload",
    "BATCH_FILE_RETENTION_DAYS": "next upload",
    "BATCH_CHUNK_ROWS": "next chunk",
    "PROBE_MAX_ARMED": "next arm",
    # ENH-006 D83 (replaces 039's PROBE_MAX_ARMED_BEHAVIOURS)
    "PROBE_SLOWDOWN_BUDGET_PCT": "next arm",
    "PROBE_VERDICT_HEADER_MAX_BYTES": "next request",
    "PROBE_MAX_IMPORTED": "next import",
    "PROBE_MAX_PER_BEHAVIOUR": "next arm",
    "PROBE_LIST_PAGE_SIZE": "next request",
    "RULE_SIGNALS_ENABLED": "next request",
    "RULE_TIME_BUDGET_MS": "next request",
    "RULE_SETS_MAX_IMPORTED": "next import",
    "APPROVAL_TTL_HOURS": "next agent request",
    "RULE_SET_MAX_SIGNALS": "next import",
    "RULE_SET_MAX_BYTES": "next import",
    "RULE_CODE_CALL_WALL_MS": "next code-rule job",
    "RULE_CODE_CALL_MEMORY_MIB": "next code-rule job",
    "RULE_CODE_CALL_OUTPUT_BYTES": "next code-rule job",
    "RULE_CODE_CALLS_PER_REQUEST": "next code-rule job",
    "RULE_CODE_MODULES_MAX_BYTES": "next import",
    "PROBE_MAX_AGE_DAYS": "next recorded event",
    "PROBE_MAX_EVENTS_PER_PROBE": "next recorded event",
    "SENSING_MAX_AGE_DAYS": "next recorded event",
    "CIRCUIT_SENSING_MAX_AGE_DAYS": "next recorded event",
    "SENSING_EPSILON": "next arm",
    "CIRCUIT_SENSING_EPSILON": "next arm",
    "CIRCUIT_SENSING_MAX_TOKEN_LAG": "next arm",
    "CLUSTER_INTENSITY_MIN": "next request",
    "CLUSTER_INTENSITY_MAX": "next request",
    "CIRCUIT_INTENSITY_MIN": "next request",
    "CIRCUIT_INTENSITY_MAX": "next request",
    "MULTISAE_VRAM_ENVELOPE_MB": "next attach or status read",
    "SCORING_FLOAT32_DEVICE": "next float32 scoring request",
    "SCORING_FLOAT32_HEADROOM_MB": "next float32 scoring request",
    "PROBE_SCORE_MAX_INPUTS": "next request",
    # Feature 37 (many probes, measured).
    "PROBE_MAX_OVERHEAD_MS_PER_PASS": "next arm",
    "PROBE_MAX_PREFILL_MS_PER_1K": "next arm",
    "PROBE_BUDGET_HEADROOM": "next arm",
    "PROBE_COST_SAMPLES": "next arm",
    "PROBE_CAPACITY_DECODE_STEPS": "next capacity measurement",
    # Feature 36 (probe sessions).
    "PROBE_TRANSCRIPTS": "next request",
    "PROBE_TRANSCRIPT_MAX_AGE_DAYS": "next retention pass",
    "PROBE_TRANSCRIPT_MAX_TURNS": "next retention pass",
    "PROBE_SESSION_IDLE_MINUTES": "next request",
    "PROBE_RETENTION_INTERVAL_S": "next retention pass",
    "PROBE_SESSION_MAX_AGE_DAYS": "next retention pass",
    # ENH-006 D75: every telemetry key rebuilds the export pipeline.
    **{k: "next export; drains and restarts the export pipeline"
       for k in VALID_VALUES if k.startswith("TELEMETRY_")},
}


#: 038 S2 (FTDD §5.8): the telemetry settings by name.
TELEMETRY_SETTING_NAMES = (
    "TELEMETRY_TRACES_ENABLED", "TELEMETRY_METRICS_ENABLED", "TELEMETRY_LOGS_ENABLED",
    "TELEMETRY_ENDPOINT", "TELEMETRY_PROTOCOL", "TELEMETRY_HEADERS", "TELEMETRY_TLS_CA_FILE",
    "TELEMETRY_TLS_CLIENT_CERT_FILE", "TELEMETRY_TLS_CLIENT_KEY_FILE", "TELEMETRY_ALLOW_PLAINTEXT",
    "TELEMETRY_SERVICE_NAME", "TELEMETRY_SERVICE_INSTANCE_ID", "TELEMETRY_DEPLOYMENT_ENVIRONMENT",
    "TELEMETRY_QUEUE_SIZE", "TELEMETRY_BATCH_SIZE", "TELEMETRY_EXPORT_TIMEOUT_S",
    "TELEMETRY_RETRY_HORIZON_S", "TELEMETRY_DRAIN_DEADLINE_S", "TELEMETRY_METRIC_INTERVAL_S",
    "TELEMETRY_TRACES_SAMPLER_RATIO", "TELEMETRY_SERIES_CAP", "TELEMETRY_STATELESS_EXPORT_PERMITTED",
    "TELEMETRY_RESULT_LOG_ORIGINS", "TELEMETRY_HEALTH_DEGRADED_WINDOW_S",
    "TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S", "OTEL_SDK_DISABLED",
)


# =============================================================================
# Classification
# =============================================================================


class TestClassification:
    def test_every_field_is_in_exactly_one_group(self):
        fields = set(Settings.model_fields)
        runtime, load, restart = rs.RUNTIME_KEYS, set(rs.MODEL_LOAD_KEYS), set(rs.RESTART_KEYS)
        assert runtime <= fields and load <= fields and restart <= fields, "a group names no field"
        assert not (runtime & load) and not (runtime & restart) and not (load & restart)
        assert not (rs.NEVER_WRITABLE & runtime), "a never-writable key is writable"
        assert rs.SECRET_KEYS <= rs.NEVER_WRITABLE
        groups = {k: rs.group_of(k) for k in fields}
        assert set(groups.values()) == {"runtime", "model_load", "restart", "environment"}

    def test_never_writable_is_pinned_by_name(self):
        """The spec's list, written out: dropping a name from NEVER_WRITABLE (mutation M24 survived
        before this test) would also drop it from the route test that PUTs each one."""
        assert rs.NEVER_WRITABLE == {
            "DATABASE_URL", "HF_TOKEN", "SETTINGS_ENCRYPTION_KEY", "WEIGHTS_TOKENS",
            "MODEL_CACHE_DIR", "SAE_CACHE_DIR", "BATCH_FILES_DIR", "WEIGHTS_CHECKPOINT_DIR",
            "WEIGHTS_CHECKPOINT_READ_ROOTS", "HOST", "PORT", "CORS_ORIGINS", "DEBUG",
            "MAX_CONCURRENT_REQUESTS", "SENSING_FORCE_SERIAL", "CIRCUIT_SENSING_FORCE_SERIAL",
            "PROBE_FORCE_SERIAL", "PROBE_PARITY_TOLERANCE", "PROBE_PARITY_SCORE_TOLERANCE",
            "PROBE_PARITY_MATCHED_DTYPE_FLOOR", "PROBE_PARITY_MATCHED_RELATIVE_FLOOR",
            "PROBE_PARITY_MIN_ROBUST_FRACTION", "BATCH_PACK_DEFAULT", "TORCH_COMPILE_MODE",
            "CLUSTER_HUB_TAG", "PROBE_HUB_TAG", "WEIGHTS_COMMIT_WATCHDOG_MS",
            "LEASE_HOLDER_MAX_CHARS", "LEASE_REASON_MAX_CHARS",
            # ENH-006 D75 (reversing 038's OT-D5 for the rest): the telemetry settings that stay
            # environment-only — the secrets, the two certificate paths, the SDK kill switch.
            "TELEMETRY_HEADERS", "TELEMETRY_TLS_CLIENT_KEY_FILE", "TELEMETRY_TLS_CA_FILE",
            "TELEMETRY_TLS_CLIENT_CERT_FILE", "OTEL_SDK_DISABLED",
        }
        assert rs.SECRET_KEYS == {"DATABASE_URL", "HF_TOKEN", "SETTINGS_ENCRYPTION_KEY",
                                  "WEIGHTS_TOKENS", "TELEMETRY_HEADERS",
                                  "TELEMETRY_TLS_CLIENT_KEY_FILE"}

    def test_the_counts_are_the_decided_ones(self):
        """D2 (30 keys) + Feature 36's six + Feature 37's four + the capacity measurement's one +
        039 S1's eight + 039 S2's five + 039 S3's five + ENH-008's approval expiry + ENH-006
        D75's 21 telemetry keys, D9 (10 next-model-load keys read-only), the 13 restart keys + 039
        S3's runner socket."""
        assert (len(rs.RUNTIME_KEYS), len(rs.MODEL_LOAD_KEYS), len(rs.RESTART_KEYS)) == (78, 10, 14)

    def test_every_runtime_key_has_a_test_value(self):
        assert set(VALID_VALUES) == set(rs.RUNTIME_KEYS)

    @pytest.mark.parametrize("key", sorted(EFFECTS))
    def test_effect_is_pinned(self, key):
        assert set(EFFECTS) == set(rs.RUNTIME_KEYS)
        assert rs.RUNTIME_SPECS[key].effect == EFFECTS[key]

    def test_read_only_groups_state_their_effect(self):
        assert rs.effect_of("TORCH_COMPILE") == "next model load"
        assert "restart millm-backend" in rs.effect_of("KV_CACHE_MODE")
        assert "ends in-flight" in rs.effect_of("AUTO_LOAD_MODEL")

    def test_no_writable_key_can_pin_a_gpu_index(self):
        """Suite rule (design for 1 + n GPUs): the page names no card by index."""
        assert "cuda:0" not in rs.RUNTIME_SPECS["SCORING_FLOAT32_DEVICE"].choices
        with pytest.raises(rs.SettingInvalidError):
            rs.validate_change("SCORING_FLOAT32_DEVICE", "cuda:1")


def scan_reads(source: str, rel: str, keys: set[str]) -> list[tuple[str, str, int, str]]:
    """(key, file, line, scope) for every attribute named a key, and every string constant equal
    to one (`getattr(settings, self.TTL_SETTING)` names its key that way). Scope is `function`,
    `init` (an `__init__` body), `default` (a default argument), `class` or `module`."""
    out: list[tuple[str, str, int, str]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in child.args.defaults + child.args.kw_defaults:
                    if default is not None:
                        record(default, "default")
                        visit(default, "default")
                body_scope = "init" if child.name == "__init__" else "function"
                for stmt in child.body:
                    record(stmt, body_scope)
                    visit(stmt, body_scope)
                continue
            child_scope = "class" if isinstance(child, ast.ClassDef) else scope
            record(child, child_scope)
            visit(child, child_scope)

    def record(node: ast.AST, scope: str) -> None:
        if isinstance(node, ast.Attribute) and node.attr in keys:
            out.append((node.attr, rel, node.lineno, scope))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in keys:
            out.append((node.value, rel, node.lineno, "name"))

    visit(ast.parse(source), "module")
    return out


def _all_reads() -> dict[str, list[tuple[str, int, str]]]:
    found: dict[str, list[tuple[str, int, str]]] = {k: [] for k in rs.RUNTIME_KEYS}
    for path in sorted((REPO / "millm").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in ("millm/core/config.py", "millm/services/runtime_settings.py"):
            continue
        if "/migrations/" in rel:
            continue
        for key, f, line, scope in scan_reads(path.read_text(), rel, set(rs.RUNTIME_KEYS)):
            found[key].append((f, line, scope))
    return found


class TestEveryRuntimeKeyIsReadAtUse:
    """Each writable key has a reader in `millm/`, and none reads it once and keeps it: no read in
    an `__init__`, a default argument, or at module or class level. (That is the shape of the two
    Hub TTLs, captured in `__init__`. A value captured by being PASSED to a constructor inside a
    function — the queue limit and the retention interval — cannot be seen this way; those two
    are exercised through their real readers in `test_runtime_captured_keys.py`.)"""

    def test_the_scanner_classifies_scopes(self):
        """Guard the guard: each capture shape is classified, so the check below can see it."""
        source = (
            "X = settings.LOG_LEVEL\n"
            "class A:\n"
            "    Y = settings.LOG_LEVEL\n"
            "    def __init__(self):\n"
            "        self.t = settings.LOG_LEVEL\n"
            "    def f(self, d=settings.LOG_LEVEL):\n"
            "        return settings.LOG_LEVEL\n"
        )
        scopes = [s for _, _, _, s in scan_reads(source, "x.py", {"LOG_LEVEL"})]
        assert scopes == ["module", "class", "init", "default", "function"]

    def test_the_scan_sees_reads(self):
        assert any(scope == "function" for *_, scope in _all_reads()["MAX_PENDING_REQUESTS"])

    @pytest.mark.parametrize("key", sorted(VALID_VALUES))
    def test_read_in_a_function_and_never_captured(self, key):
        reads = _all_reads()[key]
        assert reads, f"{key} is writable but nothing in millm/ reads it"
        captured = [r for r in reads if r[2] in ("init", "module", "class", "default")]
        assert not captured, f"{key} is captured once (an override would be ignored): {captured}"


# =============================================================================
# Validation
# =============================================================================


class TestValidation:
    @pytest.mark.parametrize("key", sorted(VALID_VALUES))
    def test_each_valid_value_is_accepted(self, key):
        assert rs.validate_change(key, VALID_VALUES[key]) == VALID_VALUES[key]

    @pytest.mark.parametrize("key,value,needle", [
        ("MAX_PENDING_REQUESTS", 0, "at least 1"),
        ("MAX_PENDING_REQUESTS", 1001, "at most 1000"),
        ("MAX_PENDING_REQUESTS", True, "must be a number"),
        ("MAX_PENDING_REQUESTS", "7", "must be a number"),
        ("MAX_PENDING_REQUESTS", 2.5, "whole number"),
        ("RULE_SIGNALS_ENABLED", "false", "true or false"),
        ("RULE_SIGNALS_ENABLED", 0, "true or false"),
        ("MAX_PENDING_REQUESTS", None, "must be a number"),
        ("EMBEDDINGS_MAX_INPUTS", 5000, "less than or equal to 2048"),
        ("LOG_LEVEL", "LOUD", "must be one of"),
        ("LOG_LEVEL", 3, "must be one of"),
        ("PROBE_SCORE_MAX_INPUTS", 63, "at least 64"),
        ("LEASE_DEFAULT_TTL_SECONDS", 9000, "LEASE_MAX_TTL_SECONDS"),
        ("LEASE_MAX_TTL_SECONDS", 600, "BATCH_LEASE_TTL_S"),
        ("CLUSTER_INTENSITY_MIN", 1.5, "must be below CLUSTER_INTENSITY_MAX"),
        ("CIRCUIT_INTENSITY_MAX", 0.0, "must be below CIRCUIT_INTENSITY_MAX"),
        ("SCORING_FLOAT32_DEVICE", "gpu", "must be one of"),
        ("BATCH_RETENTION_INTERVAL_S", 5, "at least 60"),
    ])
    def test_refusals_carry_the_rule(self, key, value, needle):
        with pytest.raises(rs.SettingInvalidError) as err:
            rs.validate_change(key, value)
        assert needle in err.value.message

    def test_normalises_choices(self):
        assert rs.validate_change("LOG_LEVEL", " debug ") == "DEBUG"
        assert rs.validate_change("SCORING_FLOAT32_DEVICE", "CPU") == "cpu"

    def test_a_whole_float_is_an_int(self):
        assert rs.validate_change("MAX_PENDING_REQUESTS", 12.0) == 12

    def test_validation_never_reads_the_environment(self, monkeypatch):
        """`Settings.model_validate` must not pick up an env var over the candidate's values."""
        monkeypatch.setenv("MAX_PENDING_REQUESTS", "999")
        assert rs.validate_change("MAX_PENDING_REQUESTS", 4) == 4

    def test_not_runtime_and_unknown(self):
        with pytest.raises(rs.SettingNotRuntimeError):
            rs.validate_change("DEBUG", True)
        with pytest.raises(rs.SettingNotFoundError):
            rs.validate_change("NOT_A_SETTING", 1)

    def test_refusal_never_echoes_a_secret(self, monkeypatch):
        """A whole-model validation error must not quote another field's input (a secret)."""
        monkeypatch.setattr(settings, "WEIGHTS_TOKENS", "SENTINEL-SECRET-123")
        monkeypatch.setattr(settings, "WEIGHTS_HEARTBEAT_INTERVAL_HINT_S", 99.0)  # now invalid
        with pytest.raises(rs.SettingInvalidError) as err:
            rs.validate_change("MAX_PENDING_REQUESTS", 4)
        assert "SENTINEL" not in err.value.message and "SENTINEL" not in str(err.value.details)


# =============================================================================
# Writes, apply, history
# =============================================================================


class TestWrites:
    async def test_set_stores_row_and_history_then_applies(self, batch_db):
        async with batch_db() as session:
            row = await rs.set_override(session, "MAX_PENDING_REQUESTS", 4, reason="busy day")
        assert settings.MAX_PENDING_REQUESTS == 4
        assert (row["value"], row["env_value"], row["source"]) == (4, 10, "override")
        assert (row["updated_by"], row["reason"]) == ("operator", "busy day")
        async with batch_db() as session:
            stored = (await session.execute(select(EnvOverride))).scalars().all()
            changes = (await session.execute(select(EnvOverrideChange))).scalars().all()
        assert [(r.key, r.value, r.updated_by) for r in stored] == [("MAX_PENDING_REQUESTS", 4, "operator")]
        assert [(c.action, c.previous_value, c.new_value, c.reason) for c in changes] == [
            ("set", 10, 4, "busy day")
        ]

    async def test_reset_restores_the_environment_and_records_it(self, batch_db):
        async with batch_db() as session:
            await rs.set_override(session, "PROBE_MAX_ARMED", 3)
            row = await rs.reset_override(session, "PROBE_MAX_ARMED", reason="done")
        assert settings.PROBE_MAX_ARMED == 64 and row["source"] == "env" and row["value"] == 64
        async with batch_db() as session:
            assert (await session.execute(select(EnvOverride))).scalars().all() == []
            changes = (await session.execute(
                select(EnvOverrideChange).order_by(EnvOverrideChange.id))).scalars().all()
        assert [(c.action, c.previous_value, c.new_value) for c in changes] == [
            ("set", 64, 3), ("reset", 3, 64)
        ]
        assert changes[1].reason == "done"

    async def test_reset_without_an_override_is_404(self, batch_db):
        async with batch_db() as session:
            with pytest.raises(rs.SettingNotOverriddenError):
                await rs.reset_override(session, "PROBE_MAX_ARMED")

    async def test_reset_refused_when_the_env_value_no_longer_fits(self, batch_db):
        """Raise the maximum, raise the default above the env maximum, then reset the maximum:
        the env maximum (7200) is below the overridden default (9000), so the reset is refused."""
        async with batch_db() as session:
            await rs.set_override(session, "LEASE_MAX_TTL_SECONDS", 10800)
            await rs.set_override(session, "LEASE_DEFAULT_TTL_SECONDS", 9000)
            with pytest.raises(rs.SettingInvalidError) as err:
                await rs.reset_override(session, "LEASE_MAX_TTL_SECONDS")
        assert "Change or reset those first" in err.value.message
        assert settings.LEASE_MAX_TTL_SECONDS == 10800

    async def test_reset_ignores_the_registry_bounds_for_the_env_value(self, batch_db, monkeypatch):
        """The environment is authoritative: an env value outside the page's bounds restores."""
        monkeypatch.setattr(settings, "MAX_PENDING_REQUESTS", 5000)
        async with batch_db() as session:
            await rs.set_override(session, "MAX_PENDING_REQUESTS", 20)
            row = await rs.reset_override(session, "MAX_PENDING_REQUESTS")
        assert row["value"] == 5000 and settings.MAX_PENDING_REQUESTS == 5000

    async def test_nothing_applies_when_the_commit_fails(self, batch_db):
        class Boom(Exception):
            pass

        async with batch_db() as session:
            async def failing_commit():
                raise Boom()

            session.commit = failing_commit  # type: ignore[method-assign]
            with pytest.raises(Boom):
                await rs.set_override(session, "MAX_PENDING_REQUESTS", 4)
        assert settings.MAX_PENDING_REQUESTS == 10 and rs.overrides() == {}

    async def test_writes_are_serialised(self, batch_db, monkeypatch):
        """Two concurrent writes of an ordered pair: the second validates against the first's
        applied value, so the pair can never end inverted (min 1.4 with max 1.0)."""
        real_commit_waits = []

        async def run(key, value):
            async with batch_db() as session:
                original = session.commit

                async def slow_commit():
                    real_commit_waits.append(key)
                    await asyncio.sleep(0.05)
                    await original()

                session.commit = slow_commit  # type: ignore[method-assign]
                try:
                    await rs.set_override(session, key, value)
                    return "ok"
                except rs.SettingInvalidError:
                    return "refused"

        results = await asyncio.gather(
            run("CLUSTER_INTENSITY_MAX", 1.0), run("CLUSTER_INTENSITY_MIN", 1.4)
        )
        assert results == ["ok", "refused"]
        assert settings.CLUSTER_INTENSITY_MIN < settings.CLUSTER_INTENSITY_MAX

    async def test_reason_is_bounded_and_blank_is_none(self, batch_db):
        async with batch_db() as session:
            row = await rs.set_override(session, "PROBE_MAX_ARMED", 3, reason="   ")
            assert row["reason"] is None
            with pytest.raises(rs.SettingInvalidError):
                await rs.set_override(session, "PROBE_MAX_ARMED", 3, reason="x" * 501)

    async def test_history_is_newest_first_and_limited(self, batch_db):
        async with batch_db() as session:
            for n in (2, 3, 4):
                await rs.set_override(session, "PROBE_MAX_ARMED", n)
            items = await rs.history(session, limit=2)
        assert [i["new_value"] for i in items] == [4, 3]
        assert all(i["actor"] == "operator" and i["approval_id"] is None for i in items)


class TestLogLevelHook:
    def test_apply_sets_the_root_level_now(self):
        rs._apply({"LOG_LEVEL": "ERROR"})
        assert logging.getLogger("millm.services.anything").getEffectiveLevel() == logging.ERROR
        rs._apply({"LOG_LEVEL": "DEBUG"})
        assert logging.getLogger("millm.services.anything").isEnabledFor(logging.DEBUG)


    def test_setup_logging_sets_the_level_even_when_root_has_a_handler(self, monkeypatch):
        """`logging.basicConfig` does nothing (level included) once the root logger has a
        handler; `setup_logging` must still put the in-force LOG_LEVEL on the root."""
        from millm.core.logging import setup_logging

        root = logging.getLogger()
        handler = logging.NullHandler()
        root.addHandler(handler)
        try:
            root.setLevel(logging.DEBUG)
            monkeypatch.setattr(settings, "LOG_LEVEL", "ERROR")
            setup_logging()
            assert root.level == logging.ERROR
        finally:
            root.removeHandler(handler)


# =============================================================================
# Startup load
# =============================================================================


async def _store(factory, **values):
    async with factory() as session:
        for key, value in values.items():
            session.add(EnvOverride(key=key, value=value, updated_by="operator"))
        await session.commit()


class TestStartupLoad:
    async def test_applies_valid_rows(self, batch_db):
        await _store(batch_db, MAX_PENDING_REQUESTS=6, LOG_LEVEL="ERROR")
        result = await rs.load_runtime_overrides_on_startup(batch_db)
        assert sorted(result["applied"]) == ["LOG_LEVEL", "MAX_PENDING_REQUESTS"]
        assert settings.MAX_PENDING_REQUESTS == 6 and settings.LOG_LEVEL == "ERROR"
        assert logging.getLogger().level == logging.ERROR
        assert rs.row_for("MAX_PENDING_REQUESTS")["source"] == "override"
        assert rs.env_value("MAX_PENDING_REQUESTS") == 10

    async def test_ignores_and_logs_invalid_and_non_runtime_rows(self, batch_db, monkeypatch):
        await _store(batch_db, DEBUG=True, MAX_PENDING_REQUESTS=0, PROBE_MAX_ARMED=5,
                     REMOVED_SETTING=1)
        # ⚠ A FRESH LOGGER PROXY (pre-existing order dependence, found 2026-10-21 when new files
        # moved this test into the same CI shard as `test_hf_token.py`). Any test that runs the
        # real `main.lifespan` calls `setup_logging()`, whose `cache_logger_on_first_use` freezes
        # this module's proxy on first use — and `capture_logs` cannot see a cached logger, so
        # this read no warnings at all. Reproduced on `main` 68a73375 with the two files alone.
        monkeypatch.setattr(rs, "logger", structlog.get_logger())
        with structlog.testing.capture_logs() as logs:
            result = await rs.load_runtime_overrides_on_startup(batch_db)
        assert result["applied"] == ["PROBE_MAX_ARMED"]
        assert sorted(i["key"] for i in result["ignored"]) == [
            "DEBUG", "MAX_PENDING_REQUESTS", "REMOVED_SETTING"
        ]
        assert settings.DEBUG is False and settings.MAX_PENDING_REQUESTS == 10
        warned = sorted(e["key"] for e in logs if e["event"] == "runtime_override_ignored")
        assert warned == ["DEBUG", "MAX_PENDING_REQUESTS", "REMOVED_SETTING"]

    async def test_dependent_rows_load_whatever_their_order(self, batch_db):
        """The default (9000) is valid only with the raised maximum (10800). Read in key order the
        default comes first and fails against the env maximum (7200); a second pass accepts it."""
        await _store(batch_db, LEASE_DEFAULT_TTL_SECONDS=9000, LEASE_MAX_TTL_SECONDS=10800)
        result = await rs.load_runtime_overrides_on_startup(batch_db)
        assert result["applied"] == ["LEASE_MAX_TTL_SECONDS", "LEASE_DEFAULT_TTL_SECONDS"]
        assert result["ignored"] == []
        assert (settings.LEASE_DEFAULT_TTL_SECONDS, settings.LEASE_MAX_TTL_SECONDS) == (9000, 10800)

    async def test_a_dependent_row_without_its_partner_is_ignored(self, batch_db):
        await _store(batch_db, LEASE_DEFAULT_TTL_SECONDS=9000)
        result = await rs.load_runtime_overrides_on_startup(batch_db)
        assert result["applied"] == []
        assert [i["key"] for i in result["ignored"]] == ["LEASE_DEFAULT_TTL_SECONDS"]
        assert "LEASE_MAX_TTL_SECONDS" in result["ignored"][0]["reason"]
        assert settings.LEASE_DEFAULT_TTL_SECONDS == 7200

    async def test_an_unmigrated_database_never_raises(self):
        def broken():
            raise RuntimeError("no such table: env_overrides")

        result = await rs.load_runtime_overrides_on_startup(broken)
        assert result["applied"] == [] and result["error"] == "RuntimeError"
