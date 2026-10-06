"""Feature 29 tasks 2.8, 3.7, 3.8: the lease rules through the REAL ModelService.

Every refusal asserts its payload (holder, reason, expires_at) AND that nothing started: the
background load worker's call count stays 0 and no row status moves (FR-29.2.6).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from structlog.testing import capture_logs

from millm.core.errors import (
    InvalidLeaseRequestError,
    ModelBusyError,
    ModelLeasedError,
    ModelLockedError,
    ModelNotFoundError,
    ModelNotResidentError,
)
from millm.db.models.model import ModelStatus
from tests.unit.lease_fixtures import (
    build_service,
    clear_resident,
    drain_executor,
    install_registry,
    no_probe_rows,
    set_resident,
)


@pytest.fixture(autouse=True)
def _clean_loader():
    clear_resident()
    yield
    clear_resident()


@pytest.fixture
def env():
    registry, clock = install_registry()
    service, repo = build_service()
    return service, repo, registry, clock


async def _lease(service, holder="midataworks", ttl=120):
    return await service.acquire_lease(1, holder, "label run 7", ttl)


def _assert_refusal_payload(exc, operation, target):
    assert exc.code == "MODEL_LEASED"
    d = exc.details
    assert d["holder"] == "midataworks"
    assert d["reason"] == "label run 7"
    assert d["expires_at"] == "2026-10-06T12:02:00+00:00"
    assert d["leased_model_id"] == 1 and d["leased_model_name"] == "m1"
    assert d["operation"] == operation and d["target_model_id"] == target


# --- 2.8: the grant ----------------------------------------------------------------------------


class TestGrant:
    async def test_grant_on_the_resident_model(self, env):
        service, _, registry, _ = env
        grant = await _lease(service)
        assert grant.record.model_id == 1 and grant.record.model_name == "m1"
        assert registry.current(1).holder == "midataworks"

    async def test_default_ttl_is_7200(self, env):
        service, _, _, _ = env
        grant = await service.acquire_lease(1, "midataworks", "r")
        assert grant.record.ttl_seconds == 7200

    async def test_non_resident_model_is_refused_naming_the_resident(self, env):
        service, _, _, _ = env
        with pytest.raises(ModelNotResidentError) as exc:
            await service.acquire_lease(2, "midataworks", "r", 60)
        assert exc.value.details["resident_model_id"] == 1
        assert exc.value.details["resident_model_name"] == "m1"

    async def test_nothing_resident_says_none(self, env):
        service, _, _, _ = env
        clear_resident()
        with pytest.raises(ModelNotResidentError) as exc:
            await service.acquire_lease(1, "midataworks", "r", 60)
        assert "none" in exc.value.message

    async def test_resident_but_row_not_loaded_is_refused(self, env):
        service, repo, _, _ = env
        repo.rows[1].status = ModelStatus.READY
        with pytest.raises(ModelNotResidentError):
            await service.acquire_lease(1, "midataworks", "r", 60)

    async def test_grant_during_a_load_is_refused(self, env):
        """Control M18: the grant must read the load slot."""
        service, _, registry, _ = env
        service._loading_model_id = 2
        with pytest.raises(ModelBusyError) as exc:
            await service.acquire_lease(1, "midataworks", "r", 60)
        assert exc.value.details["loading_model_id"] == 2
        assert registry.current(1) is None

    async def test_grant_during_an_unload_is_refused_as_an_unload(self, env):
        service, _, registry, _ = env
        service.loader.begin_unload()
        with pytest.raises(ModelBusyError) as exc:
            await service.acquire_lease(1, "midataworks", "r", 60)
        assert exc.value.details["unloading"] is True
        assert registry.current(1) is None

    async def test_grant_over_a_live_lease_is_refused(self, env):
        service, _, _, _ = env
        await _lease(service, holder="midataworks")
        with pytest.raises(ModelLeasedError) as exc:
            await _lease(service, holder="mistudio-agent")
        assert exc.value.details["holder"] == "midataworks"

    async def test_unknown_model_is_404(self, env):
        service, _, _, _ = env
        with pytest.raises(ModelNotFoundError):
            await service.acquire_lease(99, "midataworks", "r", 60)

    @pytest.mark.parametrize("ttl", [0, -1, 7201, 60.5, "60", True])
    async def test_bad_ttl_is_400_naming_field_and_limit(self, env, ttl):
        service, _, registry, _ = env
        with pytest.raises(InvalidLeaseRequestError) as exc:
            await service.acquire_lease(1, "midataworks", "r", ttl)
        assert exc.value.details["param"] == "ttl_seconds"
        assert exc.value.details["max"] == 7200
        assert "7200" in exc.value.message
        assert registry.current(1) is None

    @pytest.mark.parametrize("ttl", [1, 7200])
    async def test_ttl_bounds_are_inclusive(self, env, ttl):
        service, _, _, _ = env
        assert (await service.acquire_lease(1, "h", "r", ttl)).record.ttl_seconds == ttl

    @pytest.mark.parametrize("field, holder, reason, limit", [
        ("holder", "", "r", 128), ("holder", "   ", "r", 128), ("holder", None, "r", 128),
        ("holder", "x" * 129, "r", 128), ("holder", 7, "r", 128),
        ("reason", "h", "", 512), ("reason", "h", "y" * 513, 512),
    ])
    async def test_bad_text_is_400_naming_field_and_limit(self, env, field, holder, reason, limit):
        service, _, _, _ = env
        with pytest.raises(InvalidLeaseRequestError) as exc:
            await service.acquire_lease(1, holder, reason, 60)
        assert exc.value.details["param"] == field
        assert exc.value.details["max_chars"] == limit

    async def test_text_is_stripped_and_at_the_limit_accepted(self, env):
        service, _, _, _ = env
        grant = await service.acquire_lease(1, "  " + "h" * 128 + " ", " r ", 60)
        assert grant.record.holder == "h" * 128 and grant.record.reason == "r"

    async def test_no_log_event_carries_the_lease_id(self, env):
        service, _, _, _ = env
        with capture_logs() as logs, no_probe_rows():
            grant = await _lease(service)
            await service.renew_lease(1, grant.lease_id, 60)
            with pytest.raises(ModelLeasedError):
                await service.load_model(2)
            await service.unload_model(1, lease_id=grant.lease_id)
        assert any(e["event"] == "lease_refused" for e in logs)
        for event in logs:
            for value in event.values():
                assert grant.lease_id not in str(value), event


class TestRenewReleaseResolve:
    async def test_renew_and_release_need_the_header(self, env):
        service, _, _, _ = env
        await _lease(service)
        for call in (service.renew_lease(1, None, 60), service.release_lease(1, "")):
            with pytest.raises(InvalidLeaseRequestError) as exc:
                await call
            assert exc.value.details["param"] == "X-miLLM-Lease"

    async def test_renew_validates_ttl(self, env):
        service, _, _, _ = env
        grant = await _lease(service)
        with pytest.raises(InvalidLeaseRequestError):
            await service.renew_lease(1, grant.lease_id, 7201)

    async def test_get_and_resolve(self, env):
        service, _, _, _ = env
        grant = await _lease(service)
        live, ended = await service.get_lease(1)
        assert live.holder == "midataworks" and ended is None
        assert service.resolve_lease(grant.lease_id).model_id == 1
        await service.release_lease(1, grant.lease_id)
        live, ended = await service.get_lease(1)
        assert live is None and ended.end_reason == "released"
        assert service.resolve_lease(grant.lease_id) is None

    async def test_get_on_unknown_model_is_404(self, env):
        service, _, _, _ = env
        with pytest.raises(ModelNotFoundError):
            await service.get_lease(99)


# --- 3.7: enforcement on every load path -------------------------------------------------------


class TestForeignLeaseRefusesEveryPath:
    async def test_management_load_of_another_model(self, env):
        """Control M1."""
        service, repo, _, _ = env
        await _lease(service)
        with pytest.raises(ModelLeasedError) as exc:
            await service.load_model(2)
        _assert_refusal_payload(exc.value, "load", 2)
        assert service._load_worker.call_count == 0
        assert service._loading_model_id is None, "a refused load claimed the slot"
        assert repo.status_writes == [], "a refused load moved a row"
        assert service.loader.loaded_model_id == 1

    async def test_management_unload(self, env):
        """Control M2."""
        service, repo, _, _ = env
        await _lease(service)
        with no_probe_rows(), pytest.raises(ModelLeasedError) as exc:
            await service.unload_model(1)
        _assert_refusal_payload(exc.value, "unload", 1)
        assert service.loader.loaded_model_id == 1
        assert service.loader.is_unloading is False
        assert repo.status_writes == []

    async def test_auto_load(self, env):
        """Control M3: lease AND locked → MODEL_LEASED, never model_locked."""
        service, repo, _, _ = env
        repo.rows[1].locked = True
        await _lease(service)
        with pytest.raises(ModelLeasedError) as exc:
            await service.load_model_and_wait(2, timeout=0.1)
        assert not isinstance(exc.value, ModelLockedError)
        _assert_refusal_payload(exc.value, "auto_load", 2)
        assert service._load_worker.call_count == 0
        assert repo.status_writes == []

    async def test_a_wrong_lease_id_is_still_refused(self, env):
        """Control M6."""
        service, _, _, _ = env
        grant = await _lease(service)
        with pytest.raises(ModelLeasedError):
            await service.load_model(2, lease_id=grant.lease_id + "x")
        with pytest.raises(ModelLeasedError):
            await service.load_model_and_wait(2, timeout=0.1, lease_id="wrong")
        assert service._load_worker.call_count == 0

    async def test_an_expired_lease_refuses_nothing(self, env):
        service, _, _, clock = env
        await _lease(service, ttl=120)
        clock.advance(120)
        with no_probe_rows():
            await service.load_model(2)
        drain_executor(service)
        assert service._load_worker.call_count == 1


class TestHolderProceeds:
    async def test_holder_swap_succeeds_and_ends_the_lease(self, env):
        """Control M4 (internal unload carries the ID) and M7 (unload ends the lease)."""
        service, repo, registry, _ = env
        grant = await _lease(service)
        with no_probe_rows():
            await service.load_model(2, lease_id=grant.lease_id)
        drain_executor(service)
        assert service._load_worker.call_count == 1
        assert repo.rows[1].status == ModelStatus.READY
        assert repo.rows[2].status == ModelStatus.LOADING
        assert registry.current(1) is None
        assert registry.last_ended(1).end_reason == "model_unloaded"

    async def test_holder_unload_succeeds_and_ends_the_lease(self, env):
        service, repo, registry, _ = env
        grant = await _lease(service)
        with no_probe_rows():
            await service.unload_model(1, lease_id=grant.lease_id)
        assert repo.rows[1].status == ModelStatus.READY
        assert registry.current(1) is None
        assert registry.last_ended(1).end_reason == "model_unloaded"

    async def test_unload_ends_the_lease_even_when_the_registry_could_not_see_it(self, env):
        """Control M7, isolated from the self-heal: the residency check is stubbed to still
        report the model resident, so only `end_for_model` can end the lease."""
        service, _, registry, _ = env
        grant = await _lease(service)
        registry._resident_model_id = lambda: 1
        with no_probe_rows():
            await service.unload_model(1, lease_id=grant.lease_id)
        assert registry.current(1) is None
        assert registry.last_ended(1).end_reason == "model_unloaded"

    async def test_holder_auto_load_proceeds_and_locked_still_applies(self, env):
        """The ID lifts only the lease (FR-29.3.2): `locked` still refuses."""
        service, repo, _, _ = env
        repo.rows[1].locked = True
        grant = await _lease(service)
        with pytest.raises(ModelLockedError):
            await service.load_model_and_wait(2, timeout=0.1, lease_id=grant.lease_id)

    async def test_holder_auto_load_proceeds(self, env):
        service, repo, _, _ = env
        grant = await _lease(service)

        def finish_load(model_id, *args):
            set_resident(model_id, f"m{model_id}")
            repo.rows[model_id].status = ModelStatus.LOADED

        service._load_worker.side_effect = finish_load
        with no_probe_rows():
            row = await service.load_model_and_wait(2, timeout=5, lease_id=grant.lease_id)
        assert row.id == 2 and service._load_worker.call_count == 1

    async def test_a_request_for_the_leased_resident_model_needs_no_header(self, env):
        """FR-29.2.4: a lease blocks swaps, not use."""
        service, _, _, _ = env
        await _lease(service)
        row = await service.load_model_and_wait(1, timeout=0.1)
        assert row.id == 1 and service._load_worker.call_count == 0

    async def test_wrong_header_with_no_lease_is_ignored_with_a_warning(self, env):
        service, _, _, _ = env
        with capture_logs() as logs, no_probe_rows():
            await service.load_model(2, lease_id="stale-id")
        drain_executor(service)
        assert service._load_worker.call_count == 1
        warnings = [e for e in logs if e["event"] == "lease_header_unmatched"]
        # One per guarded operation the request performed: the load, and the swap's unload.
        assert [w["operation"] for w in warnings] == ["load", "unload"]
        assert "stale-id" not in str(warnings)


# --- 3.8: AST call tests -----------------------------------------------------------------------


def _calls_in(method) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _call_name(call: ast.Call) -> str | None:
    return getattr(call.func, "attr", None) or getattr(call.func, "id", None)


class TestEnforcementIsCalledInEachMethod:
    @pytest.mark.parametrize("name, operation", [
        ("load_model", "load"), ("unload_model", "unload"), ("load_model_and_wait", "auto_load"),
    ])
    def test_refuse_if_leased_is_called(self, name, operation):
        from millm.services.model_service import ModelService

        calls = [c for c in _calls_in(getattr(ModelService, name))
                 if _call_name(c) == "_refuse_if_leased"]
        assert len(calls) == 1, f"{name} must call _refuse_if_leased exactly once"
        args = calls[0].args
        assert isinstance(args[0], ast.Constant) and args[0].value == operation
        assert isinstance(args[2], ast.Name) and args[2].id == "lease_id"

    def test_the_scanner_sees_calls_at_all(self):
        from millm.services.model_service import ModelService

        assert len(_calls_in(ModelService.load_model)) > 5

    def test_load_model_checks_before_claiming_the_slot_with_no_await_between(self):
        """The check sits immediately before the slot claim, with no await in between."""
        from millm.services.model_service import ModelService

        body = ast.parse(textwrap.dedent(inspect.getsource(ModelService.load_model))).body[0].body
        idx_check = next(
            i for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and _call_name(stmt.value) == "_refuse_if_leased"
        )
        idx_claim = next(
            i for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign)
            and any(getattr(t, "attr", None) == "_loading_model_id" for t in stmt.targets)
        )
        assert idx_check < idx_claim
        between = body[idx_check:idx_claim + 1]
        assert not any(isinstance(n, ast.Await) for s in between for n in ast.walk(s))

    def test_internal_unload_passes_the_lease_id(self):
        from millm.services.model_service import ModelService

        unloads = [c for c in _calls_in(ModelService.load_model) if _call_name(c) == "unload_model"]
        assert len(unloads) == 1
        assert any(kw.arg == "lease_id" and getattr(kw.value, "id", None) == "lease_id"
                   for kw in unloads[0].keywords)

    def test_auto_load_passes_the_lease_id_to_load_model(self):
        from millm.services.model_service import ModelService

        loads = [c for c in _calls_in(ModelService.load_model_and_wait)
                 if _call_name(c) == "load_model"]
        assert len(loads) == 1
        assert any(kw.arg == "lease_id" for kw in loads[0].keywords)


class TestSelfHealingRead:
    """Task 4.5: the loader emptied WITHOUT unload_model (the forced path) → no lease."""

    async def test_forced_unload_leaves_no_lease(self, env):
        """Control M8."""
        service, _, registry, _ = env
        await _lease(service)
        service.loader.unload()  # what the timeout branch does; unload_model's success never ran
        assert registry.current(1) is None
        assert registry.last_ended(1).end_reason == "model_unloaded"
        live, _ = await service.get_lease(1)
        assert live is None

    async def test_a_new_model_loaded_behind_the_lease_is_not_leased(self, env):
        service, _, registry, _ = env
        await _lease(service)
        set_resident(2, "m2")  # residency moved without the service seeing it
        assert registry.current(1) is None
        assert registry.current(2) is None
