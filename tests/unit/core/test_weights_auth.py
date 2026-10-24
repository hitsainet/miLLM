"""Bearer-token identity for the gated weights routes (031 FR-31.14; addendum D-05).

The gate FAILS CLOSED: an empty or malformed `WEIGHTS_TOKENS` refuses every gated call, and a
malformed one never stops the server. A token, and any configured value, never reaches a log
line or a response. Through a real FastAPI app with the real `millm_error_handler`, a 401
carries `WWW-Authenticate: Bearer` (RFC 6750).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from millm.core import weights_auth
from millm.core.config import settings
from millm.core.errors import MiLLMError, WeightsForbiddenError, WeightsUnauthorizedError
from millm.core.weights_auth import (
    OperatorIdentityDep,
    TokenSet,
    WeightsIdentity,
    WeightsIdentityDep,
    authenticate,
    current_token_set,
    parse_weights_tokens,
    require_weights_identity,
)

TRAINER_TOKEN = "trainer-secret-token-0123456789abcdef"
OPERATOR_TOKEN = "operator-secret-token-fedcba9876543210"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _tokens(*entries: tuple[str, str, str]) -> str:
    return json.dumps([{"label": l, "role": r, "sha256": _digest(t)} for l, r, t in entries])


VALID = _tokens(("miforge", "trainer", TRAINER_TOKEN), ("ops", "operator", OPERATOR_TOKEN))


@pytest.fixture
def configured(monkeypatch):
    def install(raw: str) -> None:
        monkeypatch.setattr(settings, "WEIGHTS_TOKENS", raw)

    return install


class TestParse:
    def test_empty_is_a_legal_empty_set_that_refuses(self):
        for raw in ("", "   ", "[]"):
            tokens = parse_weights_tokens(raw)
            assert tokens.entries == () and tokens.problem is None
            assert tokens.refusal_reason == "no_tokens_configured"

    def test_valid_entries(self):
        tokens = parse_weights_tokens(VALID)
        assert tokens.problem is None and tokens.refusal_reason is None
        assert [(e.label, e.role) for e in tokens.entries] == [
            ("miforge", "trainer"), ("ops", "operator")]

    def test_upper_case_digest_is_lowered(self):
        raw = json.dumps([{"label": "a", "role": "trainer", "sha256": _digest("t").upper()}])
        assert parse_weights_tokens(raw).entries[0].sha256 == _digest("t")

    @pytest.mark.parametrize("raw, fragment", [
        ("{not json", "not valid JSON"),
        ('{"label": "a"}', "not a JSON array"),
        ('["x"]', "entry 0 is not an object"),
        ('[{"label": "a", "role": "trainer"}]', "entry 0 must have exactly"),
        (json.dumps([{"label": "a", "role": "trainer", "sha256": "0" * 64, "x": 1}]),
         "entry 0 must have exactly"),
        (json.dumps([{"label": "", "role": "trainer", "sha256": "0" * 64}]), "empty"),
        (json.dumps([{"label": "a", "role": "admin", "sha256": "0" * 64}]), "role"),
        (json.dumps([{"label": "a", "role": "trainer", "sha256": "abc"}]), "64 hex"),
        (json.dumps([{"label": "a", "role": "trainer", "sha256": "0" * 64},
                     {"label": "a", "role": "operator", "sha256": "1" * 64}]),
         "entry 1 repeats a label"),
        (json.dumps([{"label": "a", "role": "trainer", "sha256": "0" * 64},
                     {"label": "b", "role": "operator", "sha256": "0" * 64}]),
         "entry 1 repeats a digest"),
    ])
    def test_malformed_refuses_everything_and_names_the_entry(self, raw, fragment):
        tokens = parse_weights_tokens(raw)
        assert tokens.entries == ()
        assert fragment in tokens.problem
        assert tokens.refusal_reason == "tokens_malformed"

    def test_one_bad_entry_drops_the_good_ones_too(self):
        """Fail closed: a half-parsed set would admit the entries before the typo."""
        raw = json.dumps([{"label": "a", "role": "trainer", "sha256": _digest(TRAINER_TOKEN)},
                          {"label": "b", "role": "trainer", "sha256": "nope"}])
        tokens = parse_weights_tokens(raw)
        assert tokens.entries == () and tokens.identify(TRAINER_TOKEN) is None

    @pytest.mark.parametrize("raw", ["null", "1", '"s"', "[[]]", "[null]", "\x00", "[" * 50])
    def test_never_raises(self, raw):
        assert isinstance(parse_weights_tokens(raw), TokenSet)

    def test_a_plaintext_token_pasted_into_sha256_is_never_quoted(self):
        raw = json.dumps([{"label": "a", "role": "trainer", "sha256": TRAINER_TOKEN}])
        tokens = parse_weights_tokens(raw)
        assert tokens.problem is not None and TRAINER_TOKEN not in tokens.problem
        assert TRAINER_TOKEN not in repr(tokens)

    def test_the_settings_repr_hides_the_value(self, configured):
        configured(VALID)
        assert _digest(TRAINER_TOKEN) not in repr(settings)


class TestIdentify:
    def test_the_matching_digest_names_label_and_role(self):
        tokens = parse_weights_tokens(VALID)
        assert tokens.identify(TRAINER_TOKEN) == WeightsIdentity("miforge", "trainer")
        assert tokens.identify(OPERATOR_TOKEN) == WeightsIdentity("ops", "operator")
        assert tokens.identify("wrong") is None

    def test_every_entry_is_compared_with_no_early_exit(self, monkeypatch):
        calls: list[Any] = []
        real = hmac.compare_digest

        def spy(a, b):
            calls.append(b)
            return real(a, b)

        monkeypatch.setattr(weights_auth.hmac, "compare_digest", spy)
        tokens = parse_weights_tokens(VALID)
        # The first entry matches; the second must still be compared.
        assert tokens.identify(TRAINER_TOKEN) == WeightsIdentity("miforge", "trainer")
        assert calls == [e.sha256 for e in tokens.entries]


class TestAuthenticate:
    def test_accepts_a_valid_token_in_role(self, configured):
        configured(VALID)
        assert authenticate(f"Bearer {TRAINER_TOKEN}", ("trainer", "operator")).label == "miforge"
        assert authenticate(f"bearer  {OPERATOR_TOKEN} ", ("operator",)).role == "operator"

    @pytest.mark.parametrize("raw, header, reason", [
        ("", f"Bearer {TRAINER_TOKEN}", "no_tokens_configured"),
        ("[]", f"Bearer {TRAINER_TOKEN}", "no_tokens_configured"),
        ("{bad", f"Bearer {TRAINER_TOKEN}", "tokens_malformed"),
        (VALID, None, "missing"),
        (VALID, "", "missing"),
        (VALID, "Bearer ", "missing"),
        (VALID, f"Basic {TRAINER_TOKEN}", "missing"),
        (VALID, TRAINER_TOKEN, "missing"),
        (VALID, "Bearer not-a-configured-token", "invalid_token"),
    ])
    def test_fails_closed(self, configured, raw, header, reason):
        configured(raw)
        with pytest.raises(WeightsUnauthorizedError) as info:
            authenticate(header, ("trainer", "operator"))
        assert info.value.details == {"reason": reason}
        assert info.value.headers == {"WWW-Authenticate": "Bearer"}

    def test_a_valid_token_outside_the_roles_is_forbidden(self, configured):
        configured(VALID)
        with pytest.raises(WeightsForbiddenError) as info:
            authenticate(f"Bearer {TRAINER_TOKEN}", ("operator",))
        assert info.value.status_code == 403
        assert info.value.details["required_roles"] == ["operator"]

    def test_a_changed_setting_is_reparsed(self, configured):
        configured(VALID)
        assert current_token_set().refusal_reason is None
        configured("")
        assert current_token_set().refusal_reason == "no_tokens_configured"


class TestFactory:
    def test_the_factory_refuses_no_roles_so_passing_it_bare_fails_loudly(self):
        """033 review X3: `Depends(require_weights_identity)` would inject a function."""
        with pytest.raises(ValueError):
            require_weights_identity()
        with pytest.raises(ValueError):
            require_weights_identity("admin")


class _Recorder:
    """A stand-in logger recording every call's event and fields."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str):
        def record(event: str, *args: Any, **kwargs: Any) -> None:
            self.calls.append((level, event, kwargs))

        return record

    def text(self) -> str:
        return json.dumps(self.calls, default=str)


@pytest.fixture
def recorder(monkeypatch):
    """Every logger the gated path writes through: auth, audit and the error handler."""
    import millm.api.exception_handlers as handlers
    import millm.services.weights_audit as audit

    rec = _Recorder()
    for module in (weights_auth, audit, handlers):
        monkeypatch.setattr(module, "logger", rec)
    return rec


def _app() -> TestClient:
    from millm.api.exception_handlers import millm_error_handler

    app = FastAPI()
    app.add_exception_handler(MiLLMError, millm_error_handler)

    @app.post("/api/weights/_gated", name="weights_test_gated")
    async def gated(identity: WeightsIdentityDep) -> dict:
        return {"label": identity.label, "role": identity.role}

    @app.post("/api/weights/_operator", name="weights_test_operator")
    async def operator_only(identity: OperatorIdentityDep) -> dict:
        return {"label": identity.label}

    @app.post("/api/weights/_explicit")
    async def explicit(identity: WeightsIdentity = Depends(require_weights_identity("trainer"))):
        return {"label": identity.label}

    return TestClient(app)


class TestThroughTheApp:
    def test_accepted(self, configured, recorder):
        configured(VALID)
        resp = _app().post("/api/weights/_gated",
                           headers={"Authorization": f"Bearer {TRAINER_TOKEN}"})
        assert resp.status_code == 200
        assert resp.json() == {"label": "miforge", "role": "trainer"}

    def test_401_carries_www_authenticate_and_is_audited(self, configured, recorder):
        configured(VALID)
        resp = _app().post("/api/weights/_gated")
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"
        body = resp.json()
        assert body["error"]["code"] == "WEIGHTS_UNAUTHORIZED"
        assert body["error"]["details"]["reason"] == "missing"
        audits = [c for c in recorder.calls if c[1] == "weights_audit"]
        assert len(audits) == 1
        level, _, fields = audits[0]
        assert level == "warning"
        assert fields["operation"] == "weights_test_gated"
        assert fields["outcome"] == "refused" and fields["code"] == "WEIGHTS_UNAUTHORIZED"
        assert fields["reason"] == "missing"
        assert fields["role"] is None and fields["label"] is None

    def test_no_tokens_configured_refuses_a_well_formed_request(self, configured, recorder):
        configured("")
        resp = _app().post("/api/weights/_gated",
                           headers={"Authorization": f"Bearer {TRAINER_TOKEN}"})
        assert resp.status_code == 401
        assert resp.json()["error"]["details"]["reason"] == "no_tokens_configured"

    def test_403_names_the_caller_in_the_audit(self, configured, recorder):
        configured(VALID)
        resp = _app().post("/api/weights/_operator",
                           headers={"Authorization": f"Bearer {TRAINER_TOKEN}"})
        assert resp.status_code == 403
        assert "WWW-Authenticate" not in resp.headers
        audits = [c[2] for c in recorder.calls if c[1] == "weights_audit"]
        assert [(a["code"], a["role"], a["label"]) for a in audits] == [
            ("WEIGHTS_FORBIDDEN", "trainer", "miforge")]

    def test_an_explicit_factory_call_works(self, configured, recorder):
        configured(VALID)
        client = _app()
        ok = client.post("/api/weights/_explicit",
                         headers={"Authorization": f"Bearer {TRAINER_TOKEN}"})
        refused = client.post("/api/weights/_explicit",
                              headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"})
        assert ok.status_code == 200 and refused.status_code == 403

    def test_no_token_or_configured_value_reaches_a_log_line_or_a_body(self, configured, recorder):
        """FR-31.14.9 over every outcome, with a plaintext token pasted into `sha256` too."""
        client = _app()
        bodies: list[str] = []
        pasted = json.dumps([{"label": "a", "role": "trainer", "sha256": TRAINER_TOKEN}])
        for raw in (VALID, "", "{bad " + TRAINER_TOKEN, pasted):
            configured(raw)
            for token in (TRAINER_TOKEN, OPERATOR_TOKEN, "Bearer-less"):
                for path in ("/api/weights/_gated", "/api/weights/_operator"):
                    resp = client.post(path, headers={"Authorization": f"Bearer {token}"})
                    bodies.append(resp.text)
                    bodies.append(json.dumps(dict(resp.headers)))
        haystack = recorder.text() + "".join(bodies)
        assert recorder.calls, "the recorder saw nothing: the check would pass vacuously"
        for secret in (TRAINER_TOKEN, OPERATOR_TOKEN, _digest(TRAINER_TOKEN),
                       _digest(OPERATOR_TOKEN)):
            assert secret not in haystack
