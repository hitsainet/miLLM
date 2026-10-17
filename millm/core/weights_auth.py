"""Bearer-token identity for the gated `/api/weights` routes (Features 31, 32, 33).

The first authentication in miLLM (FR-31.14). Contract: addendum §2.1 and D-05.

- **Digests only.** `WEIGHTS_TOKENS` holds `{label, role, sha256}` entries; the plaintext lives
  in the trainer's own Secret, so a leaked miLLM configuration grants nothing.
- **Fails closed.** An empty setting refuses every gated call (`no_tokens_configured`); a
  malformed one refuses every gated call too (`tokens_malformed`). Neither stops the server:
  this is an optional feature, and `settings = Settings()` runs at import, so a raising parse
  would turn a typo in one Secret into a serving outage. That is why the setting is a raw
  string and is parsed here, never by pydantic, whose errors echo their input.
- **Constant time.** The presented token is digested once and compared with EVERY configured
  digest by `hmac.compare_digest`, with no early exit, so timing does not reveal a position.
- **Never logged.** No line here, in the audit, or in a refusal's `details` carries a token or
  a configured value. A malformed entry is named by its index and the rule it broke.

Check order inside the dependency: the token set (malformed, empty), the header (missing), the
digest (unknown), the role (403). Route-specific checks (attachment holder, state) come after.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
from dataclasses import dataclass
from typing import Annotated, Awaitable, Callable, Literal, Optional

from fastapi import Depends, Header, Request

from millm.core.errors import WeightsForbiddenError, WeightsUnauthorizedError
from millm.core.logging import get_logger

logger = get_logger(__name__)

Role = Literal["trainer", "operator"]
ROLES: tuple[str, ...] = ("trainer", "operator")

#: `fullmatch`, never `match`: `$` also matches before a trailing newline, so `match` accepted
#: 64 hex digits plus a trailing newline and stored a digest no token can ever equal (a silently dead entry).
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ENTRY_KEYS = frozenset({"label", "role", "sha256"})


@dataclass(frozen=True)
class WeightsIdentity:
    """Who made a gated call: the label and role of the token that matched."""

    label: str
    role: Role


@dataclass(frozen=True)
class TokenEntry:
    """One configured token: its label, role and SHA-256 digest (lower-case hex)."""

    label: str
    role: Role
    sha256: str

    def __repr__(self) -> str:  # the digest is not a secret, but nothing gains from printing it
        return f"TokenEntry(label={self.label!r}, role={self.role!r})"


@dataclass(frozen=True)
class TokenSet:
    """The parsed `WEIGHTS_TOKENS`. `problem` is set when the value did not parse, in which case
    `entries` is empty and every gated call is refused with `tokens_malformed`."""

    entries: tuple[TokenEntry, ...] = ()
    problem: Optional[str] = None

    @property
    def refusal_reason(self) -> Optional[str]:
        """Why no token can be accepted at all, or None when the set can accept tokens."""
        if self.problem is not None:
            return "tokens_malformed"
        if not self.entries:
            return "no_tokens_configured"
        return None

    def identify(self, token: str) -> Optional[WeightsIdentity]:
        """The identity whose digest equals the token's, or None. Constant time over the set."""
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        matched: Optional[TokenEntry] = None
        for entry in self.entries:
            # Every entry is compared, whatever matched earlier: no early exit.
            if hmac.compare_digest(digest, entry.sha256) and matched is None:
                matched = entry
        if matched is None:
            return None
        return WeightsIdentity(label=matched.label, role=matched.role)


def _malformed(problem: str) -> TokenSet:
    return TokenSet(entries=(), problem=problem)


def parse_weights_tokens(raw: str) -> TokenSet:
    """Parse `WEIGHTS_TOKENS`. Never raises: a malformed value returns a set that refuses.

    Empty (or whitespace, or `[]`) is a legal empty set. Otherwise a JSON array of objects with
    exactly `label` (non-empty string, unique), `role` (`trainer` or `operator`) and `sha256`
    (64 hex digits; upper case is accepted and lowered). Two entries with one digest are
    refused too: the token would mean two identities. `problem` names the entry by index and
    the rule it broke, and never quotes a value.
    """
    text = (raw or "").strip()
    if not text:
        return TokenSet()
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as e:
        # A JSONDecodeError's message gives a position, never content; keep only that.
        position = getattr(e, "pos", None)
        return _malformed(f"not valid JSON (at character {position})")
    except RecursionError:
        # Deeply nested input ("[[[[...") exhausts the decoder's recursion: still malformed, and
        # still never a raise (a raise here was a 500 on every gated call, unaudited).
        return _malformed("not valid JSON (nested too deeply)")
    if not isinstance(data, list):
        return _malformed("not a JSON array")
    entries: list[TokenEntry] = []
    labels: set[str] = set()
    digests: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            return _malformed(f"entry {index} is not an object")
        keys = set(item)
        if keys != _ENTRY_KEYS:
            missing = sorted(_ENTRY_KEYS - keys)
            extra = len(keys - _ENTRY_KEYS)
            return _malformed(
                f"entry {index} must have exactly label, role and sha256 "
                f"(missing {missing}, {extra} unknown key(s))"
            )
        label, role, digest = item["label"], item["role"], item["sha256"]
        if not isinstance(label, str) or not label.strip():
            return _malformed(f"entry {index} has an empty or non-string label")
        if role not in ROLES:
            return _malformed(f"entry {index} has a role other than trainer or operator")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest.lower()):
            return _malformed(f"entry {index} sha256 is not 64 hex digits")
        digest = digest.lower()
        if label in labels:
            return _malformed(f"entry {index} repeats a label")
        if digest in digests:
            return _malformed(f"entry {index} repeats a digest")
        labels.add(label)
        digests.add(digest)
        entries.append(TokenEntry(label=label, role=role, sha256=digest))
    return TokenSet(entries=tuple(entries))


_cache_lock = threading.Lock()
_cached_raw: Optional[str] = None
_cached_tokens = TokenSet()


def current_token_set() -> TokenSet:
    """The token set for the live `settings.WEIGHTS_TOKENS`, parsed once per distinct value.

    Read per call (cheaply, from the cache) rather than frozen at import, so a test that sets
    the setting is honoured. A malformed value is logged once, by index and rule only.
    """
    global _cached_raw, _cached_tokens
    from millm.core.config import settings

    raw = settings.WEIGHTS_TOKENS
    with _cache_lock:
        if _cached_raw != raw:
            tokens = parse_weights_tokens(raw)
            _cached_raw, _cached_tokens = raw, tokens
            if tokens.problem is not None:
                logger.warning(
                    "weights_tokens_malformed",
                    problem=tokens.problem,
                    effect="every gated /api/weights call is refused until WEIGHTS_TOKENS is fixed",
                )
        return _cached_tokens


def _bearer(authorization: Optional[str]) -> Optional[str]:
    """The token from an `Authorization: Bearer <token>` header, or None."""
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def authenticate(
    authorization: Optional[str], roles: tuple[str, ...], operation: Optional[str] = None
) -> WeightsIdentity:
    """The identity behind `authorization`, allowed one of `roles`; raises the refusal.
    `operation` (the route name) only shapes the refusal message.

    Raises:
        WeightsUnauthorizedError: tokens malformed or none configured, header missing or not
            Bearer, or a token no configured digest matches.
        WeightsForbiddenError: a valid token whose role is not in `roles`.
    """
    tokens = current_token_set()
    reason = tokens.refusal_reason
    if reason is not None:
        raise WeightsUnauthorizedError.for_reason(reason, operation)
    token = _bearer(authorization)
    if token is None:
        raise WeightsUnauthorizedError.for_reason("missing")
    identity = tokens.identify(token)
    if identity is None:
        raise WeightsUnauthorizedError.for_reason("invalid_token")
    if identity.role not in roles:
        raise WeightsForbiddenError.for_roles(identity.role, roles)
    return identity


#: Audit `operation` names for gated routes whose route `name` is not already `weights_<op>`
#: (addendum D-42: one name per operation everywhere, token refusals included). The 032 routes
#: take FastAPI's default name, their function name, so a token refusal would otherwise log
#: `save_checkpoint` while the service logs the operation itself. A route whose `name` is
#: already `weights_<op>` passes through unchanged.
AUDIT_OPERATION_BY_ROUTE_NAME: dict[str, str] = {
    "save_checkpoint": "weights_save",
    "load_checkpoint": "weights_load",
}


def audit_operation(route_name: str) -> str:
    """The audit `operation` for a gated route's name (D-42)."""
    return AUDIT_OPERATION_BY_ROUTE_NAME.get(route_name, route_name)


def require_weights_identity(*roles: str) -> Callable[..., Awaitable[WeightsIdentity]]:
    """A FastAPI dependency FACTORY: the dependency returns the caller's `WeightsIdentity`.

    Use `WeightsIdentityDep` / `OperatorIdentityDep`, or `Depends(require_weights_identity(...))`
    WITH roles. `Depends(require_weights_identity)` (the factory itself) fails loudly on every
    request instead of injecting a function, because the factory refuses an empty role list.
    Refusals are audited here (FR-31.15.3); the identity is None for a 401.
    """
    if not roles or any(role not in ROLES for role in roles):
        raise ValueError(f"require_weights_identity needs roles from {ROLES}, got {roles!r}")
    allowed = tuple(roles)

    async def dependency(
        request: Request,
        authorization: Annotated[Optional[str], Header()] = None,
    ) -> WeightsIdentity:
        route = request.scope.get("route")
        operation = audit_operation(route.name) if route is not None else request.url.path
        try:
            return authenticate(authorization, allowed, operation)
        except (WeightsUnauthorizedError, WeightsForbiddenError) as exc:
            from millm.services.weights_audit import weights_audit

            identity = None
            if isinstance(exc, WeightsForbiddenError):
                identity = current_token_set().identify(_bearer(authorization) or "")
            weights_audit(
                operation,
                "refused",
                identity,
                code=exc.code,
                reason=exc.details.get("reason"),
            )
            raise

    return dependency


WeightsIdentityDep = Annotated[
    WeightsIdentity, Depends(require_weights_identity("trainer", "operator"))
]
OperatorIdentityDep = Annotated[WeightsIdentity, Depends(require_weights_identity("operator"))]
