"""Which model a `/v1` request means (032 FR-32.6.3-4, FTDD §5.5 as amended by review 5.3).

One resolver for chat, completions, embeddings, `GET /v1/models/{id}` and the batch validator.

The resident model can answer to names no database row carries: after a checkpoint load its
served name is the manifest's, and while an adapter is applied it is the adapter's (033). A
request naming either must reach the resident without a lookup by name and without a load; a
request naming the bare base still means the resident (033 D12). Everything else resolves through
the database as before.

Rules, in order:

1. The name is the resident's served name or its base served name → `resident` (the carrier row,
   no load).
2. No row by that name → today's 404 (`model_not_found`).
3. The row is the resident's carrier, the resident identity has moved off it (a checkpoint was
   saved from it or loaded over it), and the request names the row:
   * a weights-origin row → REFUSED, 404 with the load sentence. Reloading it would read its
     directory through the ordinary loader with no hash check (review 5.3, D18).
   * a plain model row → REFUSED, 404 `model_not_found` naming the resident's served name and
     `POST /api/weights/load {"model_id": N}` (D-38). ⚠ Never `reload=True` implicitly: the
     reload would discard the trained weights and any unsaved commits on the strength of a name
     a client typed. The deliberate way back is the D-01 load, which the message names.
2a. The row was found only through `find_by_name`'s bare-name fallback (the request named a
   quantization repo, `…-GGUF`, and the row is `…-GGUF:Q4_K_M`) → REFUSED, 404
   `model_not_found` naming the exact name (D-46). No resident is served under a bare name, so
   the routes refused it after the auto-load anyway (evicting the resident first) and the batch
   D-31 re-check refused it at run time; refusing here makes validation and run agree.
4. A weights-origin row that is not resident → REFUSED, 404: such a row is never auto-loaded
   (FR-32.6.4); it is loaded through `/api/weights/load`.
5. Otherwise → `row` (loaded on demand, as today). A `/v1` auto-load of it is still refused while
   the resident weights carry unsaved commits (D-45, `ModelService._refuse_if_unsaved_commits`).

`weights_origin` is the column migration 021 added; `weights_origin_of` reads it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from fastapi.responses import JSONResponse

from millm.api.routes.openai.errors import create_openai_error, model_not_found_error
from millm.core.errors import ModelNotFoundError
from millm.core.logging import get_logger

logger = get_logger(__name__)

#: The sentence a refused weights-origin row carries (FR-32.6.4).
LOAD_THROUGH_WEIGHTS_SENTENCE = (
    "It holds weights from a checkpoint, which are loaded only through "
    "POST /api/weights/load (where the files are verified against their manifest), never "
    "on demand by a request."
)


@dataclass(frozen=True)
class Resolution:
    """The answer for one requested name.

    Exactly one of: `refusal` set (return it); `resident` True (serve the resident, no load);
    otherwise `row` is to be loaded on demand (`reload` is never set since D-38; see
    `load_kwargs`).
    """

    row: Any = None
    resident: bool = False
    reload: bool = False
    refusal: Optional[JSONResponse] = None
    #: `not_found` (no such row), `weights_origin` (a checkpoint row not loadable on demand),
    #: `identity_moved` (a plain row whose resident identity moved off it, D-38) or `bare_name`
    #: (a quantization repo named without its tag, D-46); None when not refused.
    refusal_reason: Optional[str] = None
    refusal_message: Optional[str] = None
    #: The resident's served name, set on a `resident` resolution (what `/v1` reports for it).
    served_name: Optional[str] = None


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def weights_origin_of(row: Any) -> Optional[str]:
    """The row's `weights_origin` (`checkpoint` / `instantiated`), or None for a plain model row.

    Only a non-empty string counts, as the column's type (migration 021) allows nothing else;
    `getattr` keeps the resolver usable with row stand-ins that omit the column.
    """
    value = getattr(row, "weights_origin", None)
    return value if isinstance(value, str) and value else None


def resident_names(info: Any) -> set[str]:
    """Every name that means the resident model, from a `LoadedModelInfo`: its served name and its
    base served name. Falls back to the row name when those are absent (a plain row load, or an
    info object that predates them)."""
    if info is None:
        return set()
    row_name = _text(getattr(info, "name", None))
    names = {
        _text(getattr(info, "served_name", None)) or row_name,
        _text(getattr(info, "base_served_name", None)) or row_name,
    }
    return {n for n in names if n}


def served_name_of(info: Any) -> Optional[str]:
    """The name `/v1` reports for the resident described by `info`."""
    if info is None:
        return None
    return _text(getattr(info, "served_name", None)) or _text(getattr(info, "name", None))


def _identity_moved(info: Any) -> bool:
    """Whether the resident identity is no longer the carrier row's own (a checkpoint was loaded
    over it): the base served name differs from the row name."""
    row_name = _text(getattr(info, "name", None))
    base = _text(getattr(info, "base_served_name", None)) or row_name
    return row_name is not None and base != row_name


def _weights_origin_refusal(name: str, *, resident: Optional[str]) -> Resolution:
    detail = (
        f" The resident model is '{resident}'." if resident else ""
    )
    message = f"The model '{name}' is not resident.{detail} {LOAD_THROUGH_WEIGHTS_SENTENCE}"
    return Resolution(
        refusal=create_openai_error(
            message=message,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
            status_code=404,
        ),
        refusal_reason="weights_origin",
        refusal_message=message,
    )


def _identity_moved_refusal(name: str, row: Any, *, resident: Optional[str]) -> Resolution:
    """D-38: the request names a plain row whose resident identity moved off it (a save or a 032
    load). Refused with the unknown-model shape (404, `model_not_found`, `param: model`); the
    message names what is resident and the deliberate way back to the row's own weights."""
    row_id = getattr(row, "id", None)
    now = f"'{resident}'" if resident else "a checkpoint"
    message = (
        f"The model '{name}' is not resident: the resident model is now {now}, whose weights "
        f"were saved from or loaded over the row '{name}'. Name {now} to use it. Reloading "
        f"'{name}' from its own files would discard those weights, so it is never done on "
        f"demand; to do it deliberately, call POST /api/weights/load "
        f'{{"model_id": {row_id}}}.'
    )
    return Resolution(
        refusal=create_openai_error(
            message=message,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
            status_code=404,
        ),
        refusal_reason="identity_moved",
        refusal_message=message,
    )


def _bare_name_refusal(name: str, row: Any) -> Resolution:
    """D-46: the request named a model only by a prefix of its row name (a quantization repo
    without its `:QUANT` tag). The unknown-model shape (404, `model_not_found`, `param: model`),
    with the row's exact name as the available one, exactly as the routes' post-load check
    worded it (`model_not_found_error(name, served)`)."""
    exact = getattr(row, "name", None)
    response = model_not_found_error(name, exact if isinstance(exact, str) else None)
    message = json.loads(bytes(response.body))["error"]["message"]
    return Resolution(refusal=response, refusal_reason="bare_name", refusal_message=message)


async def _carrier_row(name: str, info: Any, service: Any) -> Any:
    """The database row the resident was loaded from. By name first, which is the common case (a
    plain row load named by its row name); by id when the name is one no row carries (a manifest
    or adapter name), or names another row."""
    row = await service.find_model_by_name(name)
    if row is not None and getattr(row, "name", None) == getattr(info, "name", None):
        return row
    try:
        return await service.get_model(info.model_id)
    except ModelNotFoundError:
        return None


async def resolve_requested_model(name: str, service: Any, inference: Any) -> Resolution:
    """Resolve a `/v1` request's `model` (module docstring has the rules).

    Reads the resident once (`inference.get_loaded_model_info()`); that read replaces the routes'
    own pre-load residency check.
    """
    info = inference.get_loaded_model_info()
    if info is not None and name in resident_names(info):
        carrier = await _carrier_row(name, info, service)
        if carrier is not None:
            return Resolution(row=carrier, resident=True, served_name=served_name_of(info))
        logger.warning("resident_carrier_row_missing", model=name,
                       model_id=getattr(info, "model_id", None))

    row = await service.find_model_by_name(name)
    if row is None:
        return Resolution(refusal=model_not_found_error(name), refusal_reason="not_found")

    if getattr(row, "name", None) != name:
        # D-46 (review R2-A F-B): `find_by_name` maps a bare quantization repo name (`…-GGUF`)
        # onto its one tagged row (`…-GGUF:Q4_K_M`). A `/v1` request is answered only under a
        # name the resident is served by, which a bare name never is, so the routes 404'd it
        # AFTER the auto-load (evicting the resident first) and a batch line passed validation
        # only to fail the D-31 re-check at run time. Refused here, before anything loads: the
        # same 404 `model_not_found` the routes answered, naming the exact name to send.
        logger.info("v1_bare_model_name_refused", model=name, row=getattr(row, "name", None))
        return _bare_name_refusal(name, row)

    weights_origin = weights_origin_of(row)
    is_carrier = info is not None and getattr(row, "name", None) == getattr(info, "name", None)

    if is_carrier and _identity_moved(info) and getattr(row, "name", None) == name:
        if weights_origin:
            # ⚠ REFUSED, never reload=True: the reload would bypass the hash check (review 5.3).
            logger.info("v1_weights_origin_refused", model=name, reason="identity_moved")
            return _weights_origin_refusal(name, resident=served_name_of(info))
        # ⚠ REFUSED, never reload=True (D-38): the reload would discard the trained weights.
        logger.info("v1_identity_moved_refused", model=name,
                    resident=served_name_of(info), model_id=getattr(row, "id", None))
        return _identity_moved_refusal(name, row, resident=served_name_of(info))

    if weights_origin and not is_carrier:
        logger.info("v1_weights_origin_refused", model=name, reason="not_resident")
        return _weights_origin_refusal(name, resident=served_name_of(info))

    return Resolution(row=row)


def batch_model_resolver(
    service: Any, inference: Any
) -> Callable[[str], Awaitable[Any]]:
    """The batch validator's `resolve_model` callable (`validator.validate_file`), keeping its
    signature: a name → the row, None for an unknown name, and a refusal raised as a
    `MiLLMError` (the validator records its code, message and status for every line naming it).
    """

    async def resolve(name: str) -> Any:
        resolution = await resolve_requested_model(name, service, inference)
        if resolution.refusal_reason == "not_found":
            return None  # the validator writes its own "does not exist" refusal
        if resolution.refusal is not None:
            raise ModelNotFoundError(
                resolution.refusal_message or f"The model '{name}' is refused.",
                details={"param": "model", "reason": resolution.refusal_reason},
            )
        return resolution.row

    return resolve


def load_kwargs(resolution: Resolution) -> dict[str, Any]:
    """Keyword arguments for `ModelService.load_model_and_wait` on a non-resident resolution.

    ⚠ No resolution sets `reload` any more: rule 3 refuses a plain row whose identity moved
    (D-38) rather than reloading it on demand. The keyword path stays so a route never has to
    know which rule produced its resolution.
    """
    return {"reload": True} if resolution.reload else {}
