"""`model_lease.refuse_if_leased`: THE one lease-enforcement rule (029 FTDD §7.2, D-15), shared by
`ModelService` and `AdapterService` (W3 consolidation of W2-H's duplicate).

Fixtures disagree with the defects they guard: the resident (1) differs from the target (2), and
the refusal must name the TARGET while reading the RESIDENT's lease.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from millm.core.errors import ModelLeasedError
from millm.services.model_lease import LeaseRegistry, refuse_if_leased


@pytest.fixture
def reg():
    return LeaseRegistry(resident_model_id=lambda: 1, ended_memory=4)


def _call(reg, lease_id=None, operation="load", target=2):
    refuse_if_leased(
        reg, operation=operation, resident_model_id=1, target_model_id=target, lease_id=lease_id
    )


def test_no_lease_passes(reg):
    _call(reg)


def test_another_holders_lease_refuses_naming_the_target(reg):
    reg.grant(1, "m1", "midataworks", "label run 7", 120)
    with capture_logs() as logs, pytest.raises(ModelLeasedError) as exc:
        _call(reg, operation="unload", target=2)
    assert exc.value.details["holder"] == "midataworks"
    assert exc.value.details["leased_model_id"] == 1
    assert exc.value.details["target_model_id"] == 2
    assert exc.value.details["operation"] == "unload"
    refused = [e for e in logs if e["event"] == "lease_refused"]
    assert len(refused) == 1 and refused[0]["target_model_id"] == 2


def test_the_holder_passes(reg):
    grant = reg.grant(1, "m1", "forge", "run 9", 120)
    _call(reg, lease_id=grant.lease_id)


def test_a_wrong_id_still_refuses(reg):
    reg.grant(1, "m1", "forge", "run 9", 120)
    with pytest.raises(ModelLeasedError):
        _call(reg, lease_id="not-the-id")


def test_an_unneeded_id_is_ignored_with_a_warning_that_never_logs_it(reg):
    with capture_logs() as logs:
        _call(reg, lease_id="secret-lease-id")
    unmatched = [e for e in logs if e["event"] == "lease_header_unmatched"]
    assert len(unmatched) == 1
    assert unmatched[0]["resident_model_id"] == 1 and unmatched[0]["target_model_id"] == 2
    assert "secret-lease-id" not in repr(logs)


def test_it_reads_the_resident_lease_not_the_targets():
    """A lease on model 2 (not resident) must not stop an operation on the resident 1."""
    reg = LeaseRegistry(resident_model_id=lambda: 2, ended_memory=4)
    reg.grant(2, "m2", "midataworks", "x", 120)
    # The resident passed in is 1, which carries no lease.
    refuse_if_leased(reg, operation="load", resident_model_id=1, target_model_id=2, lease_id=None)
