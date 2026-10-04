"""A probe row names its rolling window and states each window's own bar (operator, 2026-10-04).

Two `rolling_mean_max` probes at one layer — window 32 and window 64 — differ ONLY in
`aggregation.params`, and the list did not send it, so their tiles read identically. Nor did it
send `decision.windows`, so nothing said which bar each window is judged against.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from millm.api.routes.management.probes import _probe_summary


def _row(window: int):
    definition = {
        "aggregation": {"rule": "rolling_mean_max", "params": {"window": window}},
        "decision": {
            "threshold": 46.78,
            "windows": {
                "all": {"threshold": 46.78},
                "prompt": {"threshold": 28.80},
                "response": {"threshold": 46.57},
            },
            "length_bands": [
                {"min_tokens": 0, "max_tokens": 203, "threshold": 33.4},
                {"min_tokens": 204, "max_tokens": None, "threshold": 50.0},
            ],
        },
        "model": {"load_dtype": "bfloat16"},
    }
    row = MagicMock()
    for name, value in dict(
        id="pr_x", name="p", hf_id="m", layer=21, rule="rolling_mean_max", scope="all",
        basis="residual", streamable=True, threshold=46.78, target_fpr=0.01, rung=2,
        threshold_revision=1, armed=False, paused_reason=None, parity=None, created_at=None,
        definition=definition,
    ).items():
        setattr(row, name, value)
    return row


def test_the_rolling_window_tells_two_probes_apart():
    assert _probe_summary(_row(32))["rule_params"] == {"window": 32}
    assert _probe_summary(_row(64))["rule_params"] == {"window": 64}


def test_each_windows_own_bar_and_the_band_count():
    summary = _probe_summary(_row(32))
    assert summary["window_thresholds"] == {"all": 46.78, "prompt": 28.80, "response": 46.57}
    assert summary["length_band_count"] == 2


def test_a_definition_without_them_is_empty_not_invented():
    row = _row(32)
    row.definition = {"model": {}}
    summary = _probe_summary(row)
    assert summary["rule_params"] == {} and summary["window_thresholds"] == {}
    assert summary["length_band_count"] == 0
