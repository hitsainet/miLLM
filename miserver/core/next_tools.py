"""Every MCP tool an error's `next` field can name (040 FR-040.17, FR-040.35; ID-3).

miWorkbench checks each name against its LIVE tool registry (BR-071, 042 FR-042.20), reading the
published copy `docs/schemas/next-tool-hints.json` (pinned equal to this module by
`tests/unit/core/test_next_tools.py`). The values are miWorkbench tool names: `miserver_*` since P4-H
(2026-11-11, after its MCP rename went live; FR-040.35), `millm_*` before. Plain module-level assignments on purpose:
042's resolver reads module-level string constants (F-6). Every `next_tool` in `miserver/` names one.
"""

ARM_BEHAVIOUR = "miserver_arm_behaviour"
CLEAR_BEHAVIOUR_RULE_SIGNALS = "miserver_clear_behaviour_rule_signals"
DELETE_PROBE = "miserver_delete_probe"
DELETE_RULE_SET = "miserver_delete_rule_set"
DISARM_BEHAVIOUR = "miserver_disarm_behaviour"
EVALUATE_RULE_SET = "miserver_evaluate_rule_set"
GET_APPROVAL_STATUS = "miserver_get_approval_status"
GET_PROBE_LIMIT_RECOMMENDATION = "miserver_get_probe_limit_recommendation"
LIST_APPROVALS = "miserver_list_approvals"
LIST_PROBES = "miserver_list_probes"
LIST_RULE_SETS = "miserver_list_rule_sets"
LIST_RULE_VALIDATORS = "miserver_list_rule_validators"
SCORE_PROBES = "miserver_score_probes"
VALIDATE_RULE_SET = "miserver_validate_rule_set"

__all__ = [
    "ARM_BEHAVIOUR",
    "CLEAR_BEHAVIOUR_RULE_SIGNALS",
    "DELETE_PROBE",
    "DELETE_RULE_SET",
    "DISARM_BEHAVIOUR",
    "EVALUATE_RULE_SET",
    "GET_APPROVAL_STATUS",
    "GET_PROBE_LIMIT_RECOMMENDATION",
    "LIST_APPROVALS",
    "LIST_PROBES",
    "LIST_RULE_SETS",
    "LIST_RULE_VALIDATORS",
    "SCORE_PROBES",
    "VALIDATE_RULE_SET",
]

ALL: frozenset[str] = frozenset(globals()[name] for name in __all__)
