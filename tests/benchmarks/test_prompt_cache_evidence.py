from __future__ import annotations

import json
from types import SimpleNamespace

from agent_libos.tools.builtin.process import (
    CompactCompletionAcceptanceCheck,
    CompactProcessCompletionEvidence,
    CompletionAcceptanceCheck,
    ProcessCompletionEvidence,
    ProcessExitArgs,
)
from benchmarks.prompt_cache_evidence import (
    PROCESS_EXIT_CONTRACT_IDENTIFIER_PATHS,
    TERMINAL_MODEL_TOOLS,
    aggregate_model_text_leak_details,
    collect_prompt_cache_call_evidence,
    forbidden_model_text_leak_details,
    process_exit_scan_text,
    validate_prompt_cache_leak_evidence,
)

GOAL_OID = "obj_0123456789abcdef"
MESSAGE_ID = "pmsg_0123456789abcdef"
SOURCE_REF = "obj_fedcba9876543210"
RESULT_OID = "obj_00000000deadbeef"
NO_LEAKS = {
    "host_contract_fields": 0,
    "materialization_fields": 0,
    "completion_binding_fields": 0,
    "current_process_ids": 0,
    "terminal_host_identifiers": 0,
}


def _legacy_completion_evidence(**overrides: object) -> dict[str, object]:
    """The shape the legacy cumulative exit review requires the Model to echo."""

    evidence: dict[str, object] = {
        "goal_oid": GOAL_OID,
        "reviewed_message_ids": [MESSAGE_ID],
        "acceptance_checks": [
            {
                "requirement": "Fix the JPY rounding defect",
                "source_refs": [SOURCE_REF, MESSAGE_ID],
                "status": "satisfied",
                "evidence_tool_calls": ["run_tests"],
                "evidence_summary": "pytest passed after the fix",
            },
            {
                "requirement": "Keep the changelog current",
                "source_refs": [SOURCE_REF],
                "status": "satisfied",
                "evidence_tool_calls": ["read_text_file"],
                "evidence_summary": "CHANGELOG.md gained an Unreleased entry",
            },
        ],
        "final_verification": ["run_tests"],
    }
    evidence.update(overrides)
    return evidence


def _call(*tool_calls: tuple[str, object], content: str = "goal text") -> SimpleNamespace:
    return SimpleNamespace(
        messages=[{"role": "user", "content": content}],
        response_content="",
        tool_calls=[
            {
                "name": name,
                "arguments": (
                    arguments if isinstance(arguments, str) else json.dumps(arguments)
                ),
            }
            for name, arguments in tool_calls
        ],
    )


def _counts(*tool_calls: tuple[str, object]) -> dict[str, int]:
    return aggregate_model_text_leak_details(
        forbidden_model_text_leak_details([_call(*tool_calls)])
    )


def test_contract_required_identifier_echoes_are_not_leaks() -> None:
    exit_arguments = {
        "review_token": "exitrev_" + "0" * 64,
        "result_oid": RESULT_OID,
        "payload": {"summary": "fixed and verified"},
        "completion_evidence": _legacy_completion_evidence(),
    }

    assert _counts(("process_exit", exit_arguments)) == NO_LEAKS
    assert forbidden_model_text_leak_details([_call(("process_exit", exit_arguments))]) == []


def test_string_encoded_completion_evidence_binds_to_the_same_contract_fields() -> None:
    """Providers often deliver nested objects as JSON strings; the Runtime decodes them."""

    evidence = _legacy_completion_evidence(
        reviewed_message_ids=json.dumps([MESSAGE_ID]),
    )
    checks = evidence["acceptance_checks"]
    assert isinstance(checks, list)
    checks[0]["source_refs"] = json.dumps(checks[0]["source_refs"])
    exit_arguments = {
        "review_token": "exitrev_" + "0" * 64,
        "result_oid": None,
        "completion_evidence": json.dumps(evidence),
    }

    assert _counts(("process_exit", exit_arguments)) == NO_LEAKS


def test_identifiers_outside_contract_fields_stay_counted() -> None:
    evidence = _legacy_completion_evidence(final_verification=["run_tests", "evt_ffffffffffffffff"])
    checks = evidence["acceptance_checks"]
    assert isinstance(checks, list)
    checks[0]["requirement"] = "Fix goal obj_dddddddddddddddd"
    checks[0]["evidence_summary"] = "cites pmsg_eeeeeeeeeeeeeeee"
    exit_arguments = {
        "message": "Done, see obj_aaaaaaaaaaaaaaaa",
        "payload": {
            "source_refs": ["obj_bbbbbbbbbbbbbbbb"],
            "pid_cccccccccccccccc": "an identifier used as a key",
        },
        "completion_evidence": evidence,
    }

    counts = _counts(("process_exit", exit_arguments))

    assert counts["terminal_host_identifiers"] == 6
    # ``payload`` is not a contract position, so a binding-shaped field inside
    # it is still a Host-projection-shaped leak.
    assert counts["completion_binding_fields"] == 1
    assert sum(counts.values()) == 7


def test_human_output_arguments_are_scanned_in_full() -> None:
    counts = _counts(
        ("human_output", {"message": f"result {RESULT_OID} from {MESSAGE_ID}"}),
        ("process_exit", {"completion_evidence": _legacy_completion_evidence()}),
    )

    assert counts == {**NO_LEAKS, "terminal_host_identifiers": 2}


def test_unparseable_process_exit_arguments_fail_closed() -> None:
    broken = '{"completion_evidence": {"goal_oid": "' + GOAL_OID + '", trailing'

    assert process_exit_scan_text(broken) == broken
    # The raw text is scanned whole, so the contract field name next to the id
    # also trips the categorical binding pattern: unparseable output fails
    # closed on both counts.
    assert _counts(("process_exit", broken)) == {
        **NO_LEAKS,
        "terminal_host_identifiers": 1,
        "completion_binding_fields": 1,
    }
    assert _counts(("process_exit", json.dumps([GOAL_OID]))) == {
        **NO_LEAKS,
        "terminal_host_identifiers": 1,
    }


def test_compact_completion_evidence_has_no_exempt_fields() -> None:
    compact = {
        "acceptance_checks": [
            {
                "status": "satisfied",
                "evidence_tool_calls": ["run_tests"],
                "evidence_summary": f"tests passed for {GOAL_OID}",
            }
        ],
        "final_verification": ["run_tests"],
    }

    counts = _counts(("process_exit", {"completion_evidence": compact}))

    assert counts == {**NO_LEAKS, "terminal_host_identifiers": 1}
    exempt = {path[-1] for path in PROCESS_EXIT_CONTRACT_IDENTIFIER_PATHS}
    assert not exempt & set(CompactProcessCompletionEvidence.model_fields)
    assert not exempt & set(CompactCompletionAcceptanceCheck.model_fields)


def test_contract_identifier_paths_match_the_process_exit_models() -> None:
    owners = {
        (): ProcessExitArgs,
        ("completion_evidence",): ProcessCompletionEvidence,
        ("completion_evidence", "acceptance_checks", "*"): CompletionAcceptanceCheck,
    }

    for path in PROCESS_EXIT_CONTRACT_IDENTIFIER_PATHS:
        assert path[-1] in owners[path[:-1]].model_fields, path
    assert {path[-1] for path in PROCESS_EXIT_CONTRACT_IDENTIFIER_PATHS} == {
        "result_oid",
        "goal_oid",
        "reviewed_message_ids",
        "source_refs",
    }
    # Free-text fields the contract forbids identifiers in remain scanned.
    assert {"message", "payload"} <= set(ProcessExitArgs.model_fields)
    assert {"requirement", "evidence_summary"} <= set(
        CompletionAcceptanceCheck.model_fields
    )
    assert TERMINAL_MODEL_TOOLS == {"human_output", "process_exit"}


def test_legacy_review_projection_remains_a_host_side_leak() -> None:
    """The Host showing binding ids is measured; v2's compact review removes it."""

    review = {
        "status": "completion_review_required",
        "required_evidence_shape": {
            "goal_oid": GOAL_OID,
            "source_refs": [SOURCE_REF],
        },
    }
    call = _call(
        ("process_exit", {"completion_evidence": _legacy_completion_evidence()}),
        content=json.dumps(review),
    )

    details = forbidden_model_text_leak_details([call])

    assert details == [
        {
            "call_ordinal": 1,
            "categories": {"completion_binding_fields": 2},
            "surfaces": {"messages": 2},
            "response_tools": ["process_exit"],
        }
    ]


def test_collected_evidence_keeps_the_closed_schema() -> None:
    leaking = _call(("human_output", {"message": f"see {RESULT_OID}"}))
    clean = _call(("process_exit", {"completion_evidence": _legacy_completion_evidence()}))

    evidence = collect_prompt_cache_call_evidence([leaking, clean])

    assert validate_prompt_cache_leak_evidence(evidence) == {
        "forbidden_internal_id_leaks": 1,
        "forbidden_internal_id_leaks_by_category": {
            **NO_LEAKS,
            "terminal_host_identifiers": 1,
        },
        "forbidden_internal_id_leak_call_count": 1,
    }
    assert [detail["call_ordinal"] for detail in evidence["forbidden_internal_id_leak_calls"]] == [1]
