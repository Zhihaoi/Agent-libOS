from __future__ import annotations

from copy import deepcopy

import pytest

from agent_libos.llm.prompt_cache_gate import (
    PromptCachePricing,
    calculate_prompt_cache_cost,
    evaluate_prompt_cache_release_gate,
)


def _legacy_report() -> dict[str, object]:
    return {
        "repetitions": 3,
        "providers": _providers(candidate=False),
        "metrics": {
            "runs": 3,
            "successful_runs": 3,
            "success_rate": 1.0,
            "uncached_input_tokens": 1_000,
            "total_input_tokens": 2_000,
            "cache_read_tokens": 1_000,
            "cache_total_calls": 3,
            "cache_read_reported_calls": 3,
            "cache_metric_reported_calls": 3,
            "cache_metric_input_tokens": 2_000,
            "cache_hit_rate": 0.50,
        },
    }


def _candidate_report() -> dict[str, object]:
    return {
        "repetitions": 3,
        "providers": _providers(candidate=True),
        "pricing_known": True,
        "cost": {
            "net_cost": 4.25,
            "net_cost_per_successful_task": 4.25 / 3,
        },
        "metrics": {
            "runs": 3,
            "successful_runs": 3,
            "success_rate": 1.0,
            "uncached_input_tokens": 750,
            "total_input_tokens": 1_700,
            "cache_read_tokens": 950,
            "cache_total_calls": 3,
            "cache_read_reported_calls": 3,
            "cache_metric_reported_calls": 3,
            "cache_metric_input_tokens": 1_700,
            "cache_hit_rate": 950 / 1_700,
            "forbidden_internal_id_leaks": 0,
        },
        "release_gates": {
            "all_oracles_passed": True,
            "completion_evidence_passed": True,
            "security_invariants_passed": True,
            "workflow_count": 6,
        },
    }


def _providers(*, candidate: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for provider_id, model_id, pricing_known in (
        ("custom-endpoint", "qwen-test", False),
        ("openai", "gpt-test", True),
    ):
        row: dict[str, object] = {
            "provider_id": provider_id,
            "model_id": model_id,
            "repetitions": 3,
            "workflow_count": 12,
            "all_oracles_passed": True,
            "completion_evidence_passed": True,
            "forbidden_internal_id_leaks": 0,
            "pricing_known": pricing_known,
        }
        if pricing_known:
            row["cost"] = {
                "net_cost": 2.0 if candidate else 3.0,
                "net_cost_per_successful_task": (
                    2.0 / 12 if candidate else 3.0 / 12
                ),
            }
        rows.append(row)
    return rows


def test_prompt_cache_release_gate_accepts_complete_paired_evidence() -> None:
    result = evaluate_prompt_cache_release_gate(
        _legacy_report(),
        _candidate_report(),
    )

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["metrics"]["uncached_input_reduction"] == pytest.approx(0.25)
    assert result["metrics"]["total_input_reduction"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("path", "value", "failed_check"),
    [
        (("metrics", "uncached_input_tokens"), 801, "uncached_input_reduction"),
        (("metrics", "total_input_tokens"), 1_801, "total_input_reduction"),
        (("metrics", "cache_hit_rate"), 0.49, "cache_hit_rate_not_lower"),
        (("metrics", "success_rate"), 0.9, "candidate_success_rate"),
        (("metrics", "forbidden_internal_id_leaks"), 1, "forbidden_internal_id_leaks"),
        (("release_gates", "all_oracles_passed"), False, "all_oracles_passed"),
    ],
)
def test_prompt_cache_release_gate_fails_each_required_threshold(
    path: tuple[str, str],
    value: object,
    failed_check: str,
) -> None:
    candidate = deepcopy(_candidate_report())
    section = candidate[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value

    result = evaluate_prompt_cache_release_gate(_legacy_report(), candidate)

    assert result["passed"] is False
    assert result["checks"][failed_check] is False


def test_prompt_cache_canary_gate_relaxes_only_release_evidence() -> None:
    candidate = _candidate_report()
    candidate.pop("release_gates")
    candidate["pricing_known"] = False
    candidate.pop("cost")

    result = evaluate_prompt_cache_release_gate(
        _legacy_report(),
        candidate,
        strict_release_evidence=False,
    )

    assert result["passed"] is True
    assert "all_oracles_passed" not in result["checks"]


def test_prompt_cache_gate_fails_when_candidate_omits_cache_counters() -> None:
    candidate = _candidate_report()
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    metrics["cache_total_calls"] = 6

    result = evaluate_prompt_cache_release_gate(
        _legacy_report(),
        candidate,
        strict_release_evidence=False,
    )

    assert result["passed"] is False
    assert result["checks"]["candidate_cache_telemetry_complete"] is False


def test_known_price_requires_net_and_per_success_costs() -> None:
    candidate = _candidate_report()
    candidate["cost"] = {"net_cost": 4.25}

    result = evaluate_prompt_cache_release_gate(_legacy_report(), candidate)

    assert result["passed"] is False
    assert result["checks"]["known_price_cost_accounted"] is False


def test_strict_gate_requires_paired_multi_provider_evidence() -> None:
    candidate = _candidate_report()
    providers = candidate["providers"]
    assert isinstance(providers, list)
    providers.pop()

    result = evaluate_prompt_cache_release_gate(_legacy_report(), candidate)

    assert result["passed"] is False
    assert result["checks"]["provider_coverage"] is False
    assert result["checks"]["paired_provider_coverage"] is False


def test_known_provider_price_requires_per_success_cost() -> None:
    candidate = _candidate_report()
    providers = candidate["providers"]
    assert isinstance(providers, list)
    official = providers[1]
    assert isinstance(official, dict)
    official["cost"] = {"net_cost": 2.0}

    result = evaluate_prompt_cache_release_gate(_legacy_report(), candidate)

    assert result["passed"] is False
    assert result["checks"]["known_provider_prices_accounted"] is False


def test_known_provider_cost_must_not_regress_per_successful_task() -> None:
    candidate = _candidate_report()
    providers = candidate["providers"]
    assert isinstance(providers, list)
    official = providers[1]
    assert isinstance(official, dict)
    official["cost"] = {
        "net_cost": 4.0,
        "net_cost_per_successful_task": 4.0 / 12,
    }

    result = evaluate_prompt_cache_release_gate(_legacy_report(), candidate)

    assert result["passed"] is False
    assert result["checks"]["known_provider_cost_not_higher"] is False


def test_prompt_cache_gate_rejects_missing_token_metrics() -> None:
    candidate = _candidate_report()
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    metrics.pop("uncached_input_tokens")

    with pytest.raises(ValueError, match="uncached_input_tokens"):
        evaluate_prompt_cache_release_gate(_legacy_report(), candidate)


def test_prompt_cache_gate_rejects_missing_leak_measurement() -> None:
    candidate = _candidate_report()
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    metrics.pop("forbidden_internal_id_leaks")

    with pytest.raises(ValueError, match="forbidden_internal_id_leaks"):
        evaluate_prompt_cache_release_gate(
            _legacy_report(),
            candidate,
            strict_release_evidence=False,
        )


def test_prompt_cache_cost_separates_read_write_and_output_rates() -> None:
    cost = calculate_prompt_cache_cost(
        {
            "total_input_tokens": 1_000_000,
            "cache_read_tokens": 600_000,
            "uncached_input_tokens": 400_000,
            "cache_write_tokens": 100_000,
            "total_output_tokens": 50_000,
        },
        successful_tasks=10,
        pricing=PromptCachePricing(
            input_per_million=5.0,
            cached_input_per_million=0.5,
            cache_write_input_per_million=6.25,
            output_per_million=30.0,
        ),
    )

    assert cost["input_cost"] == pytest.approx(1.5)
    assert cost["cache_read_cost"] == pytest.approx(0.3)
    assert cost["cache_write_cost"] == pytest.approx(0.625)
    assert cost["output_cost"] == pytest.approx(1.5)
    assert cost["net_cost"] == pytest.approx(3.925)
    assert cost["net_cost_per_successful_task"] == pytest.approx(0.3925)


def test_prompt_cache_cost_does_not_treat_unknown_write_tokens_as_zero() -> None:
    with pytest.raises(ValueError, match="cache_write_tokens must be reported"):
        calculate_prompt_cache_cost(
            {
                "total_input_tokens": 1_000,
                "cache_read_tokens": 500,
                "uncached_input_tokens": 500,
                "cache_write_tokens": None,
                "total_output_tokens": 100,
            },
            successful_tasks=1,
            pricing=PromptCachePricing(
                input_per_million=5.0,
                cached_input_per_million=0.5,
                cache_write_input_per_million=6.25,
                output_per_million=30.0,
            ),
        )


def _leak_categories(*, host_projection: int, terminal: int) -> dict[str, int]:
    return {
        "host_contract_fields": 0,
        "materialization_fields": 0,
        "completion_binding_fields": host_projection,
        "current_process_ids": 0,
        "terminal_host_identifiers": terminal,
    }


def _same_layout_canary_arms(
    *,
    legacy_host_projection: int = 3,
    candidate_host_projection: int = 3,
    candidate_terminal: int = 0,
) -> tuple[dict[str, object], dict[str, object]]:
    legacy = _legacy_report()
    legacy["prompt_layout"] = "legacy_v1"
    legacy_metrics = legacy["metrics"]
    assert isinstance(legacy_metrics, dict)
    legacy_metrics["forbidden_internal_id_leaks"] = legacy_host_projection
    legacy_metrics["forbidden_internal_id_leaks_by_category"] = _leak_categories(
        host_projection=legacy_host_projection, terminal=0
    )
    candidate = _candidate_report()
    candidate.pop("release_gates")
    candidate["pricing_known"] = False
    candidate.pop("cost")
    candidate["prompt_layout"] = "legacy_v1"
    candidate_metrics = candidate["metrics"]
    assert isinstance(candidate_metrics, dict)
    candidate_metrics["forbidden_internal_id_leaks"] = (
        candidate_host_projection + candidate_terminal
    )
    candidate_metrics["forbidden_internal_id_leaks_by_category"] = _leak_categories(
        host_projection=candidate_host_projection, terminal=candidate_terminal
    )
    return legacy, candidate


def test_prompt_cache_gate_reports_leak_breakdown_metrics() -> None:
    candidate = _candidate_report()
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    metrics["forbidden_internal_id_leaks_by_category"] = _leak_categories(
        host_projection=0, terminal=0
    )

    result = evaluate_prompt_cache_release_gate(_legacy_report(), candidate)

    assert result["passed"] is True
    assert result["metrics"]["legacy_forbidden_internal_id_leaks"] is None
    assert result["metrics"]["legacy_forbidden_internal_id_leaks_by_category"] is None
    assert result["metrics"]["candidate_forbidden_internal_id_leaks"] == 0
    assert result["metrics"][
        "candidate_forbidden_internal_id_leaks_by_category"
    ] == _leak_categories(host_projection=0, terminal=0)


def test_same_layout_canary_inherits_host_projection_but_not_model_identifiers() -> None:
    """Two legacy_v1 arms share the layout's review projection; only the change is judged."""

    legacy, candidate = _same_layout_canary_arms()

    result = evaluate_prompt_cache_release_gate(
        legacy,
        candidate,
        strict_release_evidence=False,
    )

    assert result["passed"] is True
    assert result["checks"]["forbidden_internal_id_leaks"] is True
    assert result["metrics"]["legacy_forbidden_internal_id_leaks"] == 3
    assert result["metrics"]["candidate_forbidden_internal_id_leaks"] == 3


@pytest.mark.parametrize(
    ("candidate_host_projection", "candidate_terminal"),
    [(4, 0), (3, 1), (0, 1)],
)
def test_same_layout_canary_rejects_leak_regressions(
    candidate_host_projection: int,
    candidate_terminal: int,
) -> None:
    legacy, candidate = _same_layout_canary_arms(
        candidate_host_projection=candidate_host_projection,
        candidate_terminal=candidate_terminal,
    )

    result = evaluate_prompt_cache_release_gate(
        legacy,
        candidate,
        strict_release_evidence=False,
    )

    assert result["passed"] is False
    assert result["checks"]["forbidden_internal_id_leaks"] is False


def test_strict_gate_requires_zero_identifiers_even_for_same_layout_arms() -> None:
    legacy, candidate = _same_layout_canary_arms()
    candidate["release_gates"] = _candidate_report()["release_gates"]

    result = evaluate_prompt_cache_release_gate(legacy, candidate)

    assert result["passed"] is False
    assert result["checks"]["forbidden_internal_id_leaks"] is False


def test_canary_across_layouts_requires_zero_identifiers() -> None:
    legacy, candidate = _same_layout_canary_arms()
    candidate["prompt_layout"] = "cache_optimized_v2"

    result = evaluate_prompt_cache_release_gate(
        legacy,
        candidate,
        strict_release_evidence=False,
    )

    assert result["checks"]["forbidden_internal_id_leaks"] is False


def test_canary_without_legacy_leak_measurement_requires_zero_identifiers() -> None:
    legacy, candidate = _same_layout_canary_arms()
    legacy_metrics = legacy["metrics"]
    assert isinstance(legacy_metrics, dict)
    legacy_metrics.pop("forbidden_internal_id_leaks")
    legacy_metrics.pop("forbidden_internal_id_leaks_by_category")

    result = evaluate_prompt_cache_release_gate(
        legacy,
        candidate,
        strict_release_evidence=False,
    )

    assert result["checks"]["forbidden_internal_id_leaks"] is False
    assert result["metrics"]["legacy_forbidden_internal_id_leaks"] is None


def test_prompt_cache_gate_rejects_unreconciled_leak_categories() -> None:
    candidate = _candidate_report()
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    metrics["forbidden_internal_id_leaks_by_category"] = _leak_categories(
        host_projection=1, terminal=0
    )

    with pytest.raises(ValueError, match="reconcile"):
        evaluate_prompt_cache_release_gate(_legacy_report(), candidate)


def test_prompt_cache_gate_rejects_fractional_leak_counts() -> None:
    candidate = _candidate_report()
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    metrics["forbidden_internal_id_leaks"] = 0.5

    with pytest.raises(ValueError, match="non-negative integer"):
        evaluate_prompt_cache_release_gate(_legacy_report(), candidate)
