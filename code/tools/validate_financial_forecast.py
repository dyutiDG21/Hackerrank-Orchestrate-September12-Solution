#!/usr/bin/env python3
"""Focused tests and dataset diagnostics for Level 3 financial forecasting."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import sys
import unittest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import Dataset, ExchangeRate, FinancialEvent, FinancialProfile, build_indexes, load_dataset  # noqa: E402
from evidence_extraction import (  # noqa: E402
    EvidenceCache,
    EvidenceExtractionConfig,
    cache_key,
    source_from_image,
    source_from_message,
    validate_for_source,
)
from evidence_schema import EvidenceClaim  # noqa: E402
from financial_forecast import (  # noqa: E402
    HypotheticalMovement,
    ScenarioDebitAdjustment,
    build_forecast,
    build_scenario_forecast,
    forecast_diagnostics,
)
from financial_state_resolution import resolve_financial_state  # noqa: E402
from recurrence_inference import infer_recurring_series  # noqa: E402


def profile(user_id: str = "user_test", currency: str = "USD", balance: Decimal = Decimal("1000"), minimum: Decimal = Decimal("100")) -> FinancialProfile:
    return FinancialProfile(
        user_id=user_id,
        home_currency=currency,
        current_available_balance=balance,
        minimum_balance_to_keep=minimum,
        financial_priorities=(),
        expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
        raw={},
    )


def event(
    event_id: str,
    *,
    user_id: str = "user_test",
    amount: Decimal | None = Decimal("100"),
    currency: str = "USD",
    event_date: date = date(2026, 1, 1),
    settlement_date: date | None = None,
    status: str = "settled",
    direction: str = "debit",
    event_type: str = "expense",
    category: str = "rent",
    description: str = "Monthly rent",
    linked_event_id: str | None = None,
    flexibility: str = "fixed",
) -> FinancialEvent:
    cash_date = settlement_date or event_date
    return FinancialEvent(
        event_id=event_id,
        user_id=user_id,
        event_type=event_type,
        description=description,
        category=category,
        direction=direction,
        amount=amount,
        currency=currency,
        event_date=event_date,
        settlement_date=cash_date,
        status=status,
        linked_event_id=linked_event_id,
        flexibility=flexibility,
        minimum_allowed_amount=None,
        raw={},
    )


def salary_event(event_id: str, cash_date: date, *, amount: Decimal = Decimal("100"), currency: str = "USD") -> FinancialEvent:
    return event(
        event_id,
        amount=amount,
        currency=currency,
        event_date=cash_date,
        settlement_date=cash_date,
        direction="credit",
        event_type="income",
        category="salary",
        description="Payroll credit",
    )


def claim(
    claim_id: str,
    *,
    amount: Decimal | None = Decimal("150"),
    currency: str | None = "USD",
    effective_date: date | None = date(2026, 4, 1),
    claim_type: str = "salary_change",
    status: str | None = None,
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        source_type="message",
        source_id="message_test",
        user_id="user_test",
        request_id=None,
        related_event_id=None,
        claim_type=claim_type,
        document_role="net_pay" if amount is not None else None,
        amount=amount,
        currency=currency,
        effective_date=effective_date,
        status=status,
        identifier_role="payroll_reference",
        referenced_external_id="EMP-TEST",
        provenance=f"synthetic provenance {claim_id}",
        raw_claim={},
    )


def dataset_with(
    events: tuple[FinancialEvent, ...],
    *,
    profiles: tuple[FinancialProfile, ...] = (profile(),),
) -> Dataset:
    exchange_rates = (
        ExchangeRate(date(2026, 1, 1), "EUR", profiles[0].home_currency, Decimal("2"), raw={}),
    )
    indexes = build_indexes(profiles, events, (), (), (), (), (), exchange_rates, ())
    return Dataset(
        dataset_dir=Path("synthetic"),
        financial_profiles=profiles,
        financial_events=events,
        requests=(),
        sample_requests=(),
        payment_options=(),
        messages=(),
        images=(),
        exchange_rates=exchange_rates,
        output_template=(),
        indexes=indexes,
    )


def forecast_for(
    events: tuple[FinancialEvent, ...],
    *,
    claims: tuple[EvidenceClaim, ...] = (),
    start: date = date(2026, 1, 1),
    horizon_days: int = 90,
    profile_row: FinancialProfile = profile(),
):
    dataset = dataset_with(events, profiles=(profile_row,))
    state = resolve_financial_state(dataset, claims)
    recurrence = infer_recurring_series(state)
    return build_forecast(dataset, state, recurrence.series, user_id=profile_row.user_id, start_date=start, horizon_days=horizon_days)


class FinancialForecastTests(unittest.TestCase):
    def test_simple_settled_debit_credit_forecast(self) -> None:
        result = forecast_for((
            event("event_1", amount=Decimal("100"), direction="debit", event_date=date(2026, 1, 3)),
            event("event_2", amount=Decimal("50"), direction="credit", event_type="income", category="bonus", event_date=date(2026, 1, 4)),
        ))
        self.assertEqual([movement.signed_amount for movement in result.movements], [Decimal("-100"), Decimal("50")])
        self.assertEqual(result.checkpoints[-1].balance, Decimal("950"))

    def test_minimum_balance_violation_detection(self) -> None:
        result = forecast_for((event("event_1", amount=Decimal("950"), event_date=date(2026, 1, 2)),))
        self.assertTrue(result.minimum_balance_violated)
        self.assertEqual(result.first_violation_date, date(2026, 1, 2))

    def test_pending_debit_reservation(self) -> None:
        result = forecast_for((event("event_1", amount=Decimal("200"), status="pending", event_date=date(2025, 12, 31)),))
        self.assertEqual(result.movements[0].source, "pending_debit_reservation")
        self.assertEqual(result.movements[0].date, date(2026, 1, 1))
        self.assertEqual(result.checkpoints[-1].balance, Decimal("800"))

    def test_future_pending_debit_reserves_from_forecast_start_once(self) -> None:
        result = forecast_for((
            event("event_1", amount=Decimal("200"), status="pending", event_date=date(2026, 1, 20), settlement_date=date(2026, 1, 20)),
        ))
        self.assertEqual(len(result.movements), 1)
        self.assertEqual(result.movements[0].source, "pending_debit_reservation")
        self.assertEqual(result.movements[0].date, date(2026, 1, 1))
        self.assertEqual(result.checkpoints[-1].balance, Decimal("800"))

    def test_pending_credit_unavailable(self) -> None:
        result = forecast_for((
            event("event_1", amount=Decimal("200"), direction="credit", event_type="income", category="bonus", status="pending"),
        ))
        self.assertEqual(len(result.movements), 0)
        self.assertEqual(len(result.blockers), 0)
        self.assertEqual(result.notices[0].reason, "pending_credit_excluded")

    def test_failed_cancelled_unrealized_exclusion(self) -> None:
        result = forecast_for((
            event("event_1", amount=Decimal("100"), status="failed"),
            event("event_2", amount=Decimal("100"), status="cancelled"),
            event("event_3", amount=Decimal("100"), status="unrealized", direction="non_cash", event_type="investment_valuation", category="investment"),
        ))
        self.assertEqual(len(result.movements), 0)

    def test_inactive_superseded_event_exclusion(self) -> None:
        result = forecast_for((
            event("event_1", amount=Decimal("90"), status="scheduled", event_date=date(2026, 1, 3)),
            event("event_2", amount=Decimal("100"), status="scheduled", event_date=date(2026, 1, 3), linked_event_id="event_1"),
        ))
        self.assertEqual([movement.source_id for movement in result.movements], ["event_2"])
        self.assertEqual(result.movements[0].amount, Decimal("100"))

    def test_missing_amount_pending_debit_becomes_blocker_not_zero(self) -> None:
        result = forecast_for((event("event_1", amount=None, status="pending"),))
        self.assertEqual(len(result.movements), 0)
        self.assertEqual(result.blockers[0].reason, "pending_debit_missing_amount")

    def test_weekly_recurrence_projection(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 1), description="Weekly fee"),
            event("event_2", event_date=date(2026, 1, 8), description="Weekly fee"),
            event("event_3", event_date=date(2026, 1, 15), description="Weekly fee"),
        ), start=date(2026, 1, 16), horizon_days=14)
        self.assertEqual([movement.date for movement in result.movements if movement.source == "recurrence_projection"], [date(2026, 1, 22), date(2026, 1, 29)])

    def test_biweekly_recurrence_projection(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 1), description="Biweekly fee"),
            event("event_2", event_date=date(2026, 1, 15), description="Biweekly fee"),
            event("event_3", event_date=date(2026, 1, 29), description="Biweekly fee"),
        ), start=date(2026, 1, 30), horizon_days=20)
        self.assertEqual([movement.date for movement in result.movements if movement.source == "recurrence_projection"], [date(2026, 2, 12)])

    def test_month_end_monthly_recurrence(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 31)),
            event("event_2", event_date=date(2026, 2, 28)),
            event("event_3", event_date=date(2026, 3, 31)),
        ), start=date(2026, 4, 1), horizon_days=61)
        self.assertEqual([movement.date for movement in result.movements if movement.source == "recurrence_projection"], [date(2026, 4, 30), date(2026, 5, 31)])

    def test_recurrence_projection_only_inside_horizon(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 1)),
            event("event_2", event_date=date(2026, 2, 1)),
            event("event_3", event_date=date(2026, 3, 1)),
        ), start=date(2026, 3, 2), horizon_days=20)
        self.assertEqual([movement for movement in result.movements if movement.source == "recurrence_projection"], [])

    def test_salary_amendment_applies_prospectively(self) -> None:
        result = forecast_for((
            salary_event("event_1", date(2026, 1, 1), amount=Decimal("100")),
            salary_event("event_2", date(2026, 2, 1), amount=Decimal("100")),
            salary_event("event_3", date(2026, 3, 1), amount=Decimal("100")),
        ), claims=(claim("claim_1", amount=Decimal("150"), effective_date=date(2026, 4, 1)),), start=date(2026, 3, 2), horizon_days=31)
        projected = [movement for movement in result.movements if movement.source == "recurrence_projection"]
        self.assertEqual(projected[0].amount, Decimal("150"))
        self.assertIn("message_test", projected[0].provenance)

    def test_variable_recurring_income_conservative_handling(self) -> None:
        result = forecast_for((
            salary_event("event_1", date(2026, 1, 1), amount=Decimal("100")),
            salary_event("event_2", date(2026, 2, 1), amount=Decimal("125")),
            salary_event("event_3", date(2026, 3, 1), amount=Decimal("110")),
        ), start=date(2026, 3, 2), horizon_days=31)
        projected = [movement for movement in result.movements if movement.source == "recurrence_projection"]
        self.assertEqual(projected[0].amount, Decimal("100"))

    def test_variable_recurring_expense_conservative_handling(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 1), amount=Decimal("100")),
            event("event_2", event_date=date(2026, 2, 1), amount=Decimal("125")),
            event("event_3", event_date=date(2026, 3, 1), amount=Decimal("110")),
        ), start=date(2026, 3, 2), horizon_days=31)
        projected = [movement for movement in result.movements if movement.source == "recurrence_projection"]
        self.assertEqual(projected[0].amount, Decimal("125"))

    def test_explicit_scheduled_occurrence_suppresses_projection(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 1)),
            event("event_2", event_date=date(2026, 2, 1)),
            event("event_3", event_date=date(2026, 3, 1)),
            event("event_4", event_date=date(2026, 4, 1), status="scheduled"),
        ), start=date(2026, 4, 1), horizon_days=31)
        self.assertEqual(result.diagnostics.recurrence_projections_suppressed, 1)
        self.assertIn("explicit_event:event_4", [movement.movement_id for movement in result.movements])
        self.assertIn(date(2026, 5, 1), [movement.date for movement in result.movements if movement.source == "recurrence_projection"])

    def test_unrelated_explicit_event_does_not_suppress_recurrence(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 1), description="Rent"),
            event("event_2", event_date=date(2026, 2, 1), description="Rent"),
            event("event_3", event_date=date(2026, 3, 1), description="Rent"),
            event("event_4", event_date=date(2026, 4, 1), description="Different bill", status="scheduled"),
        ), start=date(2026, 4, 1), horizon_days=1)
        self.assertEqual(result.diagnostics.recurrence_projections_suppressed, 0)
        self.assertIn(date(2026, 4, 1), [movement.date for movement in result.movements if movement.source == "recurrence_projection"])

    def test_same_day_deterministic_conservative_ordering(self) -> None:
        low_profile = profile(balance=Decimal("100"), minimum=Decimal("50"))
        result = forecast_for((
            event("event_1", amount=Decimal("120"), direction="debit", event_date=date(2026, 1, 1)),
            event("event_2", amount=Decimal("100"), direction="credit", event_type="income", category="salary", event_date=date(2026, 1, 1)),
        ), profile_row=low_profile)
        self.assertEqual([movement.direction for movement in result.movements], ["debit", "credit"])
        self.assertTrue(result.minimum_balance_violated)

    def test_same_day_confirmed_salary_precedes_hypothetical_payment(self) -> None:
        low_profile = profile(balance=Decimal("100"), minimum=Decimal("50"))
        baseline = forecast_for((
            event("event_1", amount=Decimal("100"), direction="credit", event_type="income", category="salary", event_date=date(2026, 1, 1)),
        ), profile_row=low_profile)
        scenario = build_scenario_forecast(
            baseline,
            hypothetical_movements=(HypotheticalMovement("request_payment", date(2026, 1, 1), Decimal("150")),),
        )
        self.assertEqual([movement.source for movement in scenario.movements], ["explicit_event", "hypothetical"])
        self.assertFalse(scenario.minimum_balance_violated)

    def test_same_day_projected_salary_precedes_hypothetical_payment(self) -> None:
        low_profile = profile(balance=Decimal("100"), minimum=Decimal("50"))
        baseline = forecast_for((
            salary_event("event_1", date(2026, 1, 1), amount=Decimal("100")),
            salary_event("event_2", date(2026, 2, 1), amount=Decimal("100")),
            salary_event("event_3", date(2026, 3, 1), amount=Decimal("100")),
        ), start=date(2026, 4, 1), horizon_days=1, profile_row=low_profile)
        scenario = build_scenario_forecast(
            baseline,
            hypothetical_movements=(HypotheticalMovement("request_payment", date(2026, 4, 1), Decimal("150")),),
        )
        self.assertEqual([movement.source for movement in scenario.movements], ["recurrence_projection", "hypothetical"])
        self.assertFalse(scenario.minimum_balance_violated)

    def test_same_day_non_salary_credit_stays_after_hypothetical_payment(self) -> None:
        low_profile = profile(balance=Decimal("100"), minimum=Decimal("50"))
        baseline = forecast_for((
            event(
                "event_1",
                amount=Decimal("100"),
                direction="credit",
                event_type="income",
                category="bonus",
                event_date=date(2026, 1, 1),
            ),
        ), profile_row=low_profile)
        scenario = build_scenario_forecast(
            baseline,
            hypothetical_movements=(HypotheticalMovement("request_payment", date(2026, 1, 1), Decimal("100")),),
        )
        self.assertEqual([movement.source for movement in scenario.movements], ["hypothetical", "explicit_event"])
        self.assertTrue(scenario.minimum_balance_violated)

    def test_hypothetical_scenario_debit(self) -> None:
        baseline = forecast_for(())
        scenario = build_scenario_forecast(
            baseline,
            hypothetical_movements=(HypotheticalMovement("request_payment", date(2026, 1, 1), Decimal("950")),),
        )
        self.assertTrue(scenario.minimum_balance_violated)
        self.assertEqual(scenario.diagnostics.hypothetical_movements_included, 1)

    def test_scenario_input_does_not_mutate_baseline(self) -> None:
        baseline = forecast_for(())
        original = baseline
        _ = build_scenario_forecast(
            baseline,
            hypothetical_movements=(HypotheticalMovement("request_payment", date(2026, 1, 1), Decimal("100")),),
        )
        self.assertEqual(baseline, original)
        self.assertEqual(len(baseline.movements), 0)

    def test_stop_adjustment_suppresses_only_matching_projected_debits(self) -> None:
        baseline = forecast_for((
            event("rent_1", event_date=date(2026, 1, 1), description="Flexible rent", flexibility="stoppable"),
            event("rent_2", event_date=date(2026, 2, 1), description="Flexible rent", flexibility="stoppable"),
            event("rent_3", event_date=date(2026, 3, 1), description="Flexible rent", flexibility="stoppable"),
            event("one_off", event_date=date(2026, 4, 2), description="Unrelated"),
        ), start=date(2026, 4, 1), horizon_days=31)
        original = baseline
        scenario = build_scenario_forecast(
            baseline,
            hypothetical_movements=(),
            scenario_adjustments=(ScenarioDebitAdjustment("stop", "rent_3", provenance=("rent_3",)),),
        )
        self.assertEqual(baseline, original)
        self.assertEqual([movement.source_id for movement in scenario.movements], ["one_off"])
        self.assertEqual(len(scenario.scenario_adjustments[0].affected_movement_ids), 2)

    def test_reduce_adjustment_replaces_projected_amount_once_and_keeps_history(self) -> None:
        baseline = forecast_for((
            event("fee_1", amount=Decimal("100"), event_date=date(2026, 1, 1), description="Flexible fee", flexibility="reducible"),
            event("fee_2", amount=Decimal("100"), event_date=date(2026, 2, 1), description="Flexible fee", flexibility="reducible"),
            event("fee_3", amount=Decimal("100"), event_date=date(2026, 3, 1), description="Flexible fee", flexibility="reducible"),
            event("one_off", amount=Decimal("25"), event_date=date(2026, 4, 2), description="Unrelated"),
        ), start=date(2026, 4, 1), horizon_days=31)
        scenario = build_scenario_forecast(
            baseline,
            hypothetical_movements=(),
            scenario_adjustments=(ScenarioDebitAdjustment("reduce_to", "fee_3", Decimal("40")),),
        )
        projected = [movement for movement in scenario.movements if movement.source == "recurrence_projection"]
        explicit = [movement for movement in scenario.movements if movement.source == "explicit_event"]
        self.assertEqual([movement.amount for movement in projected], [Decimal("40"), Decimal("40")])
        self.assertEqual(explicit[0].amount, Decimal("25"))
        self.assertTrue(all(movement.provenance.count("scenario_adjustment:reduce_to:fee_3") == 1 for movement in projected))

    def test_invalid_adjustment_combinations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_scenario_forecast(
                forecast_for(()),
                hypothetical_movements=(),
                scenario_adjustments=(
                    ScenarioDebitAdjustment("stop", "event_1"),
                    ScenarioDebitAdjustment("reduce_to", "event_1", Decimal("1")),
                ),
            )

    def test_decimal_preservation(self) -> None:
        result = forecast_for((event("event_1", amount=Decimal("0.10"), event_date=date(2026, 1, 1)),))
        self.assertIsInstance(result.movements[0].amount, Decimal)
        self.assertEqual(result.checkpoints[-1].balance, Decimal("999.90"))

    def test_deterministic_idempotent_forecast(self) -> None:
        events = (
            event("event_2", event_date=date(2026, 2, 1)),
            event("event_1", event_date=date(2026, 1, 1)),
            event("event_3", event_date=date(2026, 3, 1)),
        )
        self.assertEqual(forecast_for(events, start=date(2026, 3, 2)), forecast_for(events, start=date(2026, 3, 2)))

    def test_provenance_for_explicit_and_projected_movements(self) -> None:
        result = forecast_for((
            event("event_1", event_date=date(2026, 1, 1)),
            event("event_2", event_date=date(2026, 2, 1)),
            event("event_3", event_date=date(2026, 3, 1)),
            event("event_4", event_date=date(2026, 3, 5), description="One-off"),
        ), start=date(2026, 3, 2), horizon_days=31)
        explicit = [movement for movement in result.movements if movement.source == "explicit_event"][0]
        projected = [movement for movement in result.movements if movement.source == "recurrence_projection"][0]
        self.assertEqual(explicit.provenance, ("event_4",))
        self.assertTrue(projected.provenance[0].startswith("series_"))


def load_cached_claims(dataset) -> tuple[EvidenceClaim, ...]:  # noqa: ANN001
    config = EvidenceExtractionConfig.from_env()
    cache = EvidenceCache(config.cache_dir)
    claims: list[EvidenceClaim] = []
    for message in dataset.messages:
        source = source_from_message(message)
        payload = cache.get(cache_key(source, model=config.model))
        if payload is not None:
            claims.extend(validate_for_source(payload, source).claims)
    for image in dataset.images:
        source = source_from_image(image)
        payload = cache.get(cache_key(source, model=config.model))
        if payload is not None:
            claims.extend(validate_for_source(payload, source).claims)
    return tuple(claims)


def print_dataset_diagnostics() -> None:
    dataset = load_dataset()
    claims = load_cached_claims(dataset)
    state = resolve_financial_state(dataset, claims)
    recurrence = infer_recurring_series(state)
    representative = tuple(dataset.requests)
    forecasts = tuple(
        build_forecast(dataset, state, recurrence.series, user_id=request.user_id, start_date=request.request_date)
        for request in representative
    )
    diagnostics = forecast_diagnostics(forecasts)
    print("\nDataset forecast diagnostics:")
    print(f"- requests_audited={len(forecasts)}")
    print(f"- explicit_movements_included={diagnostics['explicit_movements_included']}")
    print(f"- recurrence_projections_generated={diagnostics['recurrence_projections_generated']}")
    print(f"- recurrence_projections_suppressed={diagnostics['recurrence_projections_suppressed']}")
    print(f"- unresolved_blockers_by_reason={diagnostics['unresolved_blockers_by_reason']}")
    print(f"- notices_by_reason={diagnostics['notices_by_reason']}")
    print(f"- minimum_balance_violations={diagnostics['minimum_balance_violations']}")
    print(f"- suspicious_duplicate_movements={diagnostics['suspicious_duplicate_movements']}")

    selected: list[tuple[str, str, str, object]] = []
    for request, forecast in zip(representative, forecasts):
        if any(movement.category == "salary" and movement.source == "recurrence_projection" for movement in forecast.movements):
            selected.append(("salary_recurrence", request.request_id, request.user_id, forecast))
            break
    for request, forecast in zip(representative, forecasts):
        if any(
            movement.source == "recurrence_projection" and movement.direction == "debit" and movement.category in {"groceries", "transport", "dining", "utilities"}
            for movement in forecast.movements
        ):
            selected.append(("variable_recurring_expense", request.request_id, request.user_id, forecast))
            break
    for request, forecast in zip(representative, forecasts):
        if any(notice.reason == "pending_credit_excluded" for notice in forecast.notices):
            selected.append(("pending_credit_notice", request.request_id, request.user_id, forecast))
            break
    for request, forecast in zip(representative, forecasts):
        if any(blocker.reason == "pending_debit_missing_amount" for blocker in forecast.blockers):
            selected.append(("pending_debit_missing_amount", request.request_id, request.user_id, forecast))
            break
    month_end_series_ids = {
        item.series_id for item in recurrence.series if item.projectable and item.cadence_evidence.end_of_month_anchor
    }
    for request, forecast in zip(representative, forecasts):
        if any(movement.source == "recurrence_projection" and movement.provenance and movement.provenance[0] in month_end_series_ids for movement in forecast.movements):
            selected.append(("month_end_recurrence", request.request_id, request.user_id, forecast))
            break
    if not any(item[0] == "month_end_recurrence" for item in selected) and month_end_series_ids:
        month_end_series = next(item for item in recurrence.series if item.series_id in month_end_series_ids)
        start_date = month_end_series.cadence_evidence.observed_dates[-1] + timedelta(days=1)
        forecast = build_forecast(dataset, state, recurrence.series, user_id=month_end_series.grouping_key.user_id, start_date=start_date)
        selected.append(("month_end_recurrence", "diagnostic_forecast", month_end_series.grouping_key.user_id, forecast))
    for request, forecast in zip(tuple(dataset.sample_requests), (
        build_forecast(dataset, state, recurrence.series, user_id=sample.request.user_id, start_date=sample.request.request_date)
        for sample in dataset.sample_requests
    )):
        if any("message_01" in movement.provenance or "message_02" in movement.provenance for movement in forecast.movements):
            selected.append(("prospective_payroll_amendment", request.request.request_id, request.request.user_id, forecast))
            break

    print("- representative_forecasts:")
    for label, request_id, user_id, forecast in selected[:8]:
        print(
            f"  - kind={label}; request_id={request_id}; user_id={user_id}; explicit={forecast.diagnostics.explicit_movements_included}; "
            f"projected={forecast.diagnostics.recurrence_projections_generated}; suppressed={forecast.diagnostics.recurrence_projections_suppressed}; "
            f"blockers={[blocker.reason for blocker in forecast.blockers]}; notices={[notice.reason for notice in forecast.notices]}; "
            f"min_balance={forecast.minimum_observed_balance}"
        )


if __name__ == "__main__":
    test_result = unittest.main(verbosity=2, exit=False)
    if test_result.result.wasSuccessful():
        print_dataset_diagnostics()
    raise SystemExit(0 if test_result.result.wasSuccessful() else 1)
