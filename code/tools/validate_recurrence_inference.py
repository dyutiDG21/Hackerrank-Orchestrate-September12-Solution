#!/usr/bin/env python3
"""Focused tests and dataset diagnostics for Level 2C recurrence inference."""

from __future__ import annotations

from datetime import date
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
from financial_state_resolution import resolve_financial_state  # noqa: E402
from recurrence_inference import infer_recurring_series, recurrence_diagnostics  # noqa: E402


def profile(user_id: str = "user_test", currency: str = "USD") -> FinancialProfile:
    return FinancialProfile(
        user_id=user_id,
        home_currency=currency,
        current_available_balance=Decimal("10000"),
        minimum_balance_to_keep=Decimal("1000"),
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
    amount: Decimal = Decimal("100"),
    currency: str = "USD",
    event_date: date = date(2026, 1, 1),
    settlement_date: date | None = None,
    status: str = "settled",
    direction: str = "debit",
    event_type: str = "expense",
    category: str = "rent",
    description: str = "Monthly rent",
    linked_event_id: str | None = None,
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
        flexibility="fixed",
        minimum_allowed_amount=None,
        raw={},
    )


def claim(
    claim_id: str,
    *,
    user_id: str = "user_test",
    claim_type: str = "salary_change",
    amount: Decimal | None = Decimal("200"),
    currency: str | None = "USD",
    effective_date: date | None = date(2026, 4, 1),
    document_role: str | None = "net_pay",
    identifier_role: str | None = "payroll_reference",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        source_type="message",
        source_id="message_test",
        user_id=user_id,
        request_id=None,
        related_event_id=None,
        claim_type=claim_type,
        document_role=document_role,
        amount=amount,
        currency=currency,
        effective_date=effective_date,
        status=None,
        identifier_role=identifier_role,
        referenced_external_id="EMP-TEST",
        provenance=f"synthetic provenance {claim_id}",
        raw_claim={},
    )


def dataset_with(events: tuple[FinancialEvent, ...], *, users: tuple[str, ...] = ("user_test",), currency: str = "USD") -> Dataset:
    profiles = tuple(profile(user_id=user_id, currency=currency) for user_id in users)
    exchange_rates = (ExchangeRate(date(2026, 1, 1), "EUR", currency, Decimal("2"), raw={}),)
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


def infer_for(events: tuple[FinancialEvent, ...], claims: tuple[EvidenceClaim, ...] = ()):
    state = resolve_financial_state(dataset_with(events), claims)
    return infer_recurring_series(state)


def salary_event(event_id: str, cash_date: date, *, amount: Decimal = Decimal("100"), currency: str = "USD", description: str = "Payroll credit") -> FinancialEvent:
    return event(
        event_id,
        amount=amount,
        currency=currency,
        event_date=cash_date,
        settlement_date=cash_date,
        direction="credit",
        event_type="income",
        category="salary",
        description=description,
    )


class RecurrenceInferenceTests(unittest.TestCase):
    def test_clear_monthly_recurrence(self) -> None:
        result = infer_for((
            event("event_1", event_date=date(2026, 1, 5)),
            event("event_2", event_date=date(2026, 2, 5)),
            event("event_3", event_date=date(2026, 3, 5)),
        ))
        self.assertEqual(len(result.series), 1)
        self.assertEqual(result.series[0].cadence_evidence.cadence, "monthly")
        self.assertTrue(result.series[0].projectable)

    def test_month_end_recurrence_across_unequal_month_lengths(self) -> None:
        result = infer_for((
            event("event_1", event_date=date(2026, 1, 31)),
            event("event_2", event_date=date(2026, 2, 28)),
            event("event_3", event_date=date(2026, 3, 31)),
        ))
        series = result.series[0]
        self.assertEqual(series.cadence_evidence.cadence, "monthly")
        self.assertTrue(series.cadence_evidence.end_of_month_anchor)

    def test_weekly_and_biweekly_recurrence(self) -> None:
        weekly = infer_for((
            event("event_1", event_date=date(2026, 1, 1), description="Weekly therapy"),
            event("event_2", event_date=date(2026, 1, 8), description="Weekly therapy"),
            event("event_3", event_date=date(2026, 1, 15), description="Weekly therapy"),
        ))
        biweekly = infer_for((
            event("event_4", event_date=date(2026, 1, 1), description="Biweekly cleaner"),
            event("event_5", event_date=date(2026, 1, 15), description="Biweekly cleaner"),
            event("event_6", event_date=date(2026, 1, 29), description="Biweekly cleaner"),
        ))
        self.assertEqual(weekly.series[0].cadence_evidence.cadence, "weekly")
        self.assertEqual(biweekly.series[0].cadence_evidence.cadence, "biweekly")

    def test_irregular_events_are_not_projectable(self) -> None:
        result = infer_for((
            event("event_1", event_date=date(2026, 1, 3), description="Taxi"),
            event("event_2", event_date=date(2026, 1, 9), description="Taxi"),
            event("event_3", event_date=date(2026, 2, 20), description="Taxi"),
        ))
        self.assertEqual(result.series[0].cadence_evidence.cadence, None)
        self.assertFalse(result.series[0].projectable)

    def test_same_category_distinct_descriptions_remain_separate(self) -> None:
        result = infer_for((
            event("event_1", event_date=date(2026, 1, 1), category="utilities", description="Water bill"),
            event("event_2", event_date=date(2026, 2, 1), category="utilities", description="Water bill"),
            event("event_3", event_date=date(2026, 3, 1), category="utilities", description="Water bill"),
            event("event_4", event_date=date(2026, 1, 1), category="utilities", description="Power bill"),
            event("event_5", event_date=date(2026, 2, 1), category="utilities", description="Power bill"),
            event("event_6", event_date=date(2026, 3, 1), category="utilities", description="Power bill"),
        ))
        self.assertEqual(len(result.series), 2)
        self.assertEqual({series.grouping_key.description_key for series in result.series}, {"power bill", "water bill"})

    def test_fixed_vs_variable_amounts(self) -> None:
        fixed = infer_for((
            event("event_1", event_date=date(2026, 1, 1), amount=Decimal("10")),
            event("event_2", event_date=date(2026, 2, 1), amount=Decimal("10")),
            event("event_3", event_date=date(2026, 3, 1), amount=Decimal("10")),
        ))
        variable = infer_for((
            event("event_4", event_date=date(2026, 1, 1), amount=Decimal("10"), description="Utility"),
            event("event_5", event_date=date(2026, 2, 1), amount=Decimal("12"), description="Utility"),
            event("event_6", event_date=date(2026, 3, 1), amount=Decimal("11"), description="Utility"),
        ))
        self.assertEqual(fixed.series[0].fixed_or_variable, "fixed")
        self.assertEqual(variable.series[0].fixed_or_variable, "variable")

    def test_inactive_superseded_event_does_not_seed_recurrence_independently(self) -> None:
        result = infer_for((
            event("event_1", event_date=date(2026, 1, 1), amount=Decimal("90")),
            event("event_2", event_date=date(2026, 1, 1), amount=Decimal("100"), linked_event_id="event_1"),
            event("event_3", event_date=date(2026, 2, 1), amount=Decimal("100")),
            event("event_4", event_date=date(2026, 3, 1), amount=Decimal("100")),
        ))
        self.assertEqual(result.series[0].member_event_ids, ("event_2", "event_3", "event_4"))

    def test_explicit_future_event_coexists_without_generated_forecast(self) -> None:
        result = infer_for((
            event("event_1", event_date=date(2026, 1, 1)),
            event("event_2", event_date=date(2026, 2, 1)),
            event("event_3", event_date=date(2026, 3, 1)),
            event("event_4", event_date=date(2026, 4, 1), status="scheduled"),
        ))
        occurrences = result.series[0].explicit_occurrences
        self.assertEqual([item.event_id for item in occurrences], ["event_1", "event_2", "event_3", "event_4"])
        self.assertEqual(occurrences[-1].status, "scheduled")

    def test_unlinked_salary_change_uniquely_matches_salary_series(self) -> None:
        result = infer_for((
            salary_event("event_1", date(2026, 1, 15)),
            salary_event("event_2", date(2026, 2, 15)),
            salary_event("event_3", date(2026, 3, 15)),
        ), (claim("claim_1", amount=Decimal("120"), effective_date=date(2026, 4, 15)),))
        self.assertEqual(len(result.unresolved_evidence), 0)
        self.assertEqual(result.series[0].evidence_amendments[0].amount, Decimal("120"))

    def test_salary_change_is_prospective_and_does_not_rewrite_history(self) -> None:
        result = infer_for((
            salary_event("event_1", date(2026, 1, 15), amount=Decimal("100")),
            salary_event("event_2", date(2026, 2, 15), amount=Decimal("100")),
            salary_event("event_3", date(2026, 3, 15), amount=Decimal("100")),
        ), (claim("claim_1", amount=Decimal("150"), effective_date=date(2026, 4, 15)),))
        observations = result.series[0].amount_model.observed_amounts
        self.assertEqual({item.amount for item in observations}, {Decimal("100")})
        self.assertEqual(result.series[0].evidence_amendments[0].effective_date, date(2026, 4, 15))

    def test_ambiguous_payroll_evidence_remains_unresolved(self) -> None:
        result = infer_for((
            salary_event("event_1", date(2026, 1, 15), description="Payroll credit"),
            salary_event("event_2", date(2026, 2, 15), description="Payroll credit"),
            salary_event("event_3", date(2026, 3, 15), description="Payroll credit"),
            salary_event("event_4", date(2026, 1, 20), description="Second job payroll"),
            salary_event("event_5", date(2026, 2, 20), description="Second job payroll"),
            salary_event("event_6", date(2026, 3, 20), description="Second job payroll"),
        ), (claim("claim_1"),))
        self.assertEqual(result.unresolved_evidence[0].reason, "ambiguous_recurring_series_match")

    def test_currency_incompatibility_prevents_matching(self) -> None:
        result = infer_for((
            salary_event("event_1", date(2026, 1, 15), currency="USD"),
            salary_event("event_2", date(2026, 2, 15), currency="USD"),
            salary_event("event_3", date(2026, 3, 15), currency="USD"),
        ), (claim("claim_1", currency="EUR"),))
        self.assertEqual(result.unresolved_evidence[0].reason, "currency_incompatible_with_candidate_series")

    def test_generic_unlinked_claim_without_payroll_semantics_does_not_match_salary(self) -> None:
        generic = claim(
            "claim_1",
            claim_type="future_payment_clarification",
            amount=Decimal("120"),
            currency="USD",
            document_role=None,
            identifier_role=None,
        )
        result = infer_for((
            salary_event("event_1", date(2026, 1, 15)),
            salary_event("event_2", date(2026, 2, 15)),
            salary_event("event_3", date(2026, 3, 15)),
        ), (generic,))
        self.assertEqual(result.unresolved_evidence[0].reason, "no_semantically_compatible_recurring_series")

    def test_deterministic_idempotent_inference(self) -> None:
        events = (
            event("event_2", event_date=date(2026, 2, 1)),
            event("event_1", event_date=date(2026, 1, 1)),
            event("event_3", event_date=date(2026, 3, 1)),
        )
        self.assertEqual(infer_for(events), infer_for(events))

    def test_provenance_membership_and_evidence_amendment(self) -> None:
        result = infer_for((
            salary_event("event_1", date(2026, 1, 15)),
            salary_event("event_2", date(2026, 2, 15)),
            salary_event("event_3", date(2026, 3, 15)),
        ), (claim("claim_1"),))
        series = result.series[0]
        self.assertEqual(series.provenance_event_ids, ("event_1", "event_2", "event_3"))
        self.assertEqual(series.evidence_amendments[0].claim.claim_id, "claim_1")


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
    result = infer_recurring_series(state)
    diagnostics = recurrence_diagnostics(result)
    print("\nDataset recurrence diagnostics:")
    print(f"- cached_evidence_claims_loaded={len(claims)}")
    print(f"- recurrence_series_inferred={diagnostics['series_count']}")
    print(f"- cadence_counts={diagnostics['cadence_counts']}")
    print(f"- projectable_counts={diagnostics['projectable_counts']}")
    print(f"- series_length_distribution={diagnostics['series_length_distribution']}")
    print(f"- unresolved_recurring_evidence_counts={diagnostics['unresolved_evidence_counts']}")
    print("- suspicious_or_low_confidence_examples:")
    for item in diagnostics["suspicious_or_low_confidence_examples"]:
        print(f"  - {item}")


if __name__ == "__main__":
    test_result = unittest.main(verbosity=2, exit=False)
    if test_result.result.wasSuccessful():
        print_dataset_diagnostics()
    raise SystemExit(0 if test_result.result.wasSuccessful() else 1)
