#!/usr/bin/env python3
"""Focused Level 5 payment-plan candidate tests and solved-sample diagnostics."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys
import unittest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import PaymentOption, PurchaseRequest, build_indexes, load_dataset  # noqa: E402
from financial_forecast import build_forecast  # noqa: E402
from financial_state_resolution import resolve_financial_state  # noqa: E402
from payment_capacity import evaluate_capacity_from_baseline, evaluate_request_capacity  # noqa: E402
from payment_planning import (  # noqa: E402
    MAX_SPENDING_CHANGES,
    SpendingChange,
    bounded_change_sets,
    eligible_spending_changes,
    generate_plan_candidates,
    installment_legs,
)
from recurrence_inference import infer_recurring_series  # noqa: E402
from tools.validate_financial_forecast import dataset_with, event, load_cached_claims, profile, salary_event  # noqa: E402


def request(*, requested: Decimal = Decimal("100"), partial: bool = True) -> PurchaseRequest:
    return PurchaseRequest(
        request_id="request_test",
        user_id="user_test",
        request_date=date(2026, 1, 1),
        request_type="purchase",
        requested_amount=requested,
        desired_completion_date=date(2026, 2, 1),
        allows_partial_payment=partial,
        request_text="Synthetic request",
        raw={},
    )


def option(*, option_id: str = "option_1", count: int = 2, amount: Decimal = Decimal("55"), frequency: int = 30) -> PaymentOption:
    return PaymentOption(
        payment_option_id=option_id,
        request_id="request_test",
        payment_method="installments",
        payment_amount=amount,
        number_of_payments=count,
        first_payment_date=date(2026, 1, 2),
        payment_frequency_days=frequency,
        financing_fee=Decimal("10"),
        total_payable_amount=amount * count,
        raw={},
    )


def planning_profile(
    *,
    payment_methods: tuple[str, ...] = ("full_payment",),
    max_installments: int | None = None,
    protected: tuple[str, ...] = (),
    reduce: tuple[str, ...] = (),
    stop: tuple[str, ...] = (),
):
    return replace(
        profile(),
        payment_methods_user_will_consider=payment_methods,
        max_installment_months=max_installments,
        expense_categories_to_protect=protected,
        expense_categories_user_is_willing_to_reduce=reduce,
        expense_categories_user_is_willing_to_stop=stop,
    )


def planning_context(*, events=(), profile_row=None, options=(), request_row=None):
    active_profile = profile_row or planning_profile()
    active_request = request_row or request()
    base = dataset_with(tuple(events), profiles=(active_profile,))
    indexes = build_indexes(
        base.financial_profiles, base.financial_events, (active_request,), (), tuple(options), (), (), base.exchange_rates, (),
    )
    dataset = replace(base, requests=(active_request,), payment_options=tuple(options), indexes=indexes)
    state = resolve_financial_state(dataset, ())
    recurrence = infer_recurring_series(state)
    baseline = build_forecast(dataset, state, recurrence.series, user_id=active_request.user_id, start_date=active_request.request_date)
    capacity = evaluate_capacity_from_baseline(active_request, baseline)
    return dataset, active_request, recurrence.series, baseline, capacity


class PaymentPlanningTests(unittest.TestCase):
    def test_full_now_uses_l3_safety(self) -> None:
        dataset, active_request, series, baseline, capacity = planning_context()
        result = generate_plan_candidates(dataset, active_request, capacity, baseline, series)
        self.assertEqual([candidate.method for candidate in result.candidates], ["full_payment"])

    def test_unsafe_full_now_is_not_candidate(self) -> None:
        dataset, active_request, series, baseline, capacity = planning_context(
            events=(event("debit", amount=Decimal("950"), event_date=date(2026, 1, 2)),)
        )
        result = generate_plan_candidates(dataset, active_request, capacity, baseline, series)
        self.assertEqual(result.candidates, ())

    def test_partial_has_exactly_two_l4_derived_legs(self) -> None:
        active_profile = planning_profile(payment_methods=("partial_payment",))
        dataset, active_request, series, baseline, capacity = planning_context(
            events=(
                event("debit", amount=Decimal("500"), event_date=date(2026, 1, 10)),
                salary_event("salary", date(2026, 1, 11), amount=Decimal("600")),
            ),
            profile_row=active_profile,
            request_row=request(requested=Decimal("500")),
        )
        self.assertEqual(capacity.earliest_date_for_full_payment, date(2026, 1, 11))
        result = generate_plan_candidates(dataset, active_request, capacity, baseline, series)
        candidate = next(item for item in result.candidates if item.method == "partial_payment")
        self.assertEqual(len(candidate.payment_legs), 2)
        self.assertEqual(candidate.payment_legs[0].amount, Decimal("400"))
        self.assertEqual(candidate.payment_legs[1].amount, Decimal("100"))
        self.assertEqual(candidate.total_paid, Decimal("500"))

    def test_partial_requires_all_documented_gates(self) -> None:
        active_profile = planning_profile(payment_methods=("partial_payment",))
        dataset, active_request, series, baseline, capacity = planning_context(profile_row=active_profile)
        blocked = replace(capacity, amount_safe_to_pay=Decimal("0"), earliest_date_for_full_payment=date(2026, 1, 2))
        result = generate_plan_candidates(dataset, active_request, blocked, baseline, series)
        self.assertFalse(any(item.method == "partial_payment" for item in result.candidates))

    def test_installment_schedule_and_restrictions_are_preserved(self) -> None:
        active_profile = planning_profile(payment_methods=("installments",), max_installments=2)
        accepted = option(option_id="accepted", count=2)
        rejected = option(option_id="rejected", count=3, amount=Decimal("40"))
        dataset, active_request, series, baseline, capacity = planning_context(
            profile_row=active_profile, options=(accepted, rejected)
        )
        result = generate_plan_candidates(dataset, active_request, capacity, baseline, series)
        candidates = [item for item in result.candidates if item.method == "installments"]
        self.assertEqual([item.payment_option_id for item in candidates], ["accepted"])
        self.assertEqual(candidates[0].payment_legs, installment_legs(accepted))
        self.assertEqual(candidates[0].total_paid, Decimal("110"))

    def test_wait_uses_exact_l4_date_and_records_late_completion(self) -> None:
        active_profile = planning_profile(payment_methods=("full_payment",))
        dataset, active_request, series, baseline, capacity = planning_context(profile_row=active_profile)
        capacity = replace(capacity, earliest_date_for_full_payment=date(2026, 2, 2))
        result = generate_plan_candidates(dataset, active_request, capacity, baseline, series)
        candidate = next(item for item in result.candidates if item.method == "wait")
        self.assertEqual(candidate.payment_legs[0].payment_date, date(2026, 2, 2))
        self.assertFalse(candidate.completes_by_desired_date)

    def test_blockers_prevent_candidates(self) -> None:
        dataset, active_request, series, baseline, capacity = planning_context(
            events=(event("unknown", amount=None, status="pending"),)
        )
        self.assertTrue(baseline.blockers)
        self.assertEqual(generate_plan_candidates(dataset, active_request, capacity, baseline, series).candidates, ())

    def test_spending_change_eligibility_excludes_protected_and_nonrecurring(self) -> None:
        active_profile = planning_profile(
            protected=("groceries",), reduce=("dining",), stop=("dining",), payment_methods=("full_payment",)
        )
        events = (
            replace(event("dining_1", category="dining", description="Dinner", flexibility="reducible_or_stoppable", event_date=date(2025, 10, 1)), minimum_allowed_amount=Decimal("50")),
            replace(event("dining_2", category="dining", description="Dinner", flexibility="reducible_or_stoppable", event_date=date(2025, 11, 1)), minimum_allowed_amount=Decimal("50")),
            replace(event("dining_3", category="dining", description="Dinner", flexibility="reducible_or_stoppable", event_date=date(2025, 12, 1)), minimum_allowed_amount=Decimal("50")),
            event("groceries_1", category="groceries", description="Groceries", flexibility="stoppable", event_date=date(2025, 10, 1)),
            event("groceries_2", category="groceries", description="Groceries", flexibility="stoppable", event_date=date(2025, 11, 1)),
            event("groceries_3", category="groceries", description="Groceries", flexibility="stoppable", event_date=date(2025, 12, 1)),
        )
        dataset, active_request, series, _baseline, _capacity = planning_context(events=events, profile_row=active_profile)
        changes = eligible_spending_changes(dataset, active_request, series)
        self.assertEqual([(change.action, change.event_id) for change in changes], [
            ("reduce_to", "dining_1"), ("stop", "dining_1"), ("reduce_to", "dining_2"),
            ("stop", "dining_2"), ("reduce_to", "dining_3"), ("stop", "dining_3"),
        ])

    def test_bounded_change_sets_are_deterministic_and_never_exceed_three(self) -> None:
        changes = tuple(SpendingChange("stop", f"event_{index}", None, "dining", (f"event_{index}",)) for index in range(8))
        first = bounded_change_sets(changes)
        self.assertEqual(first, bounded_change_sets(changes))
        self.assertTrue(all(len(item) <= MAX_SPENDING_CHANGES for item in first))
        self.assertTrue(all(len({change.event_id for change in item}) == len(item) for item in first))

    def test_decimal_precision_and_candidate_deduplication(self) -> None:
        active_profile = planning_profile(payment_methods=("full_payment",))
        dataset, active_request, series, baseline, capacity = planning_context(
            profile_row=active_profile, request_row=request(requested=Decimal("0.03"))
        )
        result = generate_plan_candidates(dataset, active_request, capacity, baseline, series)
        self.assertEqual(result.candidates[0].total_paid, Decimal("0.03"))
        self.assertEqual(len(result.candidates), 1)


def print_sample_diagnostics() -> None:
    dataset = load_dataset()
    state = resolve_financial_state(dataset, load_cached_claims(dataset))
    recurrence = infer_recurring_series(state)
    counts = {"full_payment": [], "partial_payment": [], "installments": [], "wait": [], "spending_changes": [], "zero": []}
    for sample in dataset.sample_requests:
        request_row = sample.request
        baseline = build_forecast(dataset, state, recurrence.series, user_id=request_row.user_id, start_date=request_row.request_date)
        capacity = evaluate_capacity_from_baseline(request_row, baseline)
        result = generate_plan_candidates(dataset, request_row, capacity, baseline, recurrence.series)
        methods = {candidate.method for candidate in result.candidates}
        for method in ("full_payment", "partial_payment", "installments", "wait"):
            if method in methods:
                counts[method].append(request_row.request_id)
        if result.deferred_spending_changes:
            counts["spending_changes"].append(request_row.request_id)
        if not result.candidates:
            counts["zero"].append(request_row.request_id)
    print("\nL5 solved-sample candidate diagnostics:")
    for key, values in counts.items():
        print(f"- {key}={len(values)}: {','.join(values) or 'none'}")


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    if result.result.wasSuccessful():
        print_sample_diagnostics()
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
