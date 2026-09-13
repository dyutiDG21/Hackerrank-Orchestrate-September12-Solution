#!/usr/bin/env python3
"""Focused tests and dataset diagnostics for Level 4 payment capacity."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
import sys
import unittest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import PurchaseRequest, load_dataset  # noqa: E402
from financial_forecast import build_forecast  # noqa: E402
from financial_state_resolution import resolve_financial_state  # noqa: E402
from payment_capacity import evaluate_request_capacity  # noqa: E402
from recurrence_inference import infer_recurring_series  # noqa: E402
from tools.validate_financial_forecast import (  # noqa: E402
    dataset_with,
    event,
    load_cached_claims,
    profile,
)


def request(
    *,
    request_id: str = "request_test",
    user_id: str = "user_test",
    request_date: date = date(2026, 1, 1),
    requested_amount: Decimal = Decimal("100"),
    desired_completion_date: date = date(2026, 2, 1),
    allows_partial_payment: bool = True,
) -> PurchaseRequest:
    return PurchaseRequest(
        request_id=request_id,
        user_id=user_id,
        request_date=request_date,
        request_type="purchase",
        requested_amount=requested_amount,
        desired_completion_date=desired_completion_date,
        allows_partial_payment=allows_partial_payment,
        request_text="Synthetic request",
        raw={},
    )


def capacity_for(
    events=(),
    *,
    profile_row=profile(),
    request_row: PurchaseRequest | None = None,
):
    active_request = request_row or request(user_id=profile_row.user_id)
    dataset = dataset_with(tuple(events), profiles=(profile_row,))
    state = resolve_financial_state(dataset, ())
    recurrence = infer_recurring_series(state)
    return evaluate_request_capacity(dataset, state, recurrence.series, active_request)


class PaymentCapacityTests(unittest.TestCase):
    def test_full_requested_amount_safe_now(self) -> None:
        result = capacity_for(request_row=request(requested_amount=Decimal("100")))
        self.assertEqual(result.amount_safe_to_pay, Decimal("100"))
        self.assertEqual(result.earliest_date_for_full_payment, date(2026, 1, 1))

    def test_partial_safe_capacity_only(self) -> None:
        result = capacity_for(
            (event("event_1", amount=Decimal("500"), event_date=date(2026, 1, 5)),),
            request_row=request(requested_amount=Decimal("500")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("400"))
        self.assertIsNone(result.earliest_date_for_full_payment)

    def test_zero_safe_capacity(self) -> None:
        result = capacity_for(
            profile_row=profile(balance=Decimal("100"), minimum=Decimal("100")),
            request_row=request(requested_amount=Decimal("50")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("0"))
        self.assertIsNone(result.earliest_date_for_full_payment)

    def test_exact_cap_at_requested_amount(self) -> None:
        result = capacity_for(request_row=request(requested_amount=Decimal("25")))
        self.assertEqual(result.amount_safe_to_pay, Decimal("25"))

    def test_baseline_already_violates_minimum(self) -> None:
        result = capacity_for(
            (event("event_1", amount=Decimal("950"), event_date=date(2026, 1, 2)),),
            request_row=request(requested_amount=Decimal("1")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("0"))
        self.assertTrue(result.baseline_violates_minimum)
        self.assertIsNone(result.earliest_date_for_full_payment)

    def test_pending_debit_reservation_reduces_safe_amount(self) -> None:
        result = capacity_for(
            (event("event_1", amount=Decimal("300"), status="pending", event_date=date(2026, 1, 10)),),
            request_row=request(requested_amount=Decimal("800")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("600"))

    def test_pending_credit_does_not_increase_safe_amount(self) -> None:
        result = capacity_for(
            (event("event_1", amount=Decimal("500"), direction="credit", event_type="income", category="bonus", status="pending"),),
            request_row=request(requested_amount=Decimal("1200")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("900"))
        self.assertFalse(result.blocked)

    def test_unresolved_missing_debit_blocks_proving_safety(self) -> None:
        result = capacity_for(
            (event("event_1", amount=None, status="pending"),),
            request_row=request(requested_amount=Decimal("1")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("0"))
        self.assertTrue(result.blocked)
        self.assertEqual(result.blockers[0].reason, "pending_debit_missing_amount")

    def test_safe_amount_independent_of_partial_payment_flag(self) -> None:
        events = (event("event_1", amount=Decimal("500"), event_date=date(2026, 1, 5)),)
        allowed = capacity_for(events, request_row=request(requested_amount=Decimal("500"), allows_partial_payment=True))
        disallowed = capacity_for(events, request_row=request(requested_amount=Decimal("500"), allows_partial_payment=False))
        self.assertEqual(allowed.amount_safe_to_pay, disallowed.amount_safe_to_pay)

    def test_earliest_full_date_equals_request_date_when_safe_now(self) -> None:
        result = capacity_for(request_row=request(requested_amount=Decimal("100")))
        self.assertEqual(result.earliest_date_for_full_payment, result.request_date)

    def test_earliest_full_date_after_future_cash_improvement(self) -> None:
        result = capacity_for(
            (event("event_1", amount=Decimal("500"), direction="credit", event_type="income", category="salary", event_date=date(2026, 1, 10)),),
            profile_row=profile(balance=Decimal("100"), minimum=Decimal("50")),
            request_row=request(requested_amount=Decimal("100")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("50"))
        self.assertEqual(result.earliest_date_for_full_payment, date(2026, 1, 10))

    def test_same_day_bonus_credit_does_not_make_same_day_payment_safe(self) -> None:
        result = capacity_for(
            (event("event_1", amount=Decimal("100"), direction="credit", event_type="income", category="bonus", event_date=date(2026, 1, 10)),),
            profile_row=profile(balance=Decimal("100"), minimum=Decimal("50")),
            request_row=request(requested_amount=Decimal("100")),
        )
        self.assertNotEqual(result.earliest_date_for_full_payment, date(2026, 1, 10))

    def test_no_safe_full_payment_date_within_horizon(self) -> None:
        result = capacity_for(
            profile_row=profile(balance=Decimal("500"), minimum=Decimal("100")),
            request_row=request(requested_amount=Decimal("1000")),
        )
        self.assertIsNone(result.earliest_date_for_full_payment)

    def test_decimal_precision_no_float_rounding(self) -> None:
        result = capacity_for(
            profile_row=profile(balance=Decimal("100.03"), minimum=Decimal("100.00")),
            request_row=request(requested_amount=Decimal("0.04")),
        )
        self.assertEqual(result.amount_safe_to_pay, Decimal("0.03"))

    def test_baseline_forecast_remains_immutable(self) -> None:
        result = capacity_for(request_row=request(requested_amount=Decimal("100")))
        before = result
        again = capacity_for(request_row=request(requested_amount=Decimal("100")))
        self.assertEqual(result, before)
        self.assertEqual(result, again)

    def test_deterministic_idempotent_calculation(self) -> None:
        events = (event("event_1", amount=Decimal("500"), event_date=date(2026, 1, 5)),)
        self.assertEqual(capacity_for(events), capacity_for(events))


def print_dataset_diagnostics() -> None:
    dataset = load_dataset()
    claims = load_cached_claims(dataset)
    state = resolve_financial_state(dataset, claims)
    recurrence = infer_recurring_series(state)
    results = tuple(evaluate_request_capacity(dataset, state, recurrence.series, item) for item in dataset.requests)
    baselines = tuple(
        build_forecast(dataset, state, recurrence.series, user_id=item.user_id, start_date=item.request_date)
        for item in dataset.requests
    )

    full_now = sum(1 for item in results if item.amount_safe_to_pay == item.requested_amount)
    partial = sum(1 for item in results if Decimal("0") < item.amount_safe_to_pay < item.requested_amount)
    zero = sum(1 for item in results if item.amount_safe_to_pay == Decimal("0"))
    blocked = sum(1 for item in results if item.blocked)
    earliest_now = sum(1 for item in results if item.earliest_date_for_full_payment == item.request_date)
    earliest_later = sum(
        1
        for item in results
        if item.earliest_date_for_full_payment is not None and item.earliest_date_for_full_payment != item.request_date
    )
    no_earliest = sum(1 for item in results if item.earliest_date_for_full_payment is None)
    invariant_violations = [
        item.request_id
        for item in results
        if item.amount_safe_to_pay < Decimal("0") or item.amount_safe_to_pay > item.requested_amount
    ]
    suspicious_later = [
        item.request_id
        for item in results
        if item.earliest_date_for_full_payment is not None
        and item.earliest_date_for_full_payment != item.request_date
        and item.amount_safe_to_pay == item.requested_amount
    ]

    print("\nDataset payment capacity diagnostics:")
    print(f"- requests={len(results)}")
    print(f"- full_requested_amount_safe_now={full_now}")
    print(f"- partial_safe_capacity={partial}")
    print(f"- zero_safe_capacity={zero}")
    print(f"- blocked_by_unresolved_l3={blocked}")
    print(f"- earliest_full_date_request_date={earliest_now}")
    print(f"- earliest_full_date_later={earliest_later}")
    print(f"- no_earliest_full_date={no_earliest}")
    print(f"- amount_bound_invariant_violations={invariant_violations}")
    print(f"- suspicious_later_when_safe_now={suspicious_later}")

    representative = []
    for label, predicate in (
        ("pending_debit_reservation", lambda item: False),
        ("pending_or_blocked", lambda item: bool(item.blockers)),
        ("partial_capacity", lambda item: Decimal("0") < item.amount_safe_to_pay < item.requested_amount),
        ("baseline_violation", lambda item: item.baseline_violates_minimum),
        ("future_full_date", lambda item: item.earliest_date_for_full_payment is not None and item.earliest_date_for_full_payment != item.request_date),
    ):
        if label == "pending_debit_reservation":
            for item, forecast in zip(results, baselines):
                if any(movement.source == "pending_debit_reservation" for movement in forecast.movements):
                    representative.append((label, item))
                    break
            continue
        for item in results:
            if predicate(item):
                representative.append((label, item))
                break
    print("- representative_cases:")
    for label, item in representative:
        print(
            f"  - kind={label}; request_id={item.request_id}; requested={item.requested_amount}; "
            f"safe={item.amount_safe_to_pay}; earliest={item.earliest_date_for_full_payment}; "
            f"blocked={item.blocked}; blockers={[blocker.reason for blocker in item.blockers]}"
        )


if __name__ == "__main__":
    test_result = unittest.main(verbosity=2, exit=False)
    if test_result.result.wasSuccessful():
        print_dataset_diagnostics()
    raise SystemExit(0 if test_result.result.wasSuccessful() else 1)
