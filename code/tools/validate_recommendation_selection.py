#!/usr/bin/env python3
"""Focused Level 6 tests and cached-evidence solved-sample diagnostics."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
import sys
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import load_dataset  # noqa: E402
from financial_forecast import build_forecast  # noqa: E402
from financial_state_resolution import resolve_financial_state  # noqa: E402
from payment_capacity import evaluate_request_capacity  # noqa: E402
from payment_planning import PaymentLeg, PlanCandidate, SpendingChange, generate_plan_candidates  # noqa: E402
from recommendation_selection import (  # noqa: E402
    OUTPUT_COLUMNS, derive_decision, decision_row, format_decimal, format_plan_decimal, rank_key, select_candidate,
)
from recurrence_inference import infer_recurring_series  # noqa: E402
from tools.validate_financial_forecast import forecast_for, load_cached_claims  # noqa: E402
from tools.validate_payment_planning import planning_context, planning_profile, request  # noqa: E402


def fake_candidate(*, method="full_payment", total=Decimal("100"), start=date(2026, 1, 1), legs=1, changes=(), option_id=None, on_time=True):
    baseline = forecast_for(())
    payment_legs = tuple(PaymentLeg(start, total / Decimal(legs), ("test", str(index))) for index in range(legs))
    return PlanCandidate("candidate_" + method + str(total) + str(legs), method, payment_legs, changes, option_id, start, payment_legs[-1].payment_date, total, on_time, baseline, 0, ())


class RecommendationSelectionTests(unittest.TestCase):
    def test_official_ranking_order_and_stable_tie(self) -> None:
        changed = fake_candidate(changes=(SpendingChange("stop", "event_1", None, "dining", ("event_1",)),))
        plain = fake_candidate(total=Decimal("110"))
        cheaper = fake_candidate(total=Decimal("90"))
        self.assertEqual(select_candidate((changed, plain, cheaper)), cheaper)
        early = fake_candidate(total=Decimal("90"), start=date(2026, 1, 1))
        late = fake_candidate(total=Decimal("90"), start=date(2026, 1, 2))
        self.assertEqual(select_candidate((late, early)), early)
        fewer = fake_candidate(total=Decimal("90"), legs=1)
        more = fake_candidate(total=Decimal("90"), legs=2)
        self.assertEqual(select_candidate((more, fewer)), fewer)
        option_a = fake_candidate(method="installments", option_id="option_a")
        option_b = fake_candidate(method="installments", option_id="option_b")
        self.assertEqual(select_candidate((option_b, option_a)), option_a)
        self.assertEqual(rank_key(option_a), rank_key(option_a))

    def test_formatting_and_status_mapping(self) -> None:
        self.assertEqual(format_decimal(Decimal("603.30")), "603.3")
        self.assertEqual(format_plan_decimal(Decimal("620.40")), "620.40")
        self.assertEqual(format_decimal(Decimal("1E+3")), "1000")
        dataset, active_request, series, baseline, capacity = planning_context(profile_row=planning_profile())
        candidates = generate_plan_candidates(dataset, active_request, capacity, baseline, series).candidates
        decision = derive_decision(dataset, active_request, capacity, candidates)
        self.assertEqual(decision.affordability_status, "affordable_now")
        self.assertEqual(tuple(decision_row(decision)), OUTPUT_COLUMNS)

    def test_post_deadline_candidate_falls_back(self) -> None:
        dataset, active_request, _series, _baseline, capacity = planning_context(profile_row=planning_profile())
        late = fake_candidate(start=date(2026, 2, 2), on_time=False)
        decision = derive_decision(dataset, active_request, capacity, (late,))
        self.assertEqual((decision.affordability_status, decision.recommended_payment_method, decision.payment_plan), ("not_affordable", "not_recommended", "none"))

    def test_plan_statuses_and_serialization(self) -> None:
        dataset, active_request, _series, _baseline, capacity = planning_context(profile_row=planning_profile())
        changed = fake_candidate(changes=(SpendingChange("reduce_to", "event_1", Decimal("12.50"), "dining", ("event_1",)),))
        partial = fake_candidate(method="partial_payment", legs=2)
        wait = fake_candidate(method="wait", start=date(2026, 1, 2))
        changed_decision = derive_decision(dataset, active_request, capacity, (changed,))
        partial_decision = derive_decision(dataset, active_request, capacity, (partial,))
        wait_decision = derive_decision(dataset, active_request, capacity, (wait,))
        self.assertEqual(changed_decision.affordability_status, "affordable_with_plan")
        self.assertEqual(partial_decision.affordability_status, "affordable_with_plan")
        self.assertEqual(wait_decision.affordability_status, "affordable_later")
        self.assertIn("reduce_to:event_1:12.50", changed_decision.spending_changes_needed)
        self.assertEqual(partial_decision.payment_plan.count("|"), 1)


def diagnostics() -> None:
    dataset = load_dataset()
    state = resolve_financial_state(dataset, load_cached_claims(dataset))
    recurrence = infer_recurring_series(state)
    decisions = []
    for sample in dataset.sample_requests:
        request_row = sample.request
        baseline = build_forecast(dataset, state, recurrence.series, user_id=request_row.user_id, start_date=request_row.request_date)
        capacity = evaluate_request_capacity(dataset, state, recurrence.series, request_row)
        candidates = generate_plan_candidates(dataset, request_row, capacity, baseline, recurrence.series).candidates
        decisions.append((sample, derive_decision(dataset, request_row, capacity, candidates)))
    fields = ("amount_safe_to_pay", "affordability_status", "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed")
    matches = {field: 0 for field in fields}
    all_matches = 0
    differing = []
    correct_despite_l3_l4 = []
    for sample, decision in decisions:
        actual = decision_row(decision)
        expected = {
            "amount_safe_to_pay": format_decimal(sample.amount_safe_to_pay),
            "affordability_status": sample.affordability_status,
            "recommended_payment_method": sample.recommended_payment_method,
            "payment_plan": sample.payment_plan or "",
            "earliest_date_for_full_payment": sample.earliest_date_for_full_payment.isoformat() if sample.earliest_date_for_full_payment else "",
            "spending_changes_needed": sample.spending_changes_needed or "",
        }
        equal = [field for field in fields if actual[field] == expected[field]]
        for field in equal:
            matches[field] += 1
        if len(equal) == len(fields):
            all_matches += 1
        if (
            actual["affordability_status"] == expected["affordability_status"]
            and actual["recommended_payment_method"] == expected["recommended_payment_method"]
            and (actual["amount_safe_to_pay"] != expected["amount_safe_to_pay"] or actual["earliest_date_for_full_payment"] != expected["earliest_date_for_full_payment"])
        ):
            correct_despite_l3_l4.append(sample.request.request_id)
        if actual["affordability_status"] != expected["affordability_status"] or actual["recommended_payment_method"] != expected["recommended_payment_method"]:
            likely = "L3/L4" if actual["amount_safe_to_pay"] != expected["amount_safe_to_pay"] or actual["earliest_date_for_full_payment"] != expected["earliest_date_for_full_payment"] else "L5/L6"
            differing.append((sample.request.request_id, expected["affordability_status"], expected["recommended_payment_method"], actual["affordability_status"], actual["recommended_payment_method"], likely))
    print("\nL6 solved-sample diagnostics:")
    print("- " + " ".join(f"{field}={matches[field]}/25" for field in fields))
    print(f"- all_six_structured={all_matches}/25 explanations={sum(bool(item.decision_explanation) for _, item in decisions)}/25")
    print(f"- recommendation_differences={len(differing)}")
    print(f"- correct_method_status_despite_l3_l4={','.join(correct_despite_l3_l4) or 'none'}")
    for item in differing:
        print(f"  - {item[0]} | expected {item[1]}/{item[2]} | actual {item[3]}/{item[4]} | {item[5]}")


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    if result.result.wasSuccessful():
        diagnostics()
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
