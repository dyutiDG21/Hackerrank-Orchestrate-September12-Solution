"""Focused Level 7 final-orchestration validation without provider calls."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import load_dataset
from main import evaluation_scope, validate_output
from recommendation_selection import FinalDecision


class FinalOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset()
        self.request = self.dataset.requests[0]

    def decision(self, **overrides: object) -> FinalDecision:
        values: dict[str, object] = {
            "request_id": self.request.request_id,
            "amount_safe_to_pay": Decimal("0"),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": None,
            "spending_changes_needed": "none",
            "decision_explanation": "No eligible plan completes the request by the deadline while preserving the required minimum balance.",
            "selected_candidate_id": None,
        }
        values.update(overrides)
        return FinalDecision(**values)  # type: ignore[arg-type]

    def all_fallbacks(self, first: FinalDecision) -> tuple[FinalDecision, ...]:
        return (first, *(FinalDecision(
            request_id=request.request_id,
            amount_safe_to_pay=Decimal("0"),
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=None,
            spending_changes_needed="none",
            decision_explanation="No eligible plan completes the request by the deadline while preserving the required minimum balance.",
            selected_candidate_id=None,
        ) for request in self.dataset.requests[1:]))

    def test_evaluation_scope_is_limited_to_evaluation_request_users(self) -> None:
        messages, images = evaluation_scope(self.dataset)
        request_ids = {item.request_id for item in self.dataset.requests}
        user_ids = {item.user_id for item in self.dataset.requests}
        self.assertTrue(messages)
        self.assertTrue(images)
        self.assertTrue(all(item.user_id in user_ids for item in (*messages, *images)))
        self.assertTrue(all(item.request_id in request_ids or item.related_event_id for item in (*messages, *images)))

    def test_output_validator_rejects_malformed_payment_plan(self) -> None:
        decision = self.decision(
            affordability_status="affordable_now",
            recommended_payment_method="full_payment",
            payment_plan="not-a-plan",
            earliest_date_for_full_payment=self.request.request_date,
            selected_candidate_id="candidate",
        )
        with self.assertRaisesRegex(ValueError, "malformed payment plan"):
            validate_output(self.dataset, self.all_fallbacks(decision))

    def test_output_validator_rejects_conflicting_change_actions(self) -> None:
        decision = self.decision(
            affordability_status="affordable_with_plan",
            recommended_payment_method="full_payment",
            payment_plan=f"{self.request.request_date.isoformat()}:{self.request.requested_amount}",
            earliest_date_for_full_payment=self.request.request_date,
            spending_changes_needed="stop:event_1|reduce_to:event_1:1",
            selected_candidate_id="candidate",
        )
        with self.assertRaisesRegex(ValueError, "conflicting spending change"):
            validate_output(self.dataset, self.all_fallbacks(decision))


if __name__ == "__main__":
    unittest.main(verbosity=2)
