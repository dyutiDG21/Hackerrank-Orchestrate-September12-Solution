#!/usr/bin/env python3
"""Focused tests for Level 2B deterministic financial-state resolution."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import (  # noqa: E402
    Dataset,
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    build_indexes,
    load_dataset,
)
from evidence_schema import EvidenceClaim  # noqa: E402
from event_normalization import CashTreatment  # noqa: E402
from financial_state_resolution import resolve_financial_state  # noqa: E402


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
    amount: Decimal | None = Decimal("100"),
    currency: str = "USD",
    event_date: date = date(2026, 1, 1),
    settlement_date: date | None = date(2026, 1, 1),
    status: str = "scheduled",
    direction: str = "debit",
    event_type: str = "expense",
    category: str = "rent",
    description: str | None = None,
    linked_event_id: str | None = None,
    user_id: str = "user_test",
) -> FinancialEvent:
    return FinancialEvent(
        event_id=event_id,
        user_id=user_id,
        event_type=event_type,
        description=description or f"Synthetic {event_id}",
        category=category,
        direction=direction,
        amount=amount,
        currency=currency,
        event_date=event_date,
        settlement_date=settlement_date,
        status=status,
        linked_event_id=linked_event_id,
        flexibility="fixed",
        minimum_allowed_amount=None,
        raw={},
    )


def claim(
    claim_id: str,
    *,
    related_event_id: str | None = "event_1",
    claim_type: str = "amount_amendment",
    document_role: str | None = None,
    amount: Decimal | None = Decimal("100"),
    currency: str | None = "USD",
    effective_date: date | None = None,
    status: str | None = None,
    source_id: str = "source_1",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        source_type="image",
        source_id=source_id,
        user_id="user_test",
        request_id="request_test",
        related_event_id=related_event_id,
        claim_type=claim_type,
        document_role=document_role,
        amount=amount,
        currency=currency,
        effective_date=effective_date,
        status=status,
        identifier_role=None,
        referenced_external_id=None,
        provenance=f"synthetic provenance {claim_id}",
        raw_claim={},
    )


def dataset_with(events: tuple[FinancialEvent, ...], *, home_currency: str = "USD") -> Dataset:
    profiles = (profile(currency=home_currency),)
    exchange_rates = (
        ExchangeRate(date(2026, 1, 1), "EUR", home_currency, Decimal("2"), raw={}),
        ExchangeRate(date(2026, 1, 2), "EUR", home_currency, Decimal("3"), raw={}),
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


class FinancialStateResolutionTests(unittest.TestCase):
    def test_linked_event_cancellation(self) -> None:
        parent = event("event_1", status="scheduled")
        child = event("event_2", status="cancelled", linked_event_id="event_1")
        state = resolve_financial_state(dataset_with((parent, child)))
        self.assertFalse(state.by_event_id["event_1"].active)
        self.assertEqual(state.by_event_id["event_1"].inactive_reason, "linked_cancellation_invalidation")

    def test_linked_amendment_replacement(self) -> None:
        parent = event("event_1", amount=Decimal("100"), status="scheduled")
        child = event("event_2", amount=Decimal("120"), status="scheduled", linked_event_id="event_1")
        state = resolve_financial_state(dataset_with((parent, child)))
        self.assertFalse(state.by_event_id["event_1"].active)
        self.assertTrue(state.by_event_id["event_2"].active)
        self.assertEqual(state.by_event_id["event_1"].superseded_by_event_id, "event_2")

    def test_settled_record_supersedes_estimate_or_forecast(self) -> None:
        estimate = event("event_1", amount=Decimal("90"), status="scheduled")
        settled = event("event_2", amount=Decimal("95"), status="settled", linked_event_id="event_1")
        state = resolve_financial_state(dataset_with((estimate, settled)))
        self.assertFalse(state.by_event_id["event_1"].active)
        self.assertEqual(state.by_event_id["event_1"].inactive_reason, "linked_settlement_finalization")

    def test_newer_same_source_evidence_precedence(self) -> None:
        base = event("event_1", status="pending")
        older = claim("claim_1", claim_type="status_confirmation", amount=None, currency=None, status="scheduled", source_id="message_1")
        newer = claim("claim_2", claim_type="status_confirmation", amount=None, currency=None, status="settled", source_id="message_1")
        state = resolve_financial_state(dataset_with((base,)), (older, newer))
        self.assertEqual(state.by_event_id["event_1"].resolved_event.status, "settled")

    def test_non_cash_lifecycle_status_does_not_overwrite_financial_status(self) -> None:
        base = event("event_1", status="settled", direction="debit", category="groceries", description="Delivered grocery order")
        delivered = claim(
            "claim_1",
            claim_type="status_confirmation",
            amount=None,
            currency=None,
            status="Delivered",
            source_id="image_1",
        )
        state = resolve_financial_state(dataset_with((base,)), (delivered,))
        self.assertEqual(state.by_event_id["event_1"].resolved_event.status, "settled")
        self.assertEqual(state.unresolved_evidence[0].reason, "non_cash_lifecycle_status_not_applied")

    def test_case_insensitive_financial_status_can_overwrite_financial_status(self) -> None:
        base = event("event_1", status="pending")
        settled = claim(
            "claim_1",
            claim_type="status_confirmation",
            amount=None,
            currency=None,
            status="Settled",
            source_id="message_1",
        )
        state = resolve_financial_state(dataset_with((base,)), (settled,))
        self.assertEqual(state.by_event_id["event_1"].resolved_event.status, "settled")

    def test_field_level_amendment_preserves_unrelated_fields(self) -> None:
        base = event("event_1", amount=Decimal("100"), category="utilities", status="scheduled")
        amount_claim = claim("claim_1", amount=Decimal("125"), currency="USD")
        state = resolve_financial_state(dataset_with((base,)), (amount_claim,))
        resolved = state.by_event_id["event_1"].resolved_event
        self.assertEqual(resolved.amount, Decimal("125"))
        self.assertEqual(resolved.category, "utilities")
        self.assertEqual(resolved.status, "scheduled")

    def test_missing_amount_image_resolution_using_document_role(self) -> None:
        base = event("event_1", amount=None, direction="debit", status="scheduled")
        total = claim("claim_1", document_role="total_amount", amount=Decimal("200"))
        generic = claim("claim_2", document_role=None, amount=Decimal("100"))
        state = resolve_financial_state(dataset_with((base,)), (generic, total))
        self.assertEqual(state.by_event_id["event_1"].resolved_event.amount, Decimal("200"))

    def test_payroll_base_gross_vs_net_transferred_distinction(self) -> None:
        base = event("event_1", amount=None, direction="credit", event_type="income", category="salary", status="scheduled")
        claims = (
            claim("base", document_role="base_salary", amount=Decimal("4500000"), currency="IDR"),
            claim("gross", document_role="total_earnings", amount=Decimal("4780800"), currency="IDR"),
            claim("net", document_role="net_pay", amount=Decimal("4365000"), currency="IDR"),
        )
        state = resolve_financial_state(dataset_with((base,), home_currency="IDR"), claims)
        self.assertEqual(state.by_event_id["event_1"].resolved_event.amount, Decimal("4365000"))

    def test_receipt_total_received_balance_distinction(self) -> None:
        base = event("event_1", amount=None, direction="debit", status="pending")
        claims = (
            claim("total", document_role="total_amount", amount=Decimal("200000"), currency="INR"),
            claim("received", document_role="amount_received", amount=Decimal("100000"), currency="INR"),
            claim("balance", document_role="balance_due", amount=Decimal("100000"), currency="INR"),
        )
        state = resolve_financial_state(dataset_with((base,), home_currency="INR"), claims)
        self.assertEqual(state.by_event_id["event_1"].resolved_event.amount, Decimal("200000"))
        self.assertIn("conservative", state.by_event_id["event_1"].traces[0].rule)

    def test_outstanding_balance_debit_prefers_balance_due_role(self) -> None:
        base = event(
            "event_1",
            amount=None,
            direction="debit",
            status="scheduled",
            description="Outstanding rent balance",
        )
        claims = (
            claim("total", document_role="total_amount", amount=Decimal("200000"), currency="INR"),
            claim("received", document_role="amount_received", amount=Decimal("100000"), currency="INR"),
            claim("balance", document_role="balance_due", amount=Decimal("100000"), currency="INR"),
        )
        state = resolve_financial_state(dataset_with((base,), home_currency="INR"), claims)
        self.assertEqual(state.by_event_id["event_1"].resolved_event.amount, Decimal("100000"))
        self.assertEqual(state.by_event_id["event_1"].traces[0].rule, "role_specific_amount_resolution")

    def test_unrelated_image_line_items_not_applied(self) -> None:
        base = event("event_1", amount=None, direction="debit", status="scheduled")
        deduction = claim("deduction", document_role="total_deductions", amount=Decimal("415800"), currency="USD")
        state = resolve_financial_state(dataset_with((base,)), (deduction,))
        self.assertIsNone(state.by_event_id["event_1"].resolved_event.amount)
        self.assertTrue(state.unresolved_evidence)

    def test_conservative_inbound_ambiguity(self) -> None:
        base = event("event_1", amount=None, direction="credit", event_type="income", category="bonus")
        high = claim("high", document_role=None, amount=Decimal("500"))
        low = claim("low", document_role=None, amount=Decimal("300"))
        state = resolve_financial_state(dataset_with((base,)), (high, low))
        self.assertEqual(state.by_event_id["event_1"].resolved_event.amount, Decimal("300"))
        self.assertEqual(state.by_event_id["event_1"].traces[0].rule, "conservative_inbound_amount")

    def test_conservative_outbound_ambiguity(self) -> None:
        base = event("event_1", amount=None, direction="debit")
        high = claim("high", document_role=None, amount=Decimal("500"))
        low = claim("low", document_role=None, amount=Decimal("300"))
        state = resolve_financial_state(dataset_with((base,)), (high, low))
        self.assertEqual(state.by_event_id["event_1"].resolved_event.amount, Decimal("500"))
        self.assertEqual(state.by_event_id["event_1"].traces[0].rule, "conservative_outbound_amount")

    def test_unlinked_payroll_evidence_deferred_for_l2c(self) -> None:
        base = event("event_1", amount=Decimal("100"))
        unlinked = claim("claim_1", related_event_id=None, claim_type="salary_change", amount=Decimal("200"))
        state = resolve_financial_state(dataset_with((base,)), (unlinked,))
        self.assertEqual(state.unresolved_evidence[0].reason, "requires_recurrence_or_payroll_matching")

    def test_null_claims_do_not_overwrite_known_values(self) -> None:
        base = event("event_1", amount=Decimal("100"))
        null_amount = claim("claim_1", claim_type="amount_amendment", amount=None, currency=None)
        state = resolve_financial_state(dataset_with((base,)), (null_amount,))
        self.assertEqual(state.by_event_id["event_1"].resolved_event.amount, Decimal("100"))

    def test_fx_recomputed_after_evidence_changes_amount_and_date(self) -> None:
        base = event("event_1", amount=Decimal("10"), currency="EUR", settlement_date=date(2026, 1, 1), status="scheduled")
        amount_claim = claim("amount", amount=Decimal("20"), currency="EUR")
        date_claim = claim("date", claim_type="date_amendment", amount=None, currency=None, effective_date=date(2026, 1, 2))
        state = resolve_financial_state(dataset_with((base,), home_currency="USD"), (amount_claim, date_claim))
        normalized = state.by_event_id["event_1"].normalized_event
        self.assertEqual(normalized.conversion.rate, Decimal("3"))
        self.assertEqual(normalized.home_amount, Decimal("60"))

    def test_derived_cash_flow_flags_recomputed_after_status_change(self) -> None:
        base = event("event_1", amount=Decimal("100"), status="pending", direction="debit")
        cancel = claim("cancel", claim_type="cancellation", amount=None, currency=None, status="cancelled")
        state = resolve_financial_state(dataset_with((base,)), (cancel,))
        normalized = state.by_event_id["event_1"].normalized_event
        self.assertEqual(normalized.cash_treatment, CashTreatment.EXCLUDED_CANCELLED)
        self.assertFalse(normalized.reserves_cash)

    def test_pending_debit_with_missing_amount_remains_reserved_for_l3(self) -> None:
        base = event("event_1", amount=None, status="pending", direction="debit")
        state = resolve_financial_state(dataset_with((base,)), ())
        normalized = state.by_event_id["event_1"].normalized_event
        self.assertEqual(normalized.cash_treatment, CashTreatment.MISSING_AMOUNT_REQUIRES_EVIDENCE)
        self.assertTrue(normalized.reserves_cash)
        self.assertFalse(normalized.affects_available_cash)

    def test_linked_separate_cash_movement_not_collapsed(self) -> None:
        purchase = event("event_1", amount=Decimal("100"), direction="debit", event_type="expense", status="settled")
        refund = event("event_2", amount=Decimal("25"), direction="credit", event_type="refund", status="settled", linked_event_id="event_1")
        state = resolve_financial_state(dataset_with((purchase, refund)))
        self.assertTrue(state.by_event_id["event_1"].active)
        self.assertTrue(state.by_event_id["event_2"].active)

    def test_provenance_resolution_trace(self) -> None:
        base = event("event_1", amount=None)
        amount_claim = claim("claim_1", amount=Decimal("150"), source_id="image_99")
        state = resolve_financial_state(dataset_with((base,)), (amount_claim,))
        trace = state.by_event_id["event_1"].traces[0]
        self.assertEqual(trace.source_id, "image_99")
        self.assertEqual(trace.previous_value, None)
        self.assertEqual(trace.resolved_value, "150")

    def test_idempotence(self) -> None:
        base = event("event_1", amount=None)
        amount_claim = claim("claim_1", amount=Decimal("150"))
        dataset = dataset_with((base,))
        first = resolve_financial_state(dataset, (amount_claim,))
        second = resolve_financial_state(dataset, (amount_claim,))
        self.assertEqual(first, second)

    def test_stable_deterministic_ordering_tie_breaking(self) -> None:
        later = event("event_b", event_date=date(2026, 1, 2), settlement_date=date(2026, 1, 2))
        earlier = event("event_a", event_date=date(2026, 1, 1), settlement_date=date(2026, 1, 1))
        state = resolve_financial_state(dataset_with((later, earlier)))
        self.assertEqual([record.event_id for record in state.events], ["event_a", "event_b"])

    def test_real_dataset_sanity_check(self) -> None:
        dataset = load_dataset()
        state = resolve_financial_state(dataset, ())
        self.assertEqual(len(state.events), len(dataset.financial_events))
        self.assertLessEqual(len(state.effective_events), len(state.events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
