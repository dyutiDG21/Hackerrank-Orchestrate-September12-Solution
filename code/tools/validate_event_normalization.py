#!/usr/bin/env python3
"""Focused validation tests for deterministic financial-event normalization."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import load_dataset  # noqa: E402
from event_normalization import (  # noqa: E402
    CashTreatment,
    MissingExchangeRateError,
    convert_currency,
    normalize_financial_events,
)


class EventNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.normalized = normalize_financial_events(cls.dataset)

    def test_exact_decimal_currency_conversion(self) -> None:
        event = self.normalized.by_event_id["event_2167"]
        rate_key = (date(2023, 10, 15), "USD", "IDR")
        self.assertEqual(event.original_amount, Decimal("1800"))
        self.assertEqual(event.original_currency, "USD")
        self.assertEqual(event.home_currency, "IDR")
        self.assertIsNotNone(event.conversion)
        self.assertIsInstance(event.conversion.amount, Decimal)
        self.assertIsInstance(event.conversion.rate, Decimal)
        self.assertIsInstance(event.conversion.converted_amount, Decimal)
        self.assertEqual(event.conversion.rate_date, date(2023, 10, 15))
        self.assertEqual(event.conversion.rate, Decimal("15833.33"))
        self.assertIs(event.conversion.exchange_rate, self.dataset.indexes.exchange_rates_by_key[rate_key])
        self.assertEqual(event.conversion.exchange_rate.raw["rate"], "15833.33")
        self.assertEqual(event.home_amount, Decimal("28499994.00"))
        self.assertEqual(event.direction, "credit")
        self.assertEqual(event.signed_home_amount, Decimal("28499994.00"))

    def test_conversion_requires_exact_direction_and_date(self) -> None:
        amount = Decimal("10")
        converted = convert_currency(amount, "USD", "IDR", date(2023, 10, 15), self.dataset)
        self.assertEqual(converted.converted_amount, Decimal("158333.30"))
        with self.assertRaises(MissingExchangeRateError):
            convert_currency(amount, "IDR", "USD", date(2023, 10, 15), self.dataset)
        with self.assertRaises(MissingExchangeRateError):
            convert_currency(amount, "USD", "IDR", date(1999, 1, 1), self.dataset)

    def test_status_handling_flags_cash_semantics(self) -> None:
        settled = self.normalized.by_event_id["event_01"]
        scheduled_credit = self.normalized.by_event_id["event_103"]
        pending_debit = self.normalized.by_event_id["event_102"]
        pending_credit = self.normalized.by_event_id["event_1785"]
        failed = self.normalized.by_event_id["event_438"]
        cancelled = self.normalized.by_event_id["event_100"]
        unrealized = self.normalized.by_event_id["event_1856"]

        self.assertEqual(settled.cash_treatment, CashTreatment.SETTLED_CASH_FLOW)
        self.assertTrue(settled.affects_available_cash)
        self.assertEqual(scheduled_credit.cash_treatment, CashTreatment.SCHEDULED_CASH_FLOW)
        self.assertTrue(scheduled_credit.affects_available_cash)

        self.assertEqual(pending_debit.cash_treatment, CashTreatment.PENDING_DEBIT_RESERVATION)
        self.assertTrue(pending_debit.reserves_cash)
        self.assertFalse(pending_debit.affects_available_cash)

        self.assertEqual(pending_credit.cash_treatment, CashTreatment.PENDING_CREDIT_UNAVAILABLE)
        self.assertFalse(pending_credit.affects_available_cash)
        self.assertFalse(pending_credit.reserves_cash)

        self.assertEqual(failed.cash_treatment, CashTreatment.EXCLUDED_FAILED)
        self.assertFalse(failed.is_cash_flow_candidate)
        self.assertIsNone(failed.home_amount)

        self.assertEqual(cancelled.cash_treatment, CashTreatment.EXCLUDED_CANCELLED)
        self.assertFalse(cancelled.is_cash_flow_candidate)
        self.assertIsNone(cancelled.home_amount)

        self.assertEqual(unrealized.cash_treatment, CashTreatment.EXCLUDED_UNREALIZED_NON_CASH)
        self.assertFalse(unrealized.is_cash_flow_candidate)
        self.assertIsNone(unrealized.home_amount)

    def test_missing_amount_events_require_evidence_without_zeroing(self) -> None:
        missing_amount_events = [
            event for event in self.normalized.events if event.cash_treatment == CashTreatment.MISSING_AMOUNT_REQUIRES_EVIDENCE
        ]
        self.assertEqual(len(missing_amount_events), 16)
        self.assertTrue(all(event.original_amount is None for event in missing_amount_events))
        self.assertTrue(all(event.home_amount is None for event in missing_amount_events))
        self.assertTrue(all(event.requires_amount_evidence for event in missing_amount_events))

    def test_linked_lifecycle_records_are_represented_but_not_collapsed(self) -> None:
        refund = self.normalized.by_event_id["event_99"]
        parent = self.normalized.by_event_id["event_98"]
        self.assertEqual(refund.linked_event_id, "event_98")
        self.assertIn("event_99", parent.linked_child_event_ids)
        self.assertIn("event_99", self.normalized.linked_child_ids_by_event_id["event_98"])
        self.assertIn(refund, self.normalized.by_linked_event_id["event_98"])
        self.assertNotEqual(refund.cash_treatment, CashTreatment.EXCLUDED_CANCELLED)
        self.assertEqual(refund.provenance_event_ids, ("event_99",))
        self.assertEqual(parent.provenance_event_ids, ("event_98",))

    def test_provenance_preserves_source_event(self) -> None:
        event = self.normalized.by_event_id["event_102"]
        self.assertEqual(event.normalized_event_id, "event_102")
        self.assertEqual(event.source_event_id, "event_102")
        self.assertEqual(event.provenance_event_ids, ("event_102",))
        self.assertIs(event.source_event, self.dataset.indexes.events_by_event_id["event_102"])

    def test_all_source_events_have_one_normalized_event(self) -> None:
        self.assertEqual(len(self.normalized.events), len(self.dataset.financial_events))
        self.assertEqual(set(self.normalized.by_event_id), set(self.dataset.indexes.events_by_event_id))


if __name__ == "__main__":
    unittest.main(verbosity=2)
