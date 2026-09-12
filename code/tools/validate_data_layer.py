#!/usr/bin/env python3
"""Focused validation tests for the canonical Buy or Wait? data layer."""

from __future__ import annotations

import sys
import unittest
from datetime import date, timezone
from decimal import Decimal
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import DataValidationError, load_dataset, parse_decimal  # noqa: E402


class DataLayerValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def test_expected_row_counts_load(self) -> None:
        self.assertEqual(len(self.dataset.financial_profiles), 275)
        self.assertEqual(len(self.dataset.financial_events), 25342)
        self.assertEqual(len(self.dataset.requests), 250)
        self.assertEqual(len(self.dataset.sample_requests), 25)
        self.assertEqual(len(self.dataset.payment_options), 790)
        self.assertEqual(len(self.dataset.messages), 215)
        self.assertEqual(len(self.dataset.images), 16)
        self.assertEqual(len(self.dataset.exchange_rates), 134)
        self.assertEqual(len(self.dataset.output_template), 250)

    def test_money_uses_decimal_and_preserves_precision(self) -> None:
        profile = self.dataset.indexes.profiles_by_user_id["user_01"]
        event = self.dataset.indexes.events_by_event_id["event_02"]
        option = self.dataset.indexes.payment_options_by_option_id["payment_option_02"]
        self.assertIsInstance(profile.current_available_balance, Decimal)
        self.assertEqual(profile.current_available_balance, Decimal("58481.1"))
        self.assertEqual(event.amount, Decimal("1475.46"))
        self.assertEqual(option.payment_amount, Decimal("1852.11"))
        self.assertEqual(option.total_payable_amount, Decimal("27781.65"))
        self.assertEqual(parse_decimal("0.10", field="test") + parse_decimal("0.20", field="test"), Decimal("0.30"))

    def test_missing_financial_amounts_remain_none_and_are_image_linked(self) -> None:
        missing_amount_events = [event for event in self.dataset.financial_events if event.amount is None]
        image_related_event_ids = set(self.dataset.indexes.images_by_related_event_id)
        self.assertEqual(len(missing_amount_events), 16)
        self.assertTrue(all(event.event_id in image_related_event_ids for event in missing_amount_events))

    def test_nullable_fields_are_not_defaulted_to_business_values(self) -> None:
        user_01 = self.dataset.indexes.profiles_by_user_id["user_01"]
        full_payment = self.dataset.indexes.payment_options_by_option_id["payment_option_01"]
        self.assertIsNone(user_01.max_installment_months)
        self.assertIsNone(full_payment.payment_frequency_days)
        self.assertIsNone(self.dataset.indexes.events_by_event_id["event_01"].minimum_allowed_amount)

    def test_pipe_delimited_profile_fields_parse_to_tuples(self) -> None:
        user_03 = self.dataset.indexes.profiles_by_user_id["user_03"]
        self.assertEqual(user_03.financial_priorities, ("retirement_investment", "emergency_savings"))
        self.assertEqual(user_03.expense_categories_to_protect, ("rent", "utilities", "groceries"))
        self.assertEqual(user_03.expense_categories_user_is_willing_to_reduce, ("streaming", "shopping"))
        self.assertEqual(user_03.expense_categories_user_is_willing_to_stop, ("streaming", "cloud_storage"))
        self.assertEqual(user_03.payment_methods_user_will_consider, ("full_payment", "partial_payment", "installments"))

    def test_dates_timestamps_and_booleans_are_typed(self) -> None:
        request = self.dataset.indexes.requests_by_request_id["request_26"]
        message = self.dataset.indexes.messages_by_message_id["message_01"]
        self.assertEqual(request.request_date, date(2025, 8, 3))
        self.assertIsInstance(request.allows_partial_payment, bool)
        self.assertEqual(message.sent_at.tzinfo, timezone.utc)

    def test_primary_indexes_cover_all_unique_records(self) -> None:
        self.assertEqual(len(self.dataset.indexes.profiles_by_user_id), len(self.dataset.financial_profiles))
        self.assertEqual(len(self.dataset.indexes.events_by_event_id), len(self.dataset.financial_events))
        self.assertEqual(len(self.dataset.indexes.requests_by_request_id), len(self.dataset.requests))
        self.assertEqual(len(self.dataset.indexes.sample_requests_by_request_id), len(self.dataset.sample_requests))
        self.assertEqual(len(self.dataset.indexes.payment_options_by_option_id), len(self.dataset.payment_options))
        self.assertEqual(len(self.dataset.indexes.messages_by_message_id), len(self.dataset.messages))
        self.assertEqual(len(self.dataset.indexes.images_by_image_id), len(self.dataset.images))
        self.assertEqual(len(self.dataset.indexes.exchange_rates_by_key), len(self.dataset.exchange_rates))

    def test_major_joins_are_indexed_and_valid(self) -> None:
        profile_user_ids = set(self.dataset.indexes.profiles_by_user_id)
        all_request_ids = set(self.dataset.indexes.requests_by_request_id) | set(
            self.dataset.indexes.sample_requests_by_request_id
        )
        event_ids = set(self.dataset.indexes.events_by_event_id)

        self.assertTrue({request.user_id for request in self.dataset.requests}.issubset(profile_user_ids))
        self.assertTrue({event.user_id for event in self.dataset.financial_events}.issubset(profile_user_ids))
        self.assertTrue({option.request_id for option in self.dataset.payment_options}.issubset(all_request_ids))
        self.assertTrue(
            {
                message.related_event_id
                for message in self.dataset.messages
                if message.related_event_id is not None
            }.issubset(event_ids)
        )
        self.assertTrue({image.related_event_id for image in self.dataset.images}.issubset(event_ids))
        self.assertGreater(len(self.dataset.indexes.events_by_user_id["user_01"]), 0)
        self.assertGreaterEqual(len(self.dataset.indexes.payment_options_by_request_id["request_01"]), 2)

    def test_invalid_decimal_rejected(self) -> None:
        with self.assertRaises(DataValidationError):
            parse_decimal("12.3.4", field="bad_amount")


if __name__ == "__main__":
    unittest.main(verbosity=2)
