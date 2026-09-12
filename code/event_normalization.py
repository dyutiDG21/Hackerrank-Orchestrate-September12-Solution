"""Deterministic financial-event normalization for Buy or Wait?

The functions here prepare typed Level 1A events for later evidence resolution
and forecasting. They intentionally avoid recurrence inference, message/image
interpretation, forecast simulation, plan generation, and recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from data_layer import Dataset, DataValidationError, ExchangeRate, FinancialEvent, load_dataset


class MissingExchangeRateError(DataValidationError):
    """Raised when an exact dated currency-pair rate is required but absent."""


class UnsupportedCashStateError(DataValidationError):
    """Raised when an event status/direction is outside the known challenge contract."""


class CashTreatment(str, Enum):
    SETTLED_CASH_FLOW = "settled_cash_flow"
    SCHEDULED_CASH_FLOW = "scheduled_cash_flow"
    PENDING_DEBIT_RESERVATION = "pending_debit_reservation"
    PENDING_CREDIT_UNAVAILABLE = "pending_credit_unavailable"
    EXCLUDED_FAILED = "excluded_failed"
    EXCLUDED_CANCELLED = "excluded_cancelled"
    EXCLUDED_UNREALIZED_NON_CASH = "excluded_unrealized_non_cash"
    MISSING_AMOUNT_REQUIRES_EVIDENCE = "missing_amount_requires_evidence"


@dataclass(frozen=True)
class CurrencyConversion:
    amount: Decimal
    from_currency: str
    to_currency: str
    rate_date: date
    rate: Decimal
    converted_amount: Decimal
    exchange_rate: ExchangeRate | None


@dataclass(frozen=True)
class NormalizedCashEvent:
    normalized_event_id: str
    source_event_id: str
    provenance_event_ids: tuple[str, ...]
    user_id: str
    home_currency: str
    event_type: str
    category: str
    direction: str
    status: str
    flexibility: str
    event_date: date
    settlement_date: date | None
    cash_date: date
    original_amount: Decimal | None
    original_currency: str
    home_amount: Decimal | None
    signed_home_amount: Decimal | None
    cash_treatment: CashTreatment
    is_cash_flow_candidate: bool
    affects_available_cash: bool
    reserves_cash: bool
    requires_amount_evidence: bool
    linked_event_id: str | None
    linked_child_event_ids: tuple[str, ...]
    conversion: CurrencyConversion | None
    exclusion_reason: str | None
    source_event: FinancialEvent


@dataclass(frozen=True)
class NormalizedEventSet:
    events: tuple[NormalizedCashEvent, ...]
    by_event_id: dict[str, NormalizedCashEvent]
    by_user_id: dict[str, tuple[NormalizedCashEvent, ...]]
    by_linked_event_id: dict[str, tuple[NormalizedCashEvent, ...]]
    linked_child_ids_by_event_id: dict[str, tuple[str, ...]]


def convert_currency(
    amount: Decimal,
    from_currency: str,
    to_currency: str,
    rate_date: date,
    dataset: Dataset,
) -> CurrencyConversion:
    """Convert using the exact supplied rate_date and from->to direction.

    Identity conversion returns rate 1 and does not consult exchange_rates.csv.
    Reverse rates or nearest-date fallbacks are intentionally not inferred.
    """

    if from_currency == to_currency:
        return CurrencyConversion(
            amount=amount,
            from_currency=from_currency,
            to_currency=to_currency,
            rate_date=rate_date,
            rate=Decimal("1"),
            converted_amount=amount,
            exchange_rate=None,
        )

    key = (rate_date, from_currency, to_currency)
    rate = dataset.indexes.exchange_rates_by_key.get(key)
    if rate is None:
        raise MissingExchangeRateError(
            f"Missing exchange rate for {rate_date.isoformat()} {from_currency}->{to_currency}"
        )
    return CurrencyConversion(
        amount=amount,
        from_currency=from_currency,
        to_currency=to_currency,
        rate_date=rate_date,
        rate=rate.rate,
        converted_amount=amount * rate.rate,
        exchange_rate=rate,
    )


def signed_amount_for_direction(amount: Decimal, direction: str, event_id: str) -> Decimal:
    if direction == "debit":
        return -amount
    if direction == "credit":
        return amount
    raise UnsupportedCashStateError(f"Event {event_id} direction {direction!r} has no signed cash amount")


def classify_cash_treatment(event: FinancialEvent) -> CashTreatment:
    if event.status == "failed":
        return CashTreatment.EXCLUDED_FAILED
    if event.status == "cancelled":
        return CashTreatment.EXCLUDED_CANCELLED
    if event.status == "unrealized" or event.direction == "non_cash":
        return CashTreatment.EXCLUDED_UNREALIZED_NON_CASH
    if event.amount is None:
        return CashTreatment.MISSING_AMOUNT_REQUIRES_EVIDENCE
    if event.status == "settled":
        return CashTreatment.SETTLED_CASH_FLOW
    if event.status == "scheduled":
        return CashTreatment.SCHEDULED_CASH_FLOW
    if event.status == "pending" and event.direction == "debit":
        return CashTreatment.PENDING_DEBIT_RESERVATION
    if event.status == "pending" and event.direction == "credit":
        return CashTreatment.PENDING_CREDIT_UNAVAILABLE
    raise UnsupportedCashStateError(
        f"Unsupported event cash state for {event.event_id}: status={event.status!r}, direction={event.direction!r}"
    )


def treatment_flags(treatment: CashTreatment) -> tuple[bool, bool, bool, bool, str | None]:
    if treatment in {CashTreatment.SETTLED_CASH_FLOW, CashTreatment.SCHEDULED_CASH_FLOW}:
        return (True, True, False, False, None)
    if treatment == CashTreatment.PENDING_DEBIT_RESERVATION:
        return (True, False, True, False, None)
    if treatment == CashTreatment.PENDING_CREDIT_UNAVAILABLE:
        return (True, False, False, False, "pending_credit_not_available")
    if treatment == CashTreatment.MISSING_AMOUNT_REQUIRES_EVIDENCE:
        return (True, False, False, True, "missing_amount_requires_evidence")
    if treatment == CashTreatment.EXCLUDED_FAILED:
        return (False, False, False, False, "failed")
    if treatment == CashTreatment.EXCLUDED_CANCELLED:
        return (False, False, False, False, "cancelled")
    if treatment == CashTreatment.EXCLUDED_UNREALIZED_NON_CASH:
        return (False, False, False, False, "unrealized_or_non_cash")
    raise UnsupportedCashStateError(f"Unsupported cash treatment {treatment!r}")


def build_linked_child_ids(dataset: Dataset) -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = {}
    for event in dataset.financial_events:
        if event.linked_event_id is not None:
            children.setdefault(event.linked_event_id, []).append(event.event_id)
    return {event_id: tuple(sorted(child_ids)) for event_id, child_ids in children.items()}


def normalize_event(event: FinancialEvent, dataset: Dataset, linked_child_ids: dict[str, tuple[str, ...]]) -> NormalizedCashEvent:
    profile = dataset.indexes.profiles_by_user_id[event.user_id]
    treatment = classify_cash_treatment(event)
    is_candidate, affects_available_cash, reserves_cash, requires_evidence, exclusion_reason = treatment_flags(treatment)
    cash_date = event.settlement_date or event.event_date

    conversion: CurrencyConversion | None = None
    home_amount: Decimal | None = None
    signed_home_amount: Decimal | None = None
    if event.amount is not None and is_candidate and not requires_evidence:
        conversion = convert_currency(event.amount, event.currency, profile.home_currency, cash_date, dataset)
        home_amount = conversion.converted_amount
        if event.direction in {"debit", "credit"}:
            signed_home_amount = signed_amount_for_direction(home_amount, event.direction, event.event_id)

    return NormalizedCashEvent(
        normalized_event_id=event.event_id,
        source_event_id=event.event_id,
        provenance_event_ids=(event.event_id,),
        user_id=event.user_id,
        home_currency=profile.home_currency,
        event_type=event.event_type,
        category=event.category,
        direction=event.direction,
        status=event.status,
        flexibility=event.flexibility,
        event_date=event.event_date,
        settlement_date=event.settlement_date,
        cash_date=cash_date,
        original_amount=event.amount,
        original_currency=event.currency,
        home_amount=home_amount,
        signed_home_amount=signed_home_amount,
        cash_treatment=treatment,
        is_cash_flow_candidate=is_candidate,
        affects_available_cash=affects_available_cash,
        reserves_cash=reserves_cash,
        requires_amount_evidence=requires_evidence,
        linked_event_id=event.linked_event_id,
        linked_child_event_ids=linked_child_ids.get(event.event_id, ()),
        conversion=conversion,
        exclusion_reason=exclusion_reason,
        source_event=event,
    )


def group_normalized_by_user(events: tuple[NormalizedCashEvent, ...]) -> dict[str, tuple[NormalizedCashEvent, ...]]:
    grouped: dict[str, list[NormalizedCashEvent]] = {}
    for event in events:
        grouped.setdefault(event.user_id, []).append(event)
    return {user_id: tuple(items) for user_id, items in grouped.items()}


def group_normalized_by_linked_parent(events: tuple[NormalizedCashEvent, ...]) -> dict[str, tuple[NormalizedCashEvent, ...]]:
    grouped: dict[str, list[NormalizedCashEvent]] = {}
    for event in events:
        if event.linked_event_id is not None:
            grouped.setdefault(event.linked_event_id, []).append(event)
    return {event_id: tuple(items) for event_id, items in grouped.items()}


def normalize_financial_events(dataset: Dataset) -> NormalizedEventSet:
    linked_child_ids = build_linked_child_ids(dataset)
    normalized_events = tuple(
        normalize_event(event, dataset, linked_child_ids)
        for event in sorted(dataset.financial_events, key=lambda item: item.event_id)
    )
    by_event_id = {event.source_event_id: event for event in normalized_events}
    if len(by_event_id) != len(normalized_events):
        raise DataValidationError("Duplicate normalized event ids")
    return NormalizedEventSet(
        events=normalized_events,
        by_event_id=by_event_id,
        by_user_id=group_normalized_by_user(normalized_events),
        by_linked_event_id=group_normalized_by_linked_parent(normalized_events),
        linked_child_ids_by_event_id=linked_child_ids,
    )


def load_normalized_event_set(dataset_dir: str | None = None) -> NormalizedEventSet:
    return normalize_financial_events(load_dataset(dataset_dir))
