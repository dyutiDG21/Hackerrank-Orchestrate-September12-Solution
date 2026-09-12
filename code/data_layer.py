"""Canonical dataset ingestion for the Buy or Wait? challenge.

This module intentionally stops at deterministic parsing, validation, and
indexing. It does not infer recurrences, resolve lifecycle conflicts, interpret
evidence, forecast balances, recommend payment plans, or write predictions.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable, Mapping, TypeVar


CsvRow = dict[str, str]
T = TypeVar("T")


EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "financial_profiles.csv": (
        "user_id",
        "home_currency",
        "current_available_balance",
        "minimum_balance_to_keep",
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        "max_installment_months",
    ),
    "financial_events.csv": (
        "event_id",
        "user_id",
        "event_type",
        "description",
        "category",
        "direction",
        "amount",
        "currency",
        "event_date",
        "settlement_date",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    ),
    "exchange_rates.csv": ("rate_date", "from_currency", "to_currency", "rate"),
    "requests.csv": (
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
    ),
    "sample_requests.csv": (
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ),
    "request_payment_options.csv": (
        "payment_option_id",
        "request_id",
        "payment_method",
        "payment_amount",
        "number_of_payments",
        "first_payment_date",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
    ),
    "messages.csv": ("message_id", "user_id", "request_id", "related_event_id", "sent_at", "source_type", "message_text"),
    "images.csv": ("image_id", "user_id", "request_id", "related_event_id"),
    "output.csv": (
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ),
}


class DataValidationError(ValueError):
    """Raised when the supplied participant-facing dataset violates loader assumptions."""


def is_missing(value: object) -> bool:
    return value is None or str(value).strip() == ""


def none_if_blank(value: str | None) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def parse_decimal(value: str | None, *, field: str) -> Decimal | None:
    text = none_if_blank(value)
    if text is None:
        return None
    try:
        return Decimal(text)
    except Exception as exc:  # decimal raises several subclasses depending on context.
        raise DataValidationError(f"Invalid decimal for {field}: {value!r}") from exc


def parse_int(value: str | None, *, field: str) -> int | None:
    text = none_if_blank(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise DataValidationError(f"Invalid integer for {field}: {value!r}") from exc


def parse_date(value: str | None, *, field: str) -> date | None:
    text = none_if_blank(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise DataValidationError(f"Invalid date for {field}: {value!r}") from exc


def parse_timestamp(value: str | None, *, field: str) -> datetime | None:
    text = none_if_blank(value)
    if text is None:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DataValidationError(f"Invalid timestamp for {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_bool(value: str | None, *, field: str) -> bool | None:
    text = none_if_blank(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise DataValidationError(f"Invalid boolean for {field}: {value!r}")


def parse_pipe_list(value: str | None) -> tuple[str, ...]:
    text = none_if_blank(value)
    if text is None:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def require_text(row: CsvRow, field: str) -> str:
    value = none_if_blank(row.get(field))
    if value is None:
        raise DataValidationError(f"Missing required field {field}")
    return value


def read_csv_rows(dataset_dir: Path, filename: str) -> list[CsvRow]:
    path = dataset_dir / filename
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = EXPECTED_COLUMNS[filename]
        actual = tuple(reader.fieldnames or ())
        if actual != expected:
            raise DataValidationError(f"{filename} columns changed: expected {expected}, got {actual}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def unique_by(items: Iterable[T], key: Callable[[T], str], label: str) -> dict[str, T]:
    indexed: dict[str, T] = {}
    duplicates: list[str] = []
    for item in items:
        item_key = key(item)
        if item_key in indexed:
            duplicates.append(item_key)
        indexed[item_key] = item
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise DataValidationError(f"Duplicate {label} ids: {preview}")
    return indexed


def group_by(items: Iterable[T], key: Callable[[T], str | None]) -> dict[str, tuple[T, ...]]:
    grouped: dict[str, list[T]] = defaultdict(list)
    for item in items:
        item_key = key(item)
        if item_key is not None:
            grouped[item_key].append(item)
    return {item_key: tuple(values) for item_key, values in grouped.items()}


@dataclass(frozen=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: int | None
    raw: Mapping[str, str]


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None
    currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str | None
    flexibility: str
    minimum_allowed_amount: Decimal | None
    raw: Mapping[str, str]


@dataclass(frozen=True)
class PurchaseRequest:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    raw: Mapping[str, str]


@dataclass(frozen=True)
class SolvedSampleRequest:
    request: PurchaseRequest
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str | None
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str | None
    decision_explanation: str
    raw: Mapping[str, str]


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal
    raw: Mapping[str, str]


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: datetime
    source_type: str
    message_text: str
    raw: Mapping[str, str]


@dataclass(frozen=True)
class ImageEvidence:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str
    image_path: Path
    raw: Mapping[str, str]


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal
    raw: Mapping[str, str]


@dataclass(frozen=True)
class OutputTemplateRow:
    request_id: str
    raw: Mapping[str, str]


@dataclass(frozen=True)
class DatasetIndexes:
    profiles_by_user_id: dict[str, FinancialProfile]
    events_by_event_id: dict[str, FinancialEvent]
    events_by_user_id: dict[str, tuple[FinancialEvent, ...]]
    events_by_linked_event_id: dict[str, tuple[FinancialEvent, ...]]
    requests_by_request_id: dict[str, PurchaseRequest]
    requests_by_user_id: dict[str, tuple[PurchaseRequest, ...]]
    sample_requests_by_request_id: dict[str, SolvedSampleRequest]
    sample_requests_by_user_id: dict[str, tuple[SolvedSampleRequest, ...]]
    payment_options_by_option_id: dict[str, PaymentOption]
    payment_options_by_request_id: dict[str, tuple[PaymentOption, ...]]
    messages_by_message_id: dict[str, Message]
    messages_by_user_id: dict[str, tuple[Message, ...]]
    messages_by_request_id: dict[str, tuple[Message, ...]]
    messages_by_related_event_id: dict[str, tuple[Message, ...]]
    images_by_image_id: dict[str, ImageEvidence]
    images_by_user_id: dict[str, tuple[ImageEvidence, ...]]
    images_by_request_id: dict[str, tuple[ImageEvidence, ...]]
    images_by_related_event_id: dict[str, tuple[ImageEvidence, ...]]
    exchange_rates_by_key: dict[tuple[date, str, str], ExchangeRate]
    output_template_by_request_id: dict[str, OutputTemplateRow]


@dataclass(frozen=True)
class Dataset:
    dataset_dir: Path
    financial_profiles: tuple[FinancialProfile, ...]
    financial_events: tuple[FinancialEvent, ...]
    requests: tuple[PurchaseRequest, ...]
    sample_requests: tuple[SolvedSampleRequest, ...]
    payment_options: tuple[PaymentOption, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageEvidence, ...]
    exchange_rates: tuple[ExchangeRate, ...]
    output_template: tuple[OutputTemplateRow, ...]
    indexes: DatasetIndexes


def parse_profile(row: CsvRow) -> FinancialProfile:
    current_available_balance = parse_decimal(row["current_available_balance"], field="current_available_balance")
    minimum_balance_to_keep = parse_decimal(row["minimum_balance_to_keep"], field="minimum_balance_to_keep")
    if current_available_balance is None or minimum_balance_to_keep is None:
        raise DataValidationError("Profile balances are required")
    return FinancialProfile(
        user_id=require_text(row, "user_id"),
        home_currency=require_text(row, "home_currency"),
        current_available_balance=current_available_balance,
        minimum_balance_to_keep=minimum_balance_to_keep,
        financial_priorities=parse_pipe_list(row["financial_priorities"]),
        expense_categories_to_protect=parse_pipe_list(row["expense_categories_to_protect"]),
        expense_categories_user_is_willing_to_reduce=parse_pipe_list(row["expense_categories_user_is_willing_to_reduce"]),
        expense_categories_user_is_willing_to_stop=parse_pipe_list(row["expense_categories_user_is_willing_to_stop"]),
        payment_methods_user_will_consider=parse_pipe_list(row["payment_methods_user_will_consider"]),
        max_installment_months=parse_int(row["max_installment_months"], field="max_installment_months"),
        raw=row,
    )


def parse_event(row: CsvRow) -> FinancialEvent:
    event_date = parse_date(row["event_date"], field="event_date")
    if event_date is None:
        raise DataValidationError("Financial event event_date is required")
    return FinancialEvent(
        event_id=require_text(row, "event_id"),
        user_id=require_text(row, "user_id"),
        event_type=require_text(row, "event_type"),
        description=row["description"],
        category=require_text(row, "category"),
        direction=require_text(row, "direction"),
        amount=parse_decimal(row["amount"], field="amount"),
        currency=require_text(row, "currency"),
        event_date=event_date,
        settlement_date=parse_date(row["settlement_date"], field="settlement_date"),
        status=require_text(row, "status"),
        linked_event_id=none_if_blank(row["linked_event_id"]),
        flexibility=require_text(row, "flexibility"),
        minimum_allowed_amount=parse_decimal(row["minimum_allowed_amount"], field="minimum_allowed_amount"),
        raw=row,
    )


def parse_request(row: CsvRow) -> PurchaseRequest:
    request_date = parse_date(row["request_date"], field="request_date")
    desired_completion_date = parse_date(row["desired_completion_date"], field="desired_completion_date")
    requested_amount = parse_decimal(row["requested_amount"], field="requested_amount")
    allows_partial_payment = parse_bool(row["allows_partial_payment"], field="allows_partial_payment")
    if request_date is None or desired_completion_date is None or requested_amount is None or allows_partial_payment is None:
        raise DataValidationError("Request date, amount, completion date, and partial-payment flag are required")
    return PurchaseRequest(
        request_id=require_text(row, "request_id"),
        user_id=require_text(row, "user_id"),
        request_date=request_date,
        request_type=require_text(row, "request_type"),
        requested_amount=requested_amount,
        desired_completion_date=desired_completion_date,
        allows_partial_payment=allows_partial_payment,
        request_text=row["request_text"],
        raw=row,
    )


def parse_sample_request(row: CsvRow) -> SolvedSampleRequest:
    base_request = parse_request(row)
    amount_safe_to_pay = parse_decimal(row["amount_safe_to_pay"], field="amount_safe_to_pay")
    if amount_safe_to_pay is None:
        raise DataValidationError("Sample amount_safe_to_pay is required")
    return SolvedSampleRequest(
        request=base_request,
        amount_safe_to_pay=amount_safe_to_pay,
        affordability_status=require_text(row, "affordability_status"),
        recommended_payment_method=require_text(row, "recommended_payment_method"),
        payment_plan=none_if_blank(row["payment_plan"]),
        earliest_date_for_full_payment=parse_date(row["earliest_date_for_full_payment"], field="earliest_date_for_full_payment"),
        spending_changes_needed=none_if_blank(row["spending_changes_needed"]),
        decision_explanation=row["decision_explanation"],
        raw=row,
    )


def parse_payment_option(row: CsvRow) -> PaymentOption:
    payment_amount = parse_decimal(row["payment_amount"], field="payment_amount")
    number_of_payments = parse_int(row["number_of_payments"], field="number_of_payments")
    first_payment_date = parse_date(row["first_payment_date"], field="first_payment_date")
    financing_fee = parse_decimal(row["financing_fee"], field="financing_fee")
    total_payable_amount = parse_decimal(row["total_payable_amount"], field="total_payable_amount")
    if (
        payment_amount is None
        or number_of_payments is None
        or first_payment_date is None
        or financing_fee is None
        or total_payable_amount is None
    ):
        raise DataValidationError("Payment option amount, count, first date, fee, and total are required")
    return PaymentOption(
        payment_option_id=require_text(row, "payment_option_id"),
        request_id=require_text(row, "request_id"),
        payment_method=require_text(row, "payment_method"),
        payment_amount=payment_amount,
        number_of_payments=number_of_payments,
        first_payment_date=first_payment_date,
        payment_frequency_days=parse_int(row["payment_frequency_days"], field="payment_frequency_days"),
        financing_fee=financing_fee,
        total_payable_amount=total_payable_amount,
        raw=row,
    )


def parse_message(row: CsvRow) -> Message:
    sent_at = parse_timestamp(row["sent_at"], field="sent_at")
    if sent_at is None:
        raise DataValidationError("Message sent_at is required")
    return Message(
        message_id=require_text(row, "message_id"),
        user_id=require_text(row, "user_id"),
        request_id=none_if_blank(row["request_id"]),
        related_event_id=none_if_blank(row["related_event_id"]),
        sent_at=sent_at,
        source_type=require_text(row, "source_type"),
        message_text=row["message_text"],
        raw=row,
    )


def parse_image(row: CsvRow, dataset_dir: Path) -> ImageEvidence:
    image_id = require_text(row, "image_id")
    return ImageEvidence(
        image_id=image_id,
        user_id=require_text(row, "user_id"),
        request_id=require_text(row, "request_id"),
        related_event_id=require_text(row, "related_event_id"),
        image_path=dataset_dir / "media" / "images" / f"{image_id}.png",
        raw=row,
    )


def parse_exchange_rate(row: CsvRow) -> ExchangeRate:
    rate_date = parse_date(row["rate_date"], field="rate_date")
    rate = parse_decimal(row["rate"], field="rate")
    if rate_date is None or rate is None:
        raise DataValidationError("Exchange rate date and rate are required")
    return ExchangeRate(
        rate_date=rate_date,
        from_currency=require_text(row, "from_currency"),
        to_currency=require_text(row, "to_currency"),
        rate=rate,
        raw=row,
    )


def parse_output_template_row(row: CsvRow) -> OutputTemplateRow:
    return OutputTemplateRow(request_id=require_text(row, "request_id"), raw=row)


def validate_references(dataset: Dataset) -> None:
    user_ids = set(dataset.indexes.profiles_by_user_id)
    request_ids = set(dataset.indexes.requests_by_request_id) | set(dataset.indexes.sample_requests_by_request_id)
    event_ids = set(dataset.indexes.events_by_event_id)

    def ensure_known(ref: str, valid: set[str], label: str) -> None:
        if ref not in valid:
            raise DataValidationError(f"Unknown {label}: {ref}")

    for request in dataset.requests:
        ensure_known(request.user_id, user_ids, "request user_id")
    for sample in dataset.sample_requests:
        ensure_known(sample.request.user_id, user_ids, "sample request user_id")
    for event in dataset.financial_events:
        ensure_known(event.user_id, user_ids, "event user_id")
        if event.linked_event_id is not None:
            ensure_known(event.linked_event_id, event_ids, "linked_event_id")
    for option in dataset.payment_options:
        ensure_known(option.request_id, request_ids, "payment option request_id")
    for message in dataset.messages:
        ensure_known(message.user_id, user_ids, "message user_id")
        if message.request_id is not None:
            ensure_known(message.request_id, request_ids, "message request_id")
        if message.related_event_id is not None:
            ensure_known(message.related_event_id, event_ids, "message related_event_id")
    for image in dataset.images:
        ensure_known(image.user_id, user_ids, "image user_id")
        ensure_known(image.request_id, request_ids, "image request_id")
        ensure_known(image.related_event_id, event_ids, "image related_event_id")


def build_indexes(
    profiles: tuple[FinancialProfile, ...],
    events: tuple[FinancialEvent, ...],
    requests: tuple[PurchaseRequest, ...],
    sample_requests: tuple[SolvedSampleRequest, ...],
    payment_options: tuple[PaymentOption, ...],
    messages: tuple[Message, ...],
    images: tuple[ImageEvidence, ...],
    exchange_rates: tuple[ExchangeRate, ...],
    output_template: tuple[OutputTemplateRow, ...],
) -> DatasetIndexes:
    rates_by_key: dict[tuple[date, str, str], ExchangeRate] = {}
    duplicate_rate_keys: list[tuple[date, str, str]] = []
    for rate in exchange_rates:
        key = (rate.rate_date, rate.from_currency, rate.to_currency)
        if key in rates_by_key:
            duplicate_rate_keys.append(key)
        rates_by_key[key] = rate
    if duplicate_rate_keys:
        raise DataValidationError(f"Duplicate exchange-rate keys: {duplicate_rate_keys[:5]}")

    return DatasetIndexes(
        profiles_by_user_id=unique_by(profiles, lambda item: item.user_id, "user"),
        events_by_event_id=unique_by(events, lambda item: item.event_id, "event"),
        events_by_user_id=group_by(events, lambda item: item.user_id),
        events_by_linked_event_id=group_by(events, lambda item: item.linked_event_id),
        requests_by_request_id=unique_by(requests, lambda item: item.request_id, "request"),
        requests_by_user_id=group_by(requests, lambda item: item.user_id),
        sample_requests_by_request_id=unique_by(sample_requests, lambda item: item.request.request_id, "sample request"),
        sample_requests_by_user_id=group_by(sample_requests, lambda item: item.request.user_id),
        payment_options_by_option_id=unique_by(payment_options, lambda item: item.payment_option_id, "payment option"),
        payment_options_by_request_id=group_by(payment_options, lambda item: item.request_id),
        messages_by_message_id=unique_by(messages, lambda item: item.message_id, "message"),
        messages_by_user_id=group_by(messages, lambda item: item.user_id),
        messages_by_request_id=group_by(messages, lambda item: item.request_id),
        messages_by_related_event_id=group_by(messages, lambda item: item.related_event_id),
        images_by_image_id=unique_by(images, lambda item: item.image_id, "image"),
        images_by_user_id=group_by(images, lambda item: item.user_id),
        images_by_request_id=group_by(images, lambda item: item.request_id),
        images_by_related_event_id=group_by(images, lambda item: item.related_event_id),
        exchange_rates_by_key=rates_by_key,
        output_template_by_request_id=unique_by(output_template, lambda item: item.request_id, "output template request"),
    )


def load_dataset(dataset_dir: str | Path | None = None) -> Dataset:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_dataset_dir = Path(dataset_dir).resolve() if dataset_dir is not None else repo_root / "dataset"

    profiles = tuple(parse_profile(row) for row in read_csv_rows(resolved_dataset_dir, "financial_profiles.csv"))
    events = tuple(parse_event(row) for row in read_csv_rows(resolved_dataset_dir, "financial_events.csv"))
    requests = tuple(parse_request(row) for row in read_csv_rows(resolved_dataset_dir, "requests.csv"))
    sample_requests = tuple(parse_sample_request(row) for row in read_csv_rows(resolved_dataset_dir, "sample_requests.csv"))
    payment_options = tuple(
        parse_payment_option(row) for row in read_csv_rows(resolved_dataset_dir, "request_payment_options.csv")
    )
    messages = tuple(parse_message(row) for row in read_csv_rows(resolved_dataset_dir, "messages.csv"))
    images = tuple(parse_image(row, resolved_dataset_dir) for row in read_csv_rows(resolved_dataset_dir, "images.csv"))
    exchange_rates = tuple(parse_exchange_rate(row) for row in read_csv_rows(resolved_dataset_dir, "exchange_rates.csv"))
    output_template = tuple(parse_output_template_row(row) for row in read_csv_rows(resolved_dataset_dir, "output.csv"))

    indexes = build_indexes(
        profiles,
        events,
        requests,
        sample_requests,
        payment_options,
        messages,
        images,
        exchange_rates,
        output_template,
    )
    dataset = Dataset(
        dataset_dir=resolved_dataset_dir,
        financial_profiles=profiles,
        financial_events=events,
        requests=requests,
        sample_requests=sample_requests,
        payment_options=payment_options,
        messages=messages,
        images=images,
        exchange_rates=exchange_rates,
        output_template=output_template,
        indexes=indexes,
    )
    validate_references(dataset)
    return dataset
