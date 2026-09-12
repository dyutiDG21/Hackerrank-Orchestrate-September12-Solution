"""Deterministic recurrence inference for Buy or Wait? Level 2C.

This module consumes L2B resolved state and produces inspectable recurring
series models for later forecasting. It does not simulate balances, generate
future cash events, recommend payment plans, or write output.csv.
"""

from __future__ import annotations

import calendar
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal

from data_layer import FinancialEvent
from evidence_schema import EvidenceClaim
from financial_state_resolution import ResolvedEvent, ResolvedState, UnresolvedEvidence


SUPPORTED_CADENCES = {"weekly", "biweekly", "monthly"}
MIN_SERIES_OCCURRENCES = 3
PAYROLL_CLAIM_TYPES = {"salary_change", "payroll_change", "status_confirmation"}
PAYROLL_AMOUNT_ROLES = {None, "base_salary", "total_earnings", "net_pay", "transferred_amount"}


@dataclass(frozen=True)
class SeriesGroupingKey:
    user_id: str
    direction: str
    event_type: str
    category: str
    description_key: str
    currency: str


@dataclass(frozen=True)
class AmountObservation:
    event_id: str
    cash_date: date
    amount: Decimal
    currency: str
    home_amount: Decimal | None
    status: str


@dataclass(frozen=True)
class ExplicitOccurrence:
    event_id: str
    cash_date: date
    amount: Decimal
    currency: str
    status: str


@dataclass(frozen=True)
class CadenceEvidence:
    cadence: str | None
    observed_dates: tuple[date, ...]
    observed_gaps_days: tuple[int, ...]
    anchor_day: int | None
    end_of_month_anchor: bool
    confidence: str
    projectable: bool
    reason: str


@dataclass(frozen=True)
class AmountModel:
    amount_behavior: str
    observed_amounts: tuple[AmountObservation, ...]
    latest_observed_amount: Decimal | None
    latest_observed_currency: str | None


@dataclass(frozen=True)
class RecurringEvidenceAmendment:
    claim: EvidenceClaim
    field: str
    amount: Decimal | None
    currency: str | None
    effective_date: date | None
    status: str | None
    rule: str
    reason: str


@dataclass(frozen=True)
class RecurringSeries:
    series_id: str
    grouping_key: SeriesGroupingKey
    member_event_ids: tuple[str, ...]
    provenance_event_ids: tuple[str, ...]
    cadence_evidence: CadenceEvidence
    amount_model: AmountModel
    fixed_or_variable: str
    projectable: bool
    projectability_reason: str
    explicit_occurrences: tuple[ExplicitOccurrence, ...]
    evidence_amendments: tuple[RecurringEvidenceAmendment, ...]
    unresolved_ambiguities: tuple[str, ...]


@dataclass(frozen=True)
class RecurrenceInferenceResult:
    series: tuple[RecurringSeries, ...]
    by_series_id: dict[str, RecurringSeries]
    unresolved_evidence: tuple[UnresolvedEvidence, ...]


def normalize_description(description: str) -> str:
    return re.sub(r"\s+", " ", description.strip().lower())


def is_last_day_of_month(value: date) -> bool:
    return value.day == calendar.monthrange(value.year, value.month)[1]


def add_months_anchored(start: date, months: int, *, anchor_day: int, end_of_month: bool) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = last_day if end_of_month else min(anchor_day, last_day)
    return date(year, month, day)


def observed_gaps(dates: tuple[date, ...]) -> tuple[int, ...]:
    return tuple((dates[index] - dates[index - 1]).days for index in range(1, len(dates)))


def infer_cadence(dates: tuple[date, ...]) -> CadenceEvidence:
    sorted_dates = tuple(sorted(dates))
    gaps = observed_gaps(sorted_dates)
    if len(set(sorted_dates)) != len(sorted_dates):
        return CadenceEvidence(None, sorted_dates, gaps, None, False, "none", False, "duplicate_dates_in_group")
    if len(sorted_dates) < MIN_SERIES_OCCURRENCES:
        return CadenceEvidence(None, sorted_dates, gaps, None, False, "none", False, "insufficient_occurrences")

    confidence = "high" if len(sorted_dates) >= 4 else "medium"
    if gaps and all(gap == 7 for gap in gaps):
        return CadenceEvidence("weekly", sorted_dates, gaps, None, False, confidence, True, "exact_weekly_spacing")
    if gaps and all(gap == 14 for gap in gaps):
        return CadenceEvidence("biweekly", sorted_dates, gaps, None, False, confidence, True, "exact_biweekly_spacing")

    first = sorted_dates[0]
    end_of_month = is_last_day_of_month(first)
    expected_monthly = tuple(
        add_months_anchored(first, index, anchor_day=first.day, end_of_month=end_of_month)
        for index in range(len(sorted_dates))
    )
    if expected_monthly == sorted_dates:
        reason = "calendar_month_end_spacing" if end_of_month else "calendar_month_anchor_spacing"
        return CadenceEvidence("monthly", sorted_dates, gaps, first.day, end_of_month, confidence, True, reason)

    return CadenceEvidence(None, sorted_dates, gaps, None, False, "none", False, "ambiguous_or_irregular_spacing")


def record_is_recurrence_candidate(record: ResolvedEvent) -> bool:
    event = record.resolved_event
    normalized = record.normalized_event
    return (
        record.active
        and normalized.is_cash_flow_candidate
        and event.direction in {"debit", "credit"}
        and event.status in {"settled", "scheduled", "pending"}
        and event.amount is not None
        and bool(normalize_description(event.description))
    )


def grouping_key_for_event(event: FinancialEvent) -> SeriesGroupingKey:
    return SeriesGroupingKey(
        user_id=event.user_id,
        direction=event.direction,
        event_type=event.event_type,
        category=event.category,
        description_key=normalize_description(event.description),
        currency=event.currency,
    )


def stable_series_id(key: SeriesGroupingKey, index: int) -> str:
    payload = "|".join((key.user_id, key.direction, key.event_type, key.category, key.description_key, key.currency))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"series_{index:04d}_{digest}"


def build_amount_model(records: tuple[ResolvedEvent, ...]) -> AmountModel:
    observations = tuple(
        AmountObservation(
            event_id=record.event_id,
            cash_date=record.normalized_event.cash_date,
            amount=record.resolved_event.amount,  # type: ignore[arg-type]
            currency=record.resolved_event.currency,
            home_amount=record.normalized_event.home_amount,
            status=record.resolved_event.status,
        )
        for record in sorted(records, key=lambda item: (item.normalized_event.cash_date, item.event_id))
        if record.resolved_event.amount is not None
    )
    distinct_amounts = {observation.amount for observation in observations}
    behavior = "fixed" if len(distinct_amounts) == 1 else "variable"
    latest = observations[-1] if observations else None
    return AmountModel(
        amount_behavior=behavior,
        observed_amounts=observations,
        latest_observed_amount=latest.amount if latest is not None else None,
        latest_observed_currency=latest.currency if latest is not None else None,
    )


def build_series(records: tuple[ResolvedEvent, ...], key: SeriesGroupingKey, index: int) -> RecurringSeries:
    sorted_records = tuple(sorted(records, key=lambda item: (item.normalized_event.cash_date, item.event_id)))
    dates = tuple(record.normalized_event.cash_date for record in sorted_records)
    cadence = infer_cadence(dates)
    amount_model = build_amount_model(sorted_records)
    explicit_occurrences = tuple(
        ExplicitOccurrence(
            event_id=record.event_id,
            cash_date=record.normalized_event.cash_date,
            amount=record.resolved_event.amount,  # type: ignore[arg-type]
            currency=record.resolved_event.currency,
            status=record.resolved_event.status,
        )
        for record in sorted_records
        if record.resolved_event.amount is not None
    )
    ambiguities = () if cadence.projectable else (cadence.reason,)
    return RecurringSeries(
        series_id=stable_series_id(key, index),
        grouping_key=key,
        member_event_ids=tuple(record.event_id for record in sorted_records),
        provenance_event_ids=tuple(
            event_id
            for record in sorted_records
            for event_id in record.provenance_event_ids
        ),
        cadence_evidence=cadence,
        amount_model=amount_model,
        fixed_or_variable=amount_model.amount_behavior,
        projectable=cadence.projectable,
        projectability_reason=cadence.reason,
        explicit_occurrences=explicit_occurrences,
        evidence_amendments=(),
        unresolved_ambiguities=ambiguities,
    )


def build_candidate_series(resolved_state: ResolvedState) -> tuple[RecurringSeries, ...]:
    grouped: dict[SeriesGroupingKey, list[ResolvedEvent]] = {}
    for record in resolved_state.effective_events:
        if record_is_recurrence_candidate(record):
            grouped.setdefault(grouping_key_for_event(record.resolved_event), []).append(record)

    candidates = [
        (key, tuple(records))
        for key, records in grouped.items()
        if len(records) >= MIN_SERIES_OCCURRENCES
    ]
    candidates.sort(key=lambda item: (
        item[0].user_id,
        item[0].direction,
        item[0].event_type,
        item[0].category,
        item[0].description_key,
        item[0].currency,
    ))
    return tuple(build_series(records, key, index + 1) for index, (key, records) in enumerate(candidates))


def claim_order_key(claim: EvidenceClaim) -> tuple[str, str, str]:
    return (claim.source_type, claim.source_id, claim.claim_id)


def is_payroll_semantic_claim(claim: EvidenceClaim) -> bool:
    return (
        claim.claim_type in PAYROLL_CLAIM_TYPES
        or claim.identifier_role == "payroll_reference"
        or (claim.document_role is not None and claim.document_role in PAYROLL_AMOUNT_ROLES)
    )


def payroll_compatible(series: RecurringSeries, claim: EvidenceClaim) -> bool:
    key = series.grouping_key
    if key.user_id != claim.user_id:
        return False
    if not (key.direction == "credit" and key.event_type == "income" and key.category == "salary"):
        return False
    if claim.currency is not None and claim.currency != key.currency:
        return False
    if claim.document_role not in PAYROLL_AMOUNT_ROLES:
        return False
    return True


def semantically_compatible_series(series: RecurringSeries, claim: EvidenceClaim) -> bool:
    if is_payroll_semantic_claim(claim):
        return payroll_compatible(series, claim)
    return False


def unresolved_reason_for_unmatched_claim(series: tuple[RecurringSeries, ...], claim: EvidenceClaim) -> str:
    same_user_payroll = tuple(
        item
        for item in series
        if item.grouping_key.user_id == claim.user_id
        and item.grouping_key.direction == "credit"
        and item.grouping_key.event_type == "income"
        and item.grouping_key.category == "salary"
    )
    if claim.currency is not None and same_user_payroll and all(item.grouping_key.currency != claim.currency for item in same_user_payroll):
        return "currency_incompatible_with_candidate_series"
    return "no_semantically_compatible_recurring_series"


def amendment_for_claim(claim: EvidenceClaim) -> RecurringEvidenceAmendment | None:
    if claim.claim_type in {"salary_change", "payroll_change"} and claim.amount is not None:
        if claim.effective_date is None:
            return None
        return RecurringEvidenceAmendment(
            claim=claim,
            field="amount",
            amount=claim.amount,
            currency=claim.currency,
            effective_date=claim.effective_date,
            status=None,
            rule="unique_payroll_series_salary_change",
            reason="Applied salary/payroll change prospectively to the recurring series model only.",
        )
    if claim.claim_type == "status_confirmation" and claim.status is not None:
        return RecurringEvidenceAmendment(
            claim=claim,
            field="status",
            amount=None,
            currency=None,
            effective_date=claim.effective_date,
            status=claim.status,
            rule="unique_payroll_series_status_confirmation",
            reason="Attached payroll status evidence to the uniquely matched recurring series.",
        )
    if claim.claim_type in {"date_confirmation", "date_amendment"} and claim.effective_date is not None:
        return RecurringEvidenceAmendment(
            claim=claim,
            field="future_occurrence_date",
            amount=None,
            currency=None,
            effective_date=claim.effective_date,
            status=None,
            rule="unique_series_date_evidence",
            reason="Attached date evidence to the unique recurring series without generating an occurrence.",
        )
    return None


def attach_unlinked_evidence(
    series: tuple[RecurringSeries, ...],
    unresolved_from_l2b: tuple[UnresolvedEvidence, ...],
) -> tuple[tuple[RecurringSeries, ...], tuple[UnresolvedEvidence, ...]]:
    by_id = {item.series_id: item for item in series}
    unresolved: list[UnresolvedEvidence] = []

    for unresolved_item in sorted(unresolved_from_l2b, key=lambda item: claim_order_key(item.claim)):
        claim = unresolved_item.claim
        if unresolved_item.reason != "requires_recurrence_or_payroll_matching" or claim.related_event_id is not None:
            unresolved.append(unresolved_item)
            continue

        candidates = tuple(item for item in series if item.projectable and semantically_compatible_series(item, claim))
        if not candidates:
            unresolved.append(UnresolvedEvidence(claim, unresolved_reason_for_unmatched_claim(series, claim)))
            continue
        if len(candidates) > 1:
            unresolved.append(UnresolvedEvidence(claim, "ambiguous_recurring_series_match"))
            continue

        amendment = amendment_for_claim(claim)
        if amendment is None:
            reason = "recurring_change_missing_effective_date" if claim.amount is not None else "no_projectable_recurring_change_payload"
            unresolved.append(UnresolvedEvidence(claim, reason))
            continue

        matched = candidates[0]
        by_id[matched.series_id] = replace(
            matched,
            evidence_amendments=matched.evidence_amendments + (amendment,),
        )

    ordered = tuple(by_id[item.series_id] for item in series)
    return ordered, tuple(unresolved)


def infer_recurring_series(resolved_state: ResolvedState) -> RecurrenceInferenceResult:
    base_series = build_candidate_series(resolved_state)
    series, unresolved = attach_unlinked_evidence(base_series, resolved_state.unresolved_evidence)
    return RecurrenceInferenceResult(
        series=series,
        by_series_id={item.series_id: item for item in series},
        unresolved_evidence=unresolved,
    )


def recurrence_diagnostics(result: RecurrenceInferenceResult) -> dict[str, object]:
    cadence_counts = Counter(item.cadence_evidence.cadence or "ambiguous" for item in result.series)
    projectable_counts = Counter("projectable" if item.projectable else "ambiguous" for item in result.series)
    length_counts = Counter(len(item.member_event_ids) for item in result.series)
    unresolved_counts = Counter(item.reason for item in result.unresolved_evidence)
    suspicious = tuple(
        {
            "series_id": item.series_id,
            "user_id": item.grouping_key.user_id,
            "category": item.grouping_key.category,
            "description": item.grouping_key.description_key,
            "cadence": item.cadence_evidence.cadence,
            "reason": item.projectability_reason,
            "length": len(item.member_event_ids),
        }
        for item in result.series
        if not item.projectable or item.cadence_evidence.confidence != "high"
    )[:10]
    return {
        "series_count": len(result.series),
        "cadence_counts": dict(sorted(cadence_counts.items())),
        "projectable_counts": dict(sorted(projectable_counts.items())),
        "series_length_distribution": dict(sorted(length_counts.items())),
        "unresolved_evidence_counts": dict(sorted(unresolved_counts.items())),
        "suspicious_or_low_confidence_examples": suspicious,
    }
