"""Deterministic financial-state resolution for Buy or Wait? Level 2B.

This module consumes L1A typed events, L1B normalized cash semantics, and L2A
structured evidence claims. It resolves only explicit lifecycle and field-level
state. It does not infer recurrences, forecast balances, generate plans, or make
recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from typing import Any

from data_layer import Dataset, FinancialEvent
from event_normalization import NormalizedCashEvent, build_linked_child_ids, normalize_event
from evidence_schema import EvidenceClaim


@dataclass(frozen=True)
class ResolutionTrace:
    event_id: str | None
    field: str
    previous_value: str | None
    resolved_value: str | None
    source_type: str
    source_id: str
    rule: str
    reason: str


@dataclass(frozen=True)
class UnresolvedEvidence:
    claim: EvidenceClaim
    reason: str


@dataclass(frozen=True)
class ResolvedEvent:
    event_id: str
    source_event: FinancialEvent
    resolved_event: FinancialEvent
    normalized_event: NormalizedCashEvent
    active: bool
    inactive_reason: str | None
    superseded_by_event_id: str | None
    provenance_event_ids: tuple[str, ...]
    traces: tuple[ResolutionTrace, ...]


@dataclass(frozen=True)
class ResolvedState:
    events: tuple[ResolvedEvent, ...]
    effective_events: tuple[ResolvedEvent, ...]
    by_event_id: dict[str, ResolvedEvent]
    traces: tuple[ResolutionTrace, ...]
    unresolved_evidence: tuple[UnresolvedEvidence, ...]


def value_to_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def trace_for_event(
    event_id: str | None,
    field: str,
    previous_value: object,
    resolved_value: object,
    source_type: str,
    source_id: str,
    rule: str,
    reason: str,
) -> ResolutionTrace:
    return ResolutionTrace(
        event_id=event_id,
        field=field,
        previous_value=value_to_text(previous_value),
        resolved_value=value_to_text(resolved_value),
        source_type=source_type,
        source_id=source_id,
        rule=rule,
        reason=reason,
    )


def event_sort_key(event: FinancialEvent) -> tuple[date, str]:
    return (event.settlement_date or event.event_date, event.event_id)


def is_separate_cash_movement(parent: FinancialEvent, child: FinancialEvent) -> bool:
    if child.event_type in {"refund", "investment_sale"}:
        return True
    if parent.direction != child.direction and child.status not in {"cancelled", "failed"}:
        return True
    return False


def is_same_lifecycle_context(parent: FinancialEvent, child: FinancialEvent) -> bool:
    return (
        parent.user_id == child.user_id
        and parent.category == child.category
        and parent.direction == child.direction
        and not is_separate_cash_movement(parent, child)
    )


def linked_rule(parent: FinancialEvent, child: FinancialEvent) -> str | None:
    if is_separate_cash_movement(parent, child):
        return None
    if child.status in {"cancelled", "failed"} and parent.user_id == child.user_id:
        return "linked_cancellation_invalidation"
    if not is_same_lifecycle_context(parent, child):
        return None
    if child.status == "settled" and parent.status in {"pending", "scheduled", "failed"}:
        return "linked_settlement_finalization"
    if child.status in {"settled", "pending", "scheduled"} and (
        child.amount != parent.amount
        or child.settlement_date != parent.settlement_date
        or child.event_date != parent.event_date
        or child.status != parent.status
    ):
        return "linked_amendment_replacement"
    return None


def claim_order_key(claim: EvidenceClaim) -> tuple[str, str, str]:
    return (claim.source_type, claim.source_id, claim.claim_id)


def field_source_rank(claim: EvidenceClaim) -> tuple[str, str, str]:
    return claim_order_key(claim)


def is_amount_claim(claim: EvidenceClaim) -> bool:
    return claim.amount is not None and claim.currency is not None and claim.claim_type in {
        "missing_amount",
        "amount_confirmation",
        "amount_amendment",
        "payroll_change",
        "salary_change",
        "pending_payment_clarification",
        "future_payment_clarification",
    }


def amount_role_score(event: FinancialEvent, claim: EvidenceClaim) -> int:
    role = claim.document_role
    if role is None:
        return 0
    if event.direction == "credit" and event.category == "salary":
        if role in {"net_pay", "transferred_amount"}:
            return 100
        if role in {"total_earnings", "base_salary"}:
            return 40
        if role == "total_deductions":
            return -10
    if event.direction == "credit":
        if role in {"net_pay", "transferred_amount", "amount_received"}:
            return 90
        if role in {"total_amount", "total_earnings"}:
            return 50
    if event.direction == "debit":
        if event.status == "settled" and role in {"amount_paid", "amount_received", "transferred_amount"}:
            return 100
        if event.status in {"pending", "scheduled"} and role in {"balance_due", "outstanding_amount", "total_amount"}:
            return 100
        if role in {"total_amount", "balance_due", "outstanding_amount"}:
            return 80
        if role in {"amount_paid", "amount_received"}:
            return 70
    return 0


def choose_amount_claim(event: FinancialEvent, claims: tuple[EvidenceClaim, ...]) -> tuple[EvidenceClaim | None, str, str]:
    candidates = tuple(claim for claim in claims if is_amount_claim(claim))
    if not candidates:
        return None, "no_amount_claim", "No amount claim was compatible with this event."

    role_specific = tuple(claim for claim in candidates if claim.document_role is not None)
    if role_specific:
        candidates = role_specific

    scored = [(amount_role_score(event, claim), claim) for claim in candidates]
    max_score = max(score for score, _ in scored)
    if role_specific and max_score <= 0:
        return None, "incompatible_with_event_semantics", "Role-specific document facts were incompatible with the linked event context."
    best = [claim for score, claim in scored if score == max_score]
    if len(best) == 1:
        return best[0], "role_specific_amount_resolution", "Selected the best document role for the event context."

    amounts = {claim.amount for claim in best}
    if len(amounts) == 1:
        return sorted(best, key=claim_order_key)[0], "equivalent_amount_tie_break", "Multiple claims had the same amount; deterministic source ordering used."

    if event.direction == "credit":
        selected = sorted(best, key=lambda claim: (claim.amount or Decimal("0"), claim_order_key(claim)))[0]
        return selected, "conservative_inbound_amount", "Multiple plausible inbound amounts; selected the lower amount to avoid overstating incoming cash."
    if event.direction == "debit":
        selected = sorted(best, key=lambda claim: (-(claim.amount or Decimal("0")), claim_order_key(claim)))[0]
        return selected, "conservative_outbound_amount", "Multiple plausible outbound amounts; selected the higher amount to avoid understating obligations."
    return None, "incompatible_direction", "Amount evidence was not applied to a non-cash event."


def apply_field_update(
    event: FinancialEvent,
    traces: list[ResolutionTrace],
    *,
    field: str,
    value: object,
    claim: EvidenceClaim,
    rule: str,
    reason: str,
) -> FinancialEvent:
    old_value = getattr(event, field)
    if value is None or value == old_value:
        return event
    traces.append(
        trace_for_event(
            event.event_id,
            field,
            old_value,
            value,
            claim.source_type,
            claim.source_id,
            rule,
            reason,
        )
    )
    return replace(event, **{field: value})


def apply_evidence_to_event(
    event: FinancialEvent,
    claims: tuple[EvidenceClaim, ...],
    unresolved: list[UnresolvedEvidence],
) -> tuple[FinancialEvent, tuple[ResolutionTrace, ...]]:
    traces: list[ResolutionTrace] = []
    sorted_claims = tuple(sorted(claims, key=field_source_rank))
    current = event

    status_claims = [claim for claim in sorted_claims if claim.status and claim.claim_type in {"status_confirmation", "cancellation", "invalidation"}]
    if status_claims:
        claim = status_claims[-1]
        status = "cancelled" if claim.claim_type in {"cancellation", "invalidation"} else claim.status
        current = apply_field_update(
            current,
            traces,
            field="status",
            value=status,
            claim=claim,
            rule="explicit_status_or_cancellation",
            reason="Applied explicit status, cancellation, or invalidation claim.",
        )

    date_claims = [claim for claim in sorted_claims if claim.effective_date is not None and claim.claim_type in {"date_confirmation", "date_amendment", "status_confirmation", "amount_amendment"}]
    if date_claims:
        claim = date_claims[-1]
        field = "settlement_date" if current.settlement_date is not None else "event_date"
        current = apply_field_update(
            current,
            traces,
            field=field,
            value=claim.effective_date,
            claim=claim,
            rule="explicit_date_amendment",
            reason="Applied explicit date confirmation or amendment.",
        )

    should_resolve_amount = current.amount is None or any(claim.claim_type in {"missing_amount", "amount_amendment"} for claim in sorted_claims)
    if should_resolve_amount:
        amount_claim, rule, reason = choose_amount_claim(current, sorted_claims)
        if amount_claim is not None:
            current = apply_field_update(
                current,
                traces,
                field="amount",
                value=amount_claim.amount,
                claim=amount_claim,
                rule=rule,
                reason=reason,
            )
            if amount_claim.currency is not None:
                current = apply_field_update(
                    current,
                    traces,
                    field="currency",
                    value=amount_claim.currency,
                    claim=amount_claim,
                    rule="explicit_amount_currency",
                    reason="Applied currency from the selected amount claim.",
                )
        elif current.amount is None:
            for claim in sorted_claims:
                unresolved.append(UnresolvedEvidence(claim, reason))
    else:
        for claim in sorted_claims:
            if claim.amount is None and claim.claim_type in {"amount_confirmation", "amount_amendment", "missing_amount"}:
                unresolved.append(UnresolvedEvidence(claim, "null_claim_did_not_overwrite_known_value"))

    return current, tuple(traces)


def replacement_dataset_for_events(dataset: Dataset, events: tuple[FinancialEvent, ...]) -> Dataset:
    by_id = {event.event_id: event for event in events}
    merged_events = tuple(by_id.get(event.event_id, event) for event in dataset.financial_events)
    extras = tuple(event for event in events if event.event_id not in {source.event_id for source in dataset.financial_events})
    # Reuse existing indexes for profile/rate lookups; normalize_event only requires profiles.
    return replace(dataset, financial_events=merged_events + extras)


def normalize_resolved_event(
    event: FinancialEvent,
    dataset: Dataset,
    linked_child_ids: dict[str, tuple[str, ...]],
) -> NormalizedCashEvent:
    normalized = normalize_event(event, dataset, linked_child_ids)
    if event.status == "pending" and event.direction == "debit" and event.amount is None:
        return replace(normalized, reserves_cash=True)
    return normalized


def resolve_financial_state(dataset: Dataset, evidence_claims: tuple[EvidenceClaim, ...] = ()) -> ResolvedState:
    original_events = {event.event_id: event for event in dataset.financial_events}
    active = {event.event_id: True for event in dataset.financial_events}
    inactive_reason: dict[str, str | None] = {event.event_id: None for event in dataset.financial_events}
    superseded_by: dict[str, str | None] = {event.event_id: None for event in dataset.financial_events}
    traces_by_event: dict[str, list[ResolutionTrace]] = {event.event_id: [] for event in dataset.financial_events}
    unresolved: list[UnresolvedEvidence] = []

    for child in sorted((event for event in dataset.financial_events if event.linked_event_id), key=event_sort_key):
        parent = original_events.get(child.linked_event_id or "")
        if parent is None:
            continue
        rule = linked_rule(parent, child)
        if rule is None:
            continue
        active[parent.event_id] = False
        inactive_reason[parent.event_id] = rule
        superseded_by[parent.event_id] = child.event_id
        traces_by_event[parent.event_id].append(
            trace_for_event(
                parent.event_id,
                "active",
                True,
                False,
                "linked_event",
                child.event_id,
                rule,
                "Linked event explicitly resolves the earlier lifecycle record.",
            )
        )

    claims_by_event: dict[str, list[EvidenceClaim]] = {}
    for claim in evidence_claims:
        if claim.related_event_id is None:
            unresolved.append(UnresolvedEvidence(claim, "requires_recurrence_or_payroll_matching"))
            continue
        if claim.related_event_id not in original_events:
            unresolved.append(UnresolvedEvidence(claim, "unknown_related_event_id"))
            continue
        claims_by_event.setdefault(claim.related_event_id, []).append(claim)

    resolved_events_by_id: dict[str, FinancialEvent] = {}
    for event in dataset.financial_events:
        updated, traces = apply_evidence_to_event(event, tuple(claims_by_event.get(event.event_id, ())), unresolved)
        resolved_events_by_id[event.event_id] = updated
        traces_by_event[event.event_id].extend(traces)

    resolved_event_tuple = tuple(resolved_events_by_id[event.event_id] for event in dataset.financial_events)
    normalization_dataset = replacement_dataset_for_events(dataset, resolved_event_tuple)
    linked_child_ids = build_linked_child_ids(normalization_dataset)

    resolved_records: list[ResolvedEvent] = []
    for event in resolved_event_tuple:
        normalized = normalize_resolved_event(event, normalization_dataset, linked_child_ids)
        trace_tuple = tuple(traces_by_event[event.event_id])
        resolved_records.append(
            ResolvedEvent(
                event_id=event.event_id,
                source_event=original_events[event.event_id],
                resolved_event=event,
                normalized_event=normalized,
                active=active[event.event_id],
                inactive_reason=inactive_reason[event.event_id],
                superseded_by_event_id=superseded_by[event.event_id],
                provenance_event_ids=(event.event_id,),
                traces=trace_tuple,
            )
        )

    records = tuple(sorted(resolved_records, key=lambda item: event_sort_key(item.resolved_event)))
    traces = tuple(trace for record in records for trace in record.traces)
    return ResolvedState(
        events=records,
        effective_events=tuple(record for record in records if record.active),
        by_event_id={record.event_id: record for record in records},
        traces=traces,
        unresolved_evidence=tuple(unresolved),
    )
