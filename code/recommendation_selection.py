"""Level 6 deterministic recommendation selection and output serialization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from data_layer import Dataset, PurchaseRequest
from payment_capacity import PaymentCapacityResult
from payment_planning import PlanCandidate, SpendingChange


OUTPUT_COLUMNS = (
    "request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation",
)
ALLOWED_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
ALLOWED_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


@dataclass(frozen=True)
class FinalDecision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str
    selected_candidate_id: str | None


def format_decimal(value: Decimal) -> str:
    """Plain Decimal notation, retaining meaningful supplied payment precision."""
    rendered = format(value, "f")
    if "." not in rendered:
        return rendered
    return rendered.rstrip("0").rstrip(".") or "0"


def format_plan_decimal(value: Decimal) -> str:
    """Preserve meaningful Decimal scale in supplied payment/change values."""
    return format(value, "f")


def payment_plan_text(candidate: PlanCandidate) -> str:
    return "|".join(f"{leg.payment_date.isoformat()}:{format_plan_decimal(leg.amount)}" for leg in candidate.payment_legs)


def spending_changes_text(changes: tuple[SpendingChange, ...]) -> str:
    if not changes:
        return "none"
    values = []
    for change in changes:
        if change.action == "stop":
            values.append(f"stop:{change.event_id}")
        elif change.action == "reduce_to" and change.new_amount is not None:
            values.append(f"reduce_to:{change.event_id}:{format_plan_decimal(change.new_amount)}")
        else:
            raise ValueError("Invalid L5 spending change for serialization")
    return "|".join(values)


def structural_key(candidate: PlanCandidate) -> tuple[object, ...]:
    return (
        candidate.method, candidate.payment_option_id or "",
        tuple((leg.payment_date, leg.amount) for leg in candidate.payment_legs),
        tuple((item.action, item.event_id, item.new_amount) for item in candidate.spending_changes),
        candidate.candidate_id,
    )


def rank_key(candidate: PlanCandidate) -> tuple[object, ...]:
    """Official order, followed only by a canonical deterministic tie key."""
    return (
        0 if candidate.completes_by_desired_date else 1,
        0 if not candidate.spending_changes else 1,
        candidate.total_paid,
        candidate.start_date,
        len(candidate.payment_legs),
        candidate.payment_option_id or "",
        structural_key(candidate),
    )


def select_candidate(candidates: tuple[PlanCandidate, ...]) -> PlanCandidate | None:
    eligible = tuple(candidate for candidate in candidates if candidate.completes_by_desired_date)
    return min(eligible, key=rank_key) if eligible else None


def explanation(request: PurchaseRequest, capacity: PaymentCapacityResult, candidate: PlanCandidate | None) -> str:
    if candidate is None:
        return "No eligible plan completes the request by the deadline while preserving the required minimum balance."
    changes = spending_changes_text(candidate.spending_changes)
    prefix = "" if changes == "none" else f"With {changes}, "
    if candidate.method == "full_payment":
        return f"{prefix}pay the full amount today while preserving the required minimum balance."
    if candidate.method == "partial_payment":
        return f"Pay the baseline safe amount today and the exact remainder on {capacity.earliest_date_for_full_payment.isoformat()}."
    if candidate.method == "installments":
        return f"Use supplied installment option {candidate.payment_option_id} with {len(candidate.payment_legs)} payments starting {candidate.start_date.isoformat()}."
    return f"Wait until {capacity.earliest_date_for_full_payment.isoformat()} for the safe full payment."


def derive_decision(dataset: Dataset, request: PurchaseRequest, capacity: PaymentCapacityResult, candidates: tuple[PlanCandidate, ...]) -> FinalDecision:
    if capacity.request_id != request.request_id:
        raise ValueError("L4 capacity does not match request")
    winner = select_candidate(candidates)
    if winner is None:
        return FinalDecision(request.request_id, capacity.amount_safe_to_pay, "not_affordable", "not_recommended", "none", capacity.earliest_date_for_full_payment, "none", explanation(request, capacity, None), None)
    if winner.method == "full_payment" and winner.start_date == request.request_date and not winner.spending_changes:
        status = "affordable_now"
    elif winner.method == "wait":
        status = "affordable_later"
    else:
        status = "affordable_with_plan"
    decision = FinalDecision(request.request_id, capacity.amount_safe_to_pay, status, winner.method, payment_plan_text(winner), capacity.earliest_date_for_full_payment, spending_changes_text(winner.spending_changes), explanation(request, capacity, winner), winner.candidate_id)
    validate_decision(dataset, request, capacity, decision, winner)
    return decision


def validate_decision(dataset: Dataset, request: PurchaseRequest, capacity: PaymentCapacityResult, decision: FinalDecision, winner: PlanCandidate | None) -> None:
    if not Decimal("0") <= decision.amount_safe_to_pay <= request.requested_amount:
        raise ValueError("amount_safe_to_pay is outside request bounds")
    if decision.affordability_status not in ALLOWED_STATUSES or decision.recommended_payment_method not in ALLOWED_METHODS:
        raise ValueError("Unsupported output enum")
    if winner is None:
        if (decision.affordability_status, decision.recommended_payment_method, decision.payment_plan, decision.spending_changes_needed) != ("not_affordable", "not_recommended", "none", "none"):
            raise ValueError("Fallback output is inconsistent")
        return
    if not winner.completes_by_desired_date or winner.completion_date > request.desired_completion_date:
        raise ValueError("L6 cannot recommend a post-deadline candidate")
    if tuple(sorted(winner.payment_legs, key=lambda leg: (leg.payment_date, leg.amount))) != winner.payment_legs:
        raise ValueError("Candidate schedule is not chronological")
    if len(winner.spending_changes) > 3:
        raise ValueError("Too many spending changes")
    if decision.affordability_status == "affordable_now":
        if winner.method != "full_payment" or winner.start_date != request.request_date or winner.spending_changes or capacity.earliest_date_for_full_payment != request.request_date:
            raise ValueError("affordable_now integration inconsistency")
    if winner.method == "partial_payment" and (decision.affordability_status != "affordable_with_plan" or len(winner.payment_legs) != 2):
        raise ValueError("Partial plan integration inconsistency")
    if winner.method == "wait" and (decision.affordability_status != "affordable_later" or winner.start_date <= request.request_date):
        raise ValueError("Wait plan integration inconsistency")
    if winner.method == "installments":
        option = dataset.indexes.payment_options_by_option_id[winner.payment_option_id or ""]
        expected = tuple((option.first_payment_date + __import__("datetime").timedelta(days=(option.payment_frequency_days or 0) * index), option.payment_amount) for index in range(option.number_of_payments))
        if tuple((leg.payment_date, leg.amount) for leg in winner.payment_legs) != expected:
            raise ValueError("Installment plan differs from supplied option")


def decision_row(decision: FinalDecision) -> dict[str, str]:
    return {
        "request_id": decision.request_id,
        "amount_safe_to_pay": format_decimal(decision.amount_safe_to_pay),
        "affordability_status": decision.affordability_status,
        "recommended_payment_method": decision.recommended_payment_method,
        "payment_plan": decision.payment_plan,
        "earliest_date_for_full_payment": decision.earliest_date_for_full_payment.isoformat() if decision.earliest_date_for_full_payment else "",
        "spending_changes_needed": decision.spending_changes_needed,
        "decision_explanation": decision.decision_explanation,
    }


def write_decisions(path: Path, decisions: tuple[FinalDecision, ...]) -> None:
    if len({item.request_id for item in decisions}) != len(decisions):
        raise ValueError("Exactly one output row is required per processed request")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(decision_row(item) for item in decisions)
