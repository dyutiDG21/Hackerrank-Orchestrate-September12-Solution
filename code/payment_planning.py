"""Level 5 deterministic payment-plan candidate generation.

The planner constructs only preference-eligible, L3-safe payment schedules. It
does not calculate Level 4 capacity, rank candidates, select a recommendation,
or serialize output.  A supplied installment option is represented by its
documented first-payment date, frequency, count, and payment amount; no dates
or amounts are invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations

from data_layer import Dataset, FinancialEvent, PaymentOption, PurchaseRequest
from financial_forecast import ForecastResult, HypotheticalMovement, build_scenario_forecast
from payment_capacity import PaymentCapacityResult
from recurrence_inference import RecurringSeries


MAX_SPENDING_CHANGES = 3
MAX_DEFERRED_CHANGE_SETS = 64


@dataclass(frozen=True)
class PaymentLeg:
    payment_date: date
    amount: Decimal
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class SpendingChange:
    action: str
    event_id: str
    new_amount: Decimal | None
    category: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class PlanTrace:
    rule: str
    detail: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class PlanCandidate:
    candidate_id: str
    method: str
    payment_legs: tuple[PaymentLeg, ...]
    spending_changes: tuple[SpendingChange, ...]
    payment_option_id: str | None
    start_date: date
    completion_date: date
    total_paid: Decimal
    completes_by_desired_date: bool
    safety_forecast: ForecastResult
    safety_horizon_excluded_leg_count: int
    traces: tuple[PlanTrace, ...]


@dataclass(frozen=True)
class PlanGenerationResult:
    request_id: str
    candidates: tuple[PlanCandidate, ...]
    deferred_spending_changes: tuple[SpendingChange, ...]
    traces: tuple[PlanTrace, ...]


def _method_allowed(requested_method: str, permitted_methods: tuple[str, ...]) -> bool:
    return requested_method in permitted_methods


def _payment_leg_key(leg: PaymentLeg) -> tuple[date, Decimal, tuple[str, ...]]:
    return (leg.payment_date, leg.amount, leg.provenance)


def _validate_legs(legs: tuple[PaymentLeg, ...]) -> None:
    if not legs:
        raise ValueError("A payment plan must contain at least one leg")
    if any(leg.amount <= Decimal("0") for leg in legs):
        raise ValueError("Payment legs must have positive Decimal amounts")
    if tuple(sorted(legs, key=_payment_leg_key)) != legs:
        raise ValueError("Payment legs must be in deterministic chronological order")


def _validate_changes(changes: tuple[SpendingChange, ...]) -> None:
    if len(changes) > MAX_SPENDING_CHANGES:
        raise ValueError(f"A plan may contain at most {MAX_SPENDING_CHANGES} spending changes")
    event_ids = [change.event_id for change in changes]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("A plan cannot stop and reduce the same event")
    for change in changes:
        if change.action == "stop" and change.new_amount is not None:
            raise ValueError("Stop changes cannot contain a new amount")
        if change.action == "reduce_to" and (change.new_amount is None or change.new_amount < Decimal("0")):
            raise ValueError("Reduce changes require a non-negative Decimal target amount")
        if change.action not in {"stop", "reduce_to"}:
            raise ValueError(f"Unsupported spending change action: {change.action!r}")


def installment_legs(option: PaymentOption) -> tuple[PaymentLeg, ...]:
    if option.payment_method != "installments":
        raise ValueError(f"Payment option {option.payment_option_id!r} is not an installment option")
    if option.number_of_payments <= 0 or option.payment_frequency_days is None or option.payment_frequency_days <= 0:
        raise ValueError(f"Installment option {option.payment_option_id!r} lacks a valid supplied schedule")
    legs = tuple(
        PaymentLeg(
            payment_date=option.first_payment_date + timedelta(days=option.payment_frequency_days * index),
            amount=option.payment_amount,
            provenance=(option.payment_option_id, f"installment:{index + 1}"),
        )
        for index in range(option.number_of_payments)
    )
    _validate_legs(legs)
    if sum((leg.amount for leg in legs), Decimal("0")) != option.total_payable_amount:
        raise ValueError(f"Installment option {option.payment_option_id!r} does not match its supplied total")
    return legs


def eligible_spending_changes(
    dataset: Dataset,
    request: PurchaseRequest,
    recurring_series: tuple[RecurringSeries, ...],
) -> tuple[SpendingChange, ...]:
    """Return allowable recurring-event actions; L3 cannot yet validate them.

    Only events participating in projectable recurrence are eligible. This
    prevents a one-off flexible purchase from becoming a fabricated ongoing
    saving. The returned actions retain their event provenance for a later L3
    scenario API that can suppress or amend an existing forecast movement.
    """
    profile = dataset.indexes.profiles_by_user_id[request.user_id]
    recurring_event_ids = {
        event_id
        for series in recurring_series
        if series.projectable and series.grouping_key.user_id == request.user_id and series.grouping_key.direction == "debit"
        for event_id in series.member_event_ids
    }
    changes: list[SpendingChange] = []
    for event_id in sorted(recurring_event_ids):
        event: FinancialEvent = dataset.indexes.events_by_event_id[event_id]
        if event.category in profile.expense_categories_to_protect:
            continue
        provenance = (event.event_id, event.category, event.flexibility)
        if (
            event.flexibility in {"stoppable", "reducible_or_stoppable"}
            and event.category in profile.expense_categories_user_is_willing_to_stop
        ):
            changes.append(SpendingChange("stop", event.event_id, None, event.category, provenance))
        if (
            event.flexibility in {"reducible", "reducible_or_stoppable"}
            and event.category in profile.expense_categories_user_is_willing_to_reduce
            and event.amount is not None
            and event.minimum_allowed_amount is not None
            and Decimal("0") <= event.minimum_allowed_amount < event.amount
        ):
            changes.append(
                SpendingChange("reduce_to", event.event_id, event.minimum_allowed_amount, event.category, provenance)
            )
    return tuple(sorted(changes, key=lambda item: (item.event_id, item.action, item.new_amount or Decimal("0"))))


def bounded_change_sets(changes: tuple[SpendingChange, ...]) -> tuple[tuple[SpendingChange, ...], ...]:
    """Deterministically enumerate non-conflicting change sets for later L3 support."""
    results: list[tuple[SpendingChange, ...]] = [()]
    for length in range(1, MAX_SPENDING_CHANGES + 1):
        for candidate in combinations(changes, length):
            if len({item.event_id for item in candidate}) != len(candidate):
                continue
            results.append(candidate)
            if len(results) >= MAX_DEFERRED_CHANGE_SETS:
                return tuple(results)
    return tuple(results)


def _scenario_for_legs(
    baseline: ForecastResult,
    *,
    candidate_id: str,
    legs: tuple[PaymentLeg, ...],
) -> tuple[ForecastResult, int]:
    movements = tuple(
        HypotheticalMovement(
            movement_id=f"{candidate_id}:leg:{index + 1}",
            date=leg.payment_date,
            amount=leg.amount,
            direction="debit",
            description="Level 5 candidate payment",
            provenance=leg.provenance,
        )
        for index, leg in enumerate(legs)
    )
    excluded = sum(1 for movement in movements if not (baseline.start_date <= movement.date <= baseline.end_date))
    return build_scenario_forecast(baseline, hypothetical_movements=movements), excluded


def _safe_candidate(
    *,
    request: PurchaseRequest,
    candidate_id: str,
    method: str,
    legs: tuple[PaymentLeg, ...],
    payment_option_id: str | None,
    baseline: ForecastResult,
    traces: tuple[PlanTrace, ...],
) -> PlanCandidate | None:
    _validate_legs(legs)
    scenario, excluded = _scenario_for_legs(baseline, candidate_id=candidate_id, legs=legs)
    if scenario.blockers or scenario.minimum_balance_violated:
        return None
    return PlanCandidate(
        candidate_id=candidate_id,
        method=method,
        payment_legs=legs,
        spending_changes=(),
        payment_option_id=payment_option_id,
        start_date=legs[0].payment_date,
        completion_date=legs[-1].payment_date,
        total_paid=sum((leg.amount for leg in legs), Decimal("0")),
        completes_by_desired_date=legs[-1].payment_date <= request.desired_completion_date,
        safety_forecast=scenario,
        safety_horizon_excluded_leg_count=excluded,
        traces=traces + (
            PlanTrace(
                "l3_scenario_safety_check",
                "Validated the candidate's in-horizon payment legs with the unchanged Level 3 scenario forecast.",
                tuple(leg.provenance[0] for leg in legs),
            ),
        ),
    )


def _deduplicate(candidates: list[PlanCandidate]) -> tuple[PlanCandidate, ...]:
    unique: dict[tuple[str, str | None, tuple[tuple[date, Decimal], ...]], PlanCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.method,
            candidate.payment_option_id,
            tuple((leg.payment_date, leg.amount) for leg in candidate.payment_legs),
        )
        unique.setdefault(key, candidate)
    return tuple(sorted(unique.values(), key=lambda item: (item.method, item.payment_option_id or "", item.candidate_id)))


def generate_plan_candidates(
    dataset: Dataset,
    request: PurchaseRequest,
    capacity: PaymentCapacityResult,
    baseline: ForecastResult,
    recurring_series: tuple[RecurringSeries, ...] = (),
) -> PlanGenerationResult:
    """Generate safe L5 candidates without ranking or selecting one."""
    if capacity.request_id != request.request_id or capacity.user_id != request.user_id:
        raise ValueError("Payment capacity result does not belong to the supplied request")
    if baseline.user_id != request.user_id or baseline.start_date != request.request_date:
        raise ValueError("Baseline forecast must belong to the request user and start on the request date")

    profile = dataset.indexes.profiles_by_user_id[request.user_id]
    candidates: list[PlanCandidate] = []
    traces: list[PlanTrace] = [
        PlanTrace("l4_capacity_input", "Used supplied Level 4 baseline capacity; did not recompute it.", (capacity.request_id,)),
    ]

    if _method_allowed("full_payment", profile.payment_methods_user_will_consider):
        candidate = _safe_candidate(
            request=request,
            candidate_id="full_now",
            method="full_payment",
            legs=(PaymentLeg(request.request_date, request.requested_amount, (request.request_id, "full_now")),),
            payment_option_id=None,
            baseline=baseline,
            traces=(PlanTrace("full_now", "One full requested-amount payment on the request date.", (request.request_id,)),),
        )
        if candidate is not None:
            candidates.append(candidate)

    if (
        request.allows_partial_payment
        and _method_allowed("partial_payment", profile.payment_methods_user_will_consider)
        and Decimal("0") < capacity.amount_safe_to_pay < request.requested_amount
        and capacity.earliest_date_for_full_payment is not None
        and capacity.earliest_date_for_full_payment <= request.desired_completion_date
    ):
        first = capacity.amount_safe_to_pay
        second = request.requested_amount - first
        legs = (
            PaymentLeg(request.request_date, first, (request.request_id, "partial:first")),
            PaymentLeg(capacity.earliest_date_for_full_payment, second, (request.request_id, "partial:remaining")),
        )
        candidate = _safe_candidate(
            request=request,
            candidate_id="partial",
            method="partial_payment",
            legs=legs,
            payment_option_id=None,
            baseline=baseline,
            traces=(PlanTrace("partial_documented_gates", "All documented partial-payment gates passed.", (request.request_id,)),),
        )
        if candidate is not None:
            candidates.append(candidate)

    if _method_allowed("installments", profile.payment_methods_user_will_consider) and profile.max_installment_months is not None:
        for option in sorted(dataset.indexes.payment_options_by_request_id.get(request.request_id, ()), key=lambda item: item.payment_option_id):
            if option.payment_method != "installments" or option.number_of_payments > profile.max_installment_months:
                continue
            legs = installment_legs(option)
            candidate = _safe_candidate(
                request=request,
                candidate_id=f"installments:{option.payment_option_id}",
                method="installments",
                legs=legs,
                payment_option_id=option.payment_option_id,
                baseline=baseline,
                traces=(PlanTrace("supplied_installment_option", "Used the documented option schedule unchanged.", (option.payment_option_id,)),),
            )
            if candidate is not None:
                candidates.append(candidate)

    if (
        _method_allowed("full_payment", profile.payment_methods_user_will_consider)
        and capacity.earliest_date_for_full_payment is not None
        and capacity.earliest_date_for_full_payment > request.request_date
    ):
        candidate = _safe_candidate(
            request=request,
            candidate_id="wait",
            method="wait",
            legs=(
                PaymentLeg(
                    capacity.earliest_date_for_full_payment,
                    request.requested_amount,
                    (request.request_id, "wait_until_l4_earliest_full_date"),
                ),
            ),
            payment_option_id=None,
            baseline=baseline,
            traces=(PlanTrace("wait_at_l4_earliest_full_date", "Waited exactly until the supplied L4 earliest full-payment date.", (request.request_id,)),),
        )
        if candidate is not None:
            candidates.append(candidate)

    deferred_changes = eligible_spending_changes(dataset, request, recurring_series)
    if deferred_changes:
        traces.append(
            PlanTrace(
                "spending_changes_deferred",
                "Eligible recurring flexible changes were not emitted because L3 scenarios cannot suppress or amend baseline movements.",
                tuple(change.event_id for change in deferred_changes),
            )
        )

    return PlanGenerationResult(
        request_id=request.request_id,
        candidates=_deduplicate(candidates),
        deferred_spending_changes=deferred_changes,
        traces=tuple(traces),
    )
