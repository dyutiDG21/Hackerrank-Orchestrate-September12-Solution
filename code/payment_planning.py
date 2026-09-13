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
from event_normalization import MissingExchangeRateError, convert_currency
from financial_forecast import (
    ForecastResult,
    HypotheticalMovement,
    ScenarioDebitAdjustment,
    build_scenario_forecast,
)
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
    """Return allowable actions for one canonical source event per series."""
    profile = dataset.indexes.profiles_by_user_id[request.user_id]
    recurring_event_ids = {
        series.member_event_ids[-1]
        for series in recurring_series
        if series.projectable and series.grouping_key.user_id == request.user_id and series.grouping_key.direction == "debit"
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
    """Deterministically enumerate non-conflicting scenario change sets."""
    results: list[tuple[SpendingChange, ...]] = [()]
    for length in range(1, MAX_SPENDING_CHANGES + 1):
        for candidate in combinations(changes, length):
            if len({item.event_id for item in candidate}) != len(candidate):
                continue
            results.append(candidate)
            if len(results) >= MAX_DEFERRED_CHANGE_SETS:
                return tuple(results)
    return tuple(results)


def bind_scenario_adjustments(
    dataset: Dataset,
    baseline: ForecastResult,
    changes: tuple[SpendingChange, ...],
) -> tuple[ScenarioDebitAdjustment, ...] | None:
    """Bind a source-currency reduction to exact dated home-currency movements."""
    _validate_changes(changes)
    profile = dataset.indexes.profiles_by_user_id[baseline.user_id]
    adjustments: list[ScenarioDebitAdjustment] = []
    for change in changes:
        event = dataset.indexes.events_by_event_id[change.event_id]
        matching = tuple(
            movement
            for movement in baseline.movements
            if movement.source == "recurrence_projection"
            and movement.direction == "debit"
            and change.event_id in movement.provenance
        )
        if not matching:
            return None
        home_amounts: list[tuple[str, Decimal]] = []
        if change.action == "reduce_to":
            assert change.new_amount is not None
            try:
                home_amounts = [
                    (
                        movement.movement_id,
                        convert_currency(
                            change.new_amount,
                            event.currency,
                            profile.home_currency,
                            movement.date,
                            dataset,
                        ).converted_amount,
                    )
                    for movement in matching
                ]
            except MissingExchangeRateError:
                return None
        adjustments.append(
            ScenarioDebitAdjustment(
                action=change.action,
                event_id=change.event_id,
                new_amount=change.new_amount,
                reduced_home_amounts=tuple(home_amounts),
                provenance=change.provenance,
            )
        )
    return tuple(adjustments)


def _scenario_for_legs(
    baseline: ForecastResult,
    *,
    candidate_id: str,
    legs: tuple[PaymentLeg, ...],
    scenario_adjustments: tuple[ScenarioDebitAdjustment, ...] = (),
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
    return build_scenario_forecast(
        baseline,
        hypothetical_movements=movements,
        scenario_adjustments=scenario_adjustments,
    ), excluded


def _safe_candidate(
    *,
    request: PurchaseRequest,
    candidate_id: str,
    method: str,
    legs: tuple[PaymentLeg, ...],
    payment_option_id: str | None,
    baseline: ForecastResult,
    traces: tuple[PlanTrace, ...],
    spending_changes: tuple[SpendingChange, ...] = (),
    scenario_adjustments: tuple[ScenarioDebitAdjustment, ...] = (),
) -> PlanCandidate | None:
    _validate_legs(legs)
    _validate_changes(spending_changes)
    scenario, excluded = _scenario_for_legs(
        baseline,
        candidate_id=candidate_id,
        legs=legs,
        scenario_adjustments=scenario_adjustments,
    )
    if scenario.blockers or scenario.minimum_balance_violated:
        return None
    return PlanCandidate(
        candidate_id=candidate_id,
        method=method,
        payment_legs=legs,
        spending_changes=spending_changes,
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


def _change_with_amount(change: SpendingChange, amount: Decimal) -> SpendingChange:
    return SpendingChange(change.action, change.event_id, amount, change.category, change.provenance)


def derive_least_reduced_changes(
    dataset: Dataset,
    baseline: ForecastResult,
    legs: tuple[PaymentLeg, ...],
    changes: tuple[SpendingChange, ...],
    *,
    candidate_id: str,
) -> tuple[SpendingChange, ...] | None:
    """Find largest safe targets, i.e. the least Decimal reductions, by search."""
    reduce_changes = tuple(change for change in changes if change.action == "reduce_to")
    if not reduce_changes:
        return changes

    def safe(candidate_changes: tuple[SpendingChange, ...]) -> bool:
        adjustments = bind_scenario_adjustments(dataset, baseline, candidate_changes)
        if adjustments is None:
            return False
        scenario, _ = _scenario_for_legs(
            baseline,
            candidate_id=candidate_id,
            legs=legs,
            scenario_adjustments=adjustments,
        )
        return not scenario.blockers and not scenario.minimum_balance_violated

    targets = {change.event_id: change.new_amount for change in reduce_changes}
    if not safe(changes):
        return None
    for change in reduce_changes:
        event = dataset.indexes.events_by_event_id[change.event_id]
        assert change.new_amount is not None and event.amount is not None
        minimum = change.new_amount
        maximum = event.amount
        scale = min(minimum.as_tuple().exponent, maximum.as_tuple().exponent)
        step = Decimal("1").scaleb(scale)
        low = int((minimum / step).to_integral_exact())
        high = int((maximum / step).to_integral_exact())
        while low < high:
            midpoint = (low + high + 1) // 2
            target = Decimal(midpoint) * step
            trial = tuple(
                _change_with_amount(item, target) if item.event_id == change.event_id else _change_with_amount(item, targets[item.event_id])
                if item.action == "reduce_to" else item
                for item in changes
            )
            if safe(trial):
                low = midpoint
                targets[change.event_id] = target
            else:
                high = midpoint - 1
    return tuple(
        _change_with_amount(item, targets[item.event_id]) if item.action == "reduce_to" else item
        for item in changes
    )


def _deduplicate(candidates: list[PlanCandidate]) -> tuple[PlanCandidate, ...]:
    unique: dict[tuple[str, str | None, tuple[tuple[date, Decimal], ...], tuple[tuple[str, str, Decimal | None], ...]], PlanCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.method,
            candidate.payment_option_id,
            tuple((leg.payment_date, leg.amount) for leg in candidate.payment_legs),
            tuple((change.action, change.event_id, change.new_amount) for change in candidate.spending_changes),
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
    schedule_specs: list[tuple[str, str, tuple[PaymentLeg, ...], str | None, tuple[PlanTrace, ...]]] = []
    traces: list[PlanTrace] = [
        PlanTrace("l4_capacity_input", "Used supplied Level 4 baseline capacity; did not recompute it.", (capacity.request_id,)),
    ]

    if _method_allowed("full_payment", profile.payment_methods_user_will_consider):
        full_legs = (PaymentLeg(request.request_date, request.requested_amount, (request.request_id, "full_now")),)
        schedule_specs.append((
            "full_now", "full_payment", full_legs, None,
            (PlanTrace("full_now", "One full requested-amount payment on the request date.", (request.request_id,)),),
        ))
        candidate = _safe_candidate(
            request=request,
            candidate_id="full_now",
            method="full_payment",
            legs=full_legs,
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
        schedule_specs.append((
            "partial", "partial_payment", legs, None,
            (PlanTrace("partial_documented_gates", "All documented partial-payment gates passed.", (request.request_id,)),),
        ))
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
            schedule_specs.append((
                f"installments:{option.payment_option_id}", "installments", legs, option.payment_option_id,
                (PlanTrace("supplied_installment_option", "Used the documented option schedule unchanged.", (option.payment_option_id,)),),
            ))
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
        wait_legs = (
            PaymentLeg(
                capacity.earliest_date_for_full_payment,
                request.requested_amount,
                (request.request_id, "wait_until_l4_earliest_full_date"),
            ),
        )
        schedule_specs.append((
            "wait", "wait", wait_legs, None,
            (PlanTrace("wait_at_l4_earliest_full_date", "Waited exactly until the supplied L4 earliest full-payment date.", (request.request_id,)),),
        ))
        candidate = _safe_candidate(
            request=request,
            candidate_id="wait",
            method="wait",
            legs=wait_legs,
            payment_option_id=None,
            baseline=baseline,
            traces=(PlanTrace("wait_at_l4_earliest_full_date", "Waited exactly until the supplied L4 earliest full-payment date.", (request.request_id,)),),
        )
        if candidate is not None:
            candidates.append(candidate)

    eligible_changes = eligible_spending_changes(dataset, request, recurring_series)
    for change_set in bounded_change_sets(eligible_changes)[1:]:
        for spec_id, method, legs, option_id, spec_traces in schedule_specs:
            derived = derive_least_reduced_changes(
                dataset,
                baseline,
                legs,
                change_set,
                candidate_id=f"{spec_id}:changes",
            )
            if derived is None:
                continue
            adjustments = bind_scenario_adjustments(dataset, baseline, derived)
            if adjustments is None:
                continue
            change_key = "+".join(f"{item.action}:{item.event_id}" for item in derived)
            candidate = _safe_candidate(
                request=request,
                candidate_id=f"{spec_id}:changes:{change_key}",
                method=method,
                legs=legs,
                payment_option_id=option_id,
                baseline=baseline,
                spending_changes=derived,
                scenario_adjustments=adjustments,
                traces=spec_traces + (
                    PlanTrace("l3_spending_change_scenario", "Validated permitted flexible recurring changes against L3.", tuple(item.event_id for item in derived)),
                ),
            )
            if candidate is not None:
                candidates.append(candidate)

    deferred_changes = ()
    if eligible_changes:
        traces.append(
            PlanTrace(
                "spending_changes_considered",
                "Bounded permitted flexible recurring change sets were validated with L3 scenario adjustments.",
                tuple(change.event_id for change in eligible_changes),
            )
        )

    return PlanGenerationResult(
        request_id=request.request_id,
        candidates=_deduplicate(candidates),
        deferred_spending_changes=deferred_changes,
        traces=tuple(traces),
    )
