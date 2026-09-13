"""Level 4 safe-payment capacity calculations for Buy or Wait?.

This module computes amount_safe_to_pay and earliest_date_for_full_payment from
the Level 3 forecast/scenario engine. It intentionally stops before payment
method selection, plan generation, spending changes, explanations, and CSV
serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from data_layer import Dataset, PurchaseRequest
from financial_forecast import ForecastBlocker, ForecastResult, HypotheticalMovement, build_forecast, build_scenario_forecast
from financial_state_resolution import ResolvedState
from recurrence_inference import RecurringSeries


CAPACITY_PROBE_ID = "l4_capacity_probe"


@dataclass(frozen=True)
class CapacityTrace:
    rule: str
    detail: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class DateCapacity:
    payment_date: date
    safe_capacity: Decimal
    minimum_balance_after_probe: Decimal
    probe_movement_id: str


@dataclass(frozen=True)
class PaymentCapacityResult:
    request_id: str
    user_id: str
    requested_amount: Decimal
    request_date: date
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: date | None
    baseline_minimum_observed_balance: Decimal
    baseline_available_slack: Decimal
    blockers: tuple[ForecastBlocker, ...]
    blocked: bool
    baseline_violates_minimum: bool
    request_date_capacity: DateCapacity | None
    earliest_date_capacity: DateCapacity | None
    traces: tuple[CapacityTrace, ...]


def clamp_nonnegative(value: Decimal) -> Decimal:
    return value if value > Decimal("0") else Decimal("0")


def probe_movement_id(probe_id: str = CAPACITY_PROBE_ID) -> str:
    return f"hypothetical:{probe_id}"


def capacity_from_probe_scenario(scenario: ForecastResult, *, probe_id: str = CAPACITY_PROBE_ID) -> DateCapacity:
    movement_id = probe_movement_id(probe_id)
    relevant = tuple(checkpoint for checkpoint in scenario.checkpoints if checkpoint.movement_id == movement_id)
    if len(relevant) != 1:
        raise ValueError(f"Scenario did not contain exactly one capacity probe {movement_id!r}")
    probe_index = scenario.checkpoints.index(relevant[0])
    minimum_after_probe = min(checkpoint.balance for checkpoint in scenario.checkpoints[probe_index:])
    safe_capacity = clamp_nonnegative(minimum_after_probe - scenario.minimum_balance_to_keep)
    return DateCapacity(
        payment_date=relevant[0].date,
        safe_capacity=safe_capacity,
        minimum_balance_after_probe=minimum_after_probe,
        probe_movement_id=movement_id,
    )


def safe_capacity_on_date(baseline: ForecastResult, payment_date: date) -> DateCapacity:
    scenario = build_scenario_forecast(
        baseline,
        hypothetical_movements=(
            HypotheticalMovement(
                CAPACITY_PROBE_ID,
                payment_date,
                Decimal("0"),
                direction="debit",
                description="Level 4 zero-amount capacity probe",
                provenance=(CAPACITY_PROBE_ID,),
            ),
        ),
    )
    return capacity_from_probe_scenario(scenario)


def build_blocked_result(request: PurchaseRequest, baseline: ForecastResult, traces: tuple[CapacityTrace, ...]) -> PaymentCapacityResult:
    slack = baseline.minimum_observed_balance - baseline.minimum_balance_to_keep
    return PaymentCapacityResult(
        request_id=request.request_id,
        user_id=request.user_id,
        requested_amount=request.requested_amount,
        request_date=request.request_date,
        amount_safe_to_pay=Decimal("0"),
        earliest_date_for_full_payment=None,
        baseline_minimum_observed_balance=baseline.minimum_observed_balance,
        baseline_available_slack=slack,
        blockers=baseline.blockers,
        blocked=bool(baseline.blockers),
        baseline_violates_minimum=baseline.minimum_balance_violated,
        request_date_capacity=None,
        earliest_date_capacity=None,
        traces=traces,
    )


def evaluate_capacity_from_baseline(request: PurchaseRequest, baseline: ForecastResult) -> PaymentCapacityResult:
    base_trace = CapacityTrace(
        "l3_baseline_forecast",
        "Used Level 3 baseline forecast and scenario ordering as the safety oracle.",
        (request.request_id, request.user_id),
    )
    if baseline.blockers:
        return build_blocked_result(
            request,
            baseline,
            (
                base_trace,
                CapacityTrace(
                    "unresolved_forecast_blockers",
                    "Unresolved L3 blockers prevent proving any additional request payment safe.",
                    tuple(f"{blocker.source_type}:{blocker.source_id}:{blocker.reason}" for blocker in baseline.blockers),
                ),
            ),
        )
    if baseline.minimum_balance_violated:
        return build_blocked_result(
            request,
            baseline,
            (
                base_trace,
                CapacityTrace(
                    "baseline_minimum_violation",
                    "Baseline forecast already falls below the user's minimum balance.",
                    (request.request_id,),
                ),
            ),
        )

    request_capacity = safe_capacity_on_date(baseline, request.request_date)
    amount_safe = min(request.requested_amount, request_capacity.safe_capacity)
    horizon_days = (baseline.end_date - baseline.start_date).days

    earliest_date: date | None = None
    earliest_capacity: DateCapacity | None = None
    for offset in range(horizon_days + 1):
        candidate_date = request.request_date + timedelta(days=offset)
        capacity = safe_capacity_on_date(baseline, candidate_date)
        if capacity.safe_capacity >= request.requested_amount:
            earliest_date = candidate_date
            earliest_capacity = capacity
            break

    return PaymentCapacityResult(
        request_id=request.request_id,
        user_id=request.user_id,
        requested_amount=request.requested_amount,
        request_date=request.request_date,
        amount_safe_to_pay=amount_safe,
        earliest_date_for_full_payment=earliest_date,
        baseline_minimum_observed_balance=baseline.minimum_observed_balance,
        baseline_available_slack=baseline.minimum_observed_balance - baseline.minimum_balance_to_keep,
        blockers=(),
        blocked=False,
        baseline_violates_minimum=False,
        request_date_capacity=request_capacity,
        earliest_date_capacity=earliest_capacity,
        traces=(
            base_trace,
            CapacityTrace(
                "exact_request_date_slack",
                "Computed exact request-date capacity from a zero-amount L3 hypothetical debit probe.",
                (request_capacity.probe_movement_id,),
            ),
            CapacityTrace(
                "calendar_date_earliest_full_search",
                "Searched each date in the baseline 90-day horizon using L3 same-day ordering.",
                (request.request_id,),
            ),
        ),
    )


def evaluate_request_capacity(
    dataset: Dataset,
    resolved_state: ResolvedState,
    recurring_series: tuple[RecurringSeries, ...],
    request: PurchaseRequest,
) -> PaymentCapacityResult:
    baseline = build_forecast(
        dataset,
        resolved_state,
        recurring_series,
        user_id=request.user_id,
        start_date=request.request_date,
    )
    return evaluate_capacity_from_baseline(request, baseline)
