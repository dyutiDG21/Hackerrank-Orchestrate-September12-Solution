"""Deterministic 90-day balance forecasting for Buy or Wait? Level 3.

This layer consumes L1 profile state, L2B effective events, and L2C recurrence
models. It produces auditable baseline and scenario balance traces only; it
does not calculate safe purchase amounts, plans, recommendations, explanations,
or output.csv rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal

from data_layer import Dataset
from event_normalization import CashTreatment, MissingExchangeRateError, convert_currency
from financial_state_resolution import ResolvedEvent, ResolvedState
from recurrence_inference import RecurringEvidenceAmendment, RecurringSeries, add_months_anchored, normalize_description


DEFAULT_FORECAST_HORIZON_DAYS = 90


@dataclass(frozen=True)
class ForecastBlocker:
    reason: str
    date: date | None
    source_type: str
    source_id: str
    detail: str


@dataclass(frozen=True)
class ForecastNotice:
    reason: str
    date: date | None
    source_type: str
    source_id: str
    detail: str


@dataclass(frozen=True)
class ForecastMovement:
    movement_id: str
    date: date
    user_id: str
    source: str
    source_id: str
    direction: str
    amount: Decimal
    signed_amount: Decimal
    currency: str
    category: str | None
    event_type: str | None
    status: str | None
    flexibility: str | None
    provenance: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ForecastCheckpoint:
    date: date
    movement_id: str | None
    balance: Decimal
    minimum_balance_to_keep: Decimal
    violates_minimum: bool


@dataclass(frozen=True)
class ForecastDiagnostics:
    explicit_movements_included: int
    recurrence_projections_generated: int
    recurrence_projections_suppressed: int
    hypothetical_movements_included: int


@dataclass(frozen=True)
class ForecastResult:
    user_id: str
    home_currency: str
    start_date: date
    end_date: date
    starting_balance: Decimal
    minimum_balance_to_keep: Decimal
    movements: tuple[ForecastMovement, ...]
    checkpoints: tuple[ForecastCheckpoint, ...]
    minimum_observed_balance: Decimal
    minimum_balance_violated: bool
    first_violation_date: date | None
    blockers: tuple[ForecastBlocker, ...]
    notices: tuple[ForecastNotice, ...]
    diagnostics: ForecastDiagnostics


@dataclass(frozen=True)
class HypotheticalMovement:
    movement_id: str
    date: date
    amount: Decimal
    direction: str = "debit"
    description: str = "hypothetical payment"
    provenance: tuple[str, ...] = ()


def signed_amount(amount: Decimal, direction: str) -> Decimal:
    if direction == "debit":
        return -amount
    if direction == "credit":
        return amount
    raise ValueError(f"Unsupported forecast direction: {direction!r}")


def is_confirmed_income_credit(movement: ForecastMovement) -> bool:
    return (
        movement.direction == "credit"
        and movement.event_type == "income"
        and movement.category == "salary"
        and movement.source in {"explicit_event", "recurrence_projection"}
        and movement.status in {"settled", "scheduled", "projected"}
    )


def movement_sort_key(movement: ForecastMovement) -> tuple[date, int, str, str]:
    if movement.source == "pending_debit_reservation":
        direction_rank = 0
    elif movement.direction == "debit" and movement.source != "hypothetical":
        direction_rank = 1
    elif is_confirmed_income_credit(movement):
        direction_rank = 2
    elif movement.source == "hypothetical":
        direction_rank = 3
    elif movement.direction == "credit":
        direction_rank = 4
    else:
        direction_rank = 5
    source_rank = {
        "pending_debit_reservation": 0,
        "hypothetical": 1,
        "explicit_event": 2,
        "recurrence_projection": 3,
    }.get(movement.source, 9)
    return (movement.date, direction_rank, source_rank, movement.movement_id)


def compute_checkpoints(
    *,
    user_id: str,
    home_currency: str,
    start_date: date,
    end_date: date,
    starting_balance: Decimal,
    minimum_balance_to_keep: Decimal,
    movements: tuple[ForecastMovement, ...],
    blockers: tuple[ForecastBlocker, ...],
    diagnostics: ForecastDiagnostics,
    notices: tuple[ForecastNotice, ...] = (),
) -> ForecastResult:
    ordered_movements = tuple(sorted(movements, key=movement_sort_key))
    checkpoints: list[ForecastCheckpoint] = [
        ForecastCheckpoint(start_date, None, starting_balance, minimum_balance_to_keep, starting_balance < minimum_balance_to_keep)
    ]
    balance = starting_balance
    minimum_observed = starting_balance
    first_violation = start_date if starting_balance < minimum_balance_to_keep else None
    for movement in ordered_movements:
        balance += movement.signed_amount
        violates = balance < minimum_balance_to_keep
        checkpoints.append(ForecastCheckpoint(movement.date, movement.movement_id, balance, minimum_balance_to_keep, violates))
        if balance < minimum_observed:
            minimum_observed = balance
        if violates and first_violation is None:
            first_violation = movement.date
    return ForecastResult(
        user_id=user_id,
        home_currency=home_currency,
        start_date=start_date,
        end_date=end_date,
        starting_balance=starting_balance,
        minimum_balance_to_keep=minimum_balance_to_keep,
        movements=ordered_movements,
        checkpoints=tuple(checkpoints),
        minimum_observed_balance=minimum_observed,
        minimum_balance_violated=first_violation is not None,
        first_violation_date=first_violation,
        blockers=tuple(sorted(blockers, key=lambda item: (item.date or date.min, item.source_type, item.source_id, item.reason))),
        notices=tuple(sorted(notices, key=lambda item: (item.date or date.min, item.source_type, item.source_id, item.reason))),
        diagnostics=diagnostics,
    )


def explicit_event_date(record: ResolvedEvent, start_date: date) -> date:
    event = record.resolved_event
    if event.status == "pending" and event.direction == "debit":
        return start_date
    return record.normalized_event.cash_date


def explicit_movement_from_record(record: ResolvedEvent, movement_date: date) -> ForecastMovement | None:
    event = record.resolved_event
    normalized = record.normalized_event
    if normalized.cash_treatment == CashTreatment.PENDING_CREDIT_UNAVAILABLE:
        return None
    if event.amount is None or normalized.home_amount is None:
        return None
    source = "pending_debit_reservation" if normalized.reserves_cash else "explicit_event"
    return ForecastMovement(
        movement_id=f"{source}:{event.event_id}",
        date=movement_date,
        user_id=event.user_id,
        source=source,
        source_id=event.event_id,
        direction=event.direction,
        amount=normalized.home_amount,
        signed_amount=signed_amount(normalized.home_amount, event.direction),
        currency=normalized.home_currency,
        category=event.category,
        event_type=event.event_type,
        status=event.status,
        flexibility=event.flexibility,
        provenance=record.provenance_event_ids,
        description=event.description,
    )


def explicit_event_blockers(
    record: ResolvedEvent,
    movement_date: date,
) -> tuple[ForecastBlocker, ...]:
    event = record.resolved_event
    normalized = record.normalized_event
    if event.status == "pending" and event.direction == "debit" and event.amount is None:
        return (
            ForecastBlocker(
                "pending_debit_missing_amount",
                movement_date,
                "event",
                event.event_id,
                "Pending debit reserves liquidity but has no resolved amount.",
            ),
        )
    if normalized.requires_amount_evidence:
        return (
            ForecastBlocker(
                "missing_amount_requires_evidence",
                movement_date,
                "event",
                event.event_id,
                "Event amount is unresolved and cannot be forecast as zero.",
            ),
        )
    return ()


def explicit_event_notices(
    record: ResolvedEvent,
    movement_date: date,
) -> tuple[ForecastNotice, ...]:
    event = record.resolved_event
    if event.status == "pending" and event.direction == "credit":
        return (
            ForecastNotice(
                "pending_credit_excluded",
                movement_date,
                "event",
                event.event_id,
                "Pending credit is excluded from available cash until it settles.",
            ),
        )
    return ()


def explicit_movements_for_user(
    resolved_state: ResolvedState,
    *,
    user_id: str,
    start_date: date,
    end_date: date,
) -> tuple[tuple[ForecastMovement, ...], tuple[ForecastBlocker, ...], tuple[ForecastNotice, ...]]:
    movements: list[ForecastMovement] = []
    blockers: list[ForecastBlocker] = []
    notices: list[ForecastNotice] = []
    for record in resolved_state.effective_events:
        event = record.resolved_event
        if event.user_id != user_id or not record.active:
            continue
        movement_date = explicit_event_date(record, start_date)
        if movement_date < start_date or movement_date > end_date:
            continue
        blockers.extend(explicit_event_blockers(record, movement_date))
        notices.extend(explicit_event_notices(record, movement_date))
        movement = explicit_movement_from_record(record, movement_date)
        if movement is not None:
            movements.append(movement)
    return tuple(movements), tuple(blockers), tuple(notices)


def amount_amendment_for_date(series: RecurringSeries, occurrence_date: date) -> RecurringEvidenceAmendment | None:
    candidates = tuple(
        amendment
        for amendment in series.evidence_amendments
        if amendment.field == "amount"
        and amendment.amount is not None
        and amendment.effective_date is not None
        and amendment.effective_date <= occurrence_date
    )
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (item.effective_date or date.min, item.claim.source_type, item.claim.source_id, item.claim.claim_id),
    )[-1]


def conservative_recurring_amount(series: RecurringSeries, occurrence_date: date) -> tuple[Decimal | None, str | None, RecurringEvidenceAmendment | None]:
    amendment = amount_amendment_for_date(series, occurrence_date)
    if amendment is not None:
        return amendment.amount, amendment.currency or series.grouping_key.currency, amendment

    observations = series.amount_model.observed_amounts
    if not observations:
        return None, None, None
    if series.grouping_key.direction == "credit":
        selected = min(observations, key=lambda item: (item.amount, item.cash_date, item.event_id))
    else:
        selected = max(observations, key=lambda item: (item.amount, item.cash_date, item.event_id))
    return selected.amount, selected.currency, None


def next_occurrence_after(series: RecurringSeries, value: date) -> date:
    dates = series.cadence_evidence.observed_dates
    current = dates[-1]
    cadence = series.cadence_evidence.cadence
    if cadence == "weekly":
        step = timedelta(days=7)
        while current <= value:
            current += step
        return current
    if cadence == "biweekly":
        step = timedelta(days=14)
        while current <= value:
            current += step
        return current
    if cadence == "monthly":
        months = 1
        if current > value:
            return current
        while True:
            candidate = add_months_anchored(
                dates[-1],
                months,
                anchor_day=series.cadence_evidence.anchor_day or dates[-1].day,
                end_of_month=series.cadence_evidence.end_of_month_anchor,
            )
            if candidate > value:
                return candidate
            months += 1
    raise ValueError(f"Unsupported recurrence cadence: {cadence!r}")


def advance_occurrence(series: RecurringSeries, value: date) -> date:
    cadence = series.cadence_evidence.cadence
    if cadence == "weekly":
        return value + timedelta(days=7)
    if cadence == "biweekly":
        return value + timedelta(days=14)
    if cadence == "monthly":
        base = series.cadence_evidence.observed_dates[-1]
        months = (value.year - base.year) * 12 + (value.month - base.month) + 1
        return add_months_anchored(
            base,
            months,
            anchor_day=series.cadence_evidence.anchor_day or base.day,
            end_of_month=series.cadence_evidence.end_of_month_anchor,
        )
    raise ValueError(f"Unsupported recurrence cadence: {cadence!r}")


def explicit_keys_for_user(resolved_state: ResolvedState, user_id: str) -> set[tuple[str, str, str, str, str, str, date]]:
    keys: set[tuple[str, str, str, str, str, str, date]] = set()
    for record in resolved_state.effective_events:
        event = record.resolved_event
        if event.user_id != user_id or not record.active:
            continue
        keys.add((
            event.user_id,
            event.direction,
            event.event_type,
            event.category,
            normalize_description(event.description),
            event.currency,
            record.normalized_event.cash_date,
        ))
    return keys


def series_occurrence_key(series: RecurringSeries, occurrence_date: date) -> tuple[str, str, str, str, str, str, date]:
    key = series.grouping_key
    return (key.user_id, key.direction, key.event_type, key.category, key.description_key, key.currency, occurrence_date)


def convert_projection_amount(
    *,
    amount: Decimal,
    currency: str,
    occurrence_date: date,
    user_home_currency: str,
    dataset: Dataset,
) -> Decimal:
    return convert_currency(amount, currency, user_home_currency, occurrence_date, dataset).converted_amount


def recurrence_movements_for_user(
    dataset: Dataset,
    resolved_state: ResolvedState,
    series: tuple[RecurringSeries, ...],
    *,
    user_id: str,
    start_date: date,
    end_date: date,
) -> tuple[tuple[ForecastMovement, ...], tuple[ForecastBlocker, ...], int]:
    profile = dataset.indexes.profiles_by_user_id[user_id]
    explicit_keys = explicit_keys_for_user(resolved_state, user_id)
    movements: list[ForecastMovement] = []
    blockers: list[ForecastBlocker] = []
    suppressed = 0

    for item in sorted(series, key=lambda value: value.series_id):
        if item.grouping_key.user_id != user_id or not item.projectable:
            continue
        occurrence_date = next_occurrence_after(item, start_date - timedelta(days=1))
        while occurrence_date <= end_date:
            if series_occurrence_key(item, occurrence_date) in explicit_keys:
                suppressed += 1
                occurrence_date = advance_occurrence(item, occurrence_date)
                continue

            amount, currency, amendment = conservative_recurring_amount(item, occurrence_date)
            if amount is None or currency is None:
                blockers.append(
                    ForecastBlocker(
                        "recurring_amount_unavailable",
                        occurrence_date,
                        "recurrence_series",
                        item.series_id,
                        "Recurring series has no amount observation to project.",
                    )
                )
                occurrence_date = advance_occurrence(item, occurrence_date)
                continue

            try:
                home_amount = convert_projection_amount(
                    amount=amount,
                    currency=currency,
                    occurrence_date=occurrence_date,
                    user_home_currency=profile.home_currency,
                    dataset=dataset,
                )
            except MissingExchangeRateError as exc:
                blockers.append(
                    ForecastBlocker(
                        "missing_projected_exchange_rate",
                        occurrence_date,
                        "recurrence_series",
                        item.series_id,
                        str(exc),
                    )
                )
                occurrence_date = advance_occurrence(item, occurrence_date)
                continue

            provenance = (item.series_id,) + item.provenance_event_ids
            source_id = item.series_id
            description = f"projected {item.grouping_key.description_key}"
            if amendment is not None:
                provenance = provenance + (amendment.claim.source_id, amendment.claim.claim_id)
                source_id = f"{item.series_id}:{amendment.claim.source_id}:{amendment.claim.claim_id}"
                description = f"{description} with effective-dated amendment"

            movements.append(
                ForecastMovement(
                    movement_id=f"recurrence_projection:{item.series_id}:{occurrence_date.isoformat()}",
                    date=occurrence_date,
                    user_id=user_id,
                    source="recurrence_projection",
                    source_id=source_id,
                    direction=item.grouping_key.direction,
                    amount=home_amount,
                    signed_amount=signed_amount(home_amount, item.grouping_key.direction),
                    currency=profile.home_currency,
                    category=item.grouping_key.category,
                    event_type=item.grouping_key.event_type,
                    status="projected",
                    flexibility=None,
                    provenance=provenance,
                    description=description,
                )
            )
            occurrence_date = advance_occurrence(item, occurrence_date)

    return tuple(movements), tuple(blockers), suppressed


def hypothetical_to_movement(user_id: str, home_currency: str, item: HypotheticalMovement) -> ForecastMovement:
    return ForecastMovement(
        movement_id=f"hypothetical:{item.movement_id}",
        date=item.date,
        user_id=user_id,
        source="hypothetical",
        source_id=item.movement_id,
        direction=item.direction,
        amount=item.amount,
        signed_amount=signed_amount(item.amount, item.direction),
        currency=home_currency,
        category=None,
        event_type=None,
        status="hypothetical",
        flexibility=None,
        provenance=item.provenance or (item.movement_id,),
        description=item.description,
    )


def build_forecast(
    dataset: Dataset,
    resolved_state: ResolvedState,
    recurring_series: tuple[RecurringSeries, ...],
    *,
    user_id: str,
    start_date: date,
    horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    hypothetical_movements: tuple[HypotheticalMovement, ...] = (),
) -> ForecastResult:
    profile = dataset.indexes.profiles_by_user_id[user_id]
    end_date = start_date + timedelta(days=horizon_days)

    explicit_movements, explicit_blockers, explicit_notices = explicit_movements_for_user(
        resolved_state,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )
    recurrence_movements, recurrence_blockers, suppressed = recurrence_movements_for_user(
        dataset,
        resolved_state,
        recurring_series,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )
    hypotheticals = tuple(
        hypothetical_to_movement(user_id, profile.home_currency, item)
        for item in hypothetical_movements
        if start_date <= item.date <= end_date
    )
    diagnostics = ForecastDiagnostics(
        explicit_movements_included=len(explicit_movements),
        recurrence_projections_generated=len(recurrence_movements),
        recurrence_projections_suppressed=suppressed,
        hypothetical_movements_included=len(hypotheticals),
    )
    return compute_checkpoints(
        user_id=user_id,
        home_currency=profile.home_currency,
        start_date=start_date,
        end_date=end_date,
        starting_balance=profile.current_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        movements=explicit_movements + recurrence_movements + hypotheticals,
        blockers=explicit_blockers + recurrence_blockers,
        notices=explicit_notices,
        diagnostics=diagnostics,
    )


def build_scenario_forecast(
    baseline: ForecastResult,
    *,
    hypothetical_movements: tuple[HypotheticalMovement, ...],
) -> ForecastResult:
    hypotheticals = tuple(
        hypothetical_to_movement(baseline.user_id, baseline.home_currency, item)
        for item in hypothetical_movements
        if baseline.start_date <= item.date <= baseline.end_date
    )
    diagnostics = replace(
        baseline.diagnostics,
        hypothetical_movements_included=len(hypotheticals),
    )
    return compute_checkpoints(
        user_id=baseline.user_id,
        home_currency=baseline.home_currency,
        start_date=baseline.start_date,
        end_date=baseline.end_date,
        starting_balance=baseline.starting_balance,
        minimum_balance_to_keep=baseline.minimum_balance_to_keep,
        movements=baseline.movements + hypotheticals,
        blockers=baseline.blockers,
        notices=baseline.notices,
        diagnostics=diagnostics,
    )


def forecast_diagnostics(results: tuple[ForecastResult, ...]) -> dict[str, object]:
    blocker_counts = Counter(blocker.reason for result in results for blocker in result.blockers)
    notice_counts = Counter(notice.reason for result in results for notice in result.notices)
    duplicate_counts: Counter[tuple[str, date, Decimal, str, str | None]] = Counter()
    for result in results:
        for movement in result.movements:
            duplicate_counts[(movement.user_id, movement.date, movement.signed_amount, movement.source, movement.category)] += 1
    suspicious_duplicates = tuple(
        {
            "user_id": user_id,
            "date": movement_date.isoformat(),
            "signed_amount": str(amount),
            "source": source,
            "category": category,
            "count": count,
        }
        for (user_id, movement_date, amount, source, category), count in sorted(duplicate_counts.items(), key=lambda item: item[0])
        if count > 1
    )[:10]
    return {
        "forecasts": len(results),
        "explicit_movements_included": sum(result.diagnostics.explicit_movements_included for result in results),
        "recurrence_projections_generated": sum(result.diagnostics.recurrence_projections_generated for result in results),
        "recurrence_projections_suppressed": sum(result.diagnostics.recurrence_projections_suppressed for result in results),
        "unresolved_blockers_by_reason": dict(sorted(blocker_counts.items())),
        "notices_by_reason": dict(sorted(notice_counts.items())),
        "minimum_balance_violations": sum(1 for result in results if result.minimum_balance_violated),
        "suspicious_duplicate_movements": suspicious_duplicates,
    }
