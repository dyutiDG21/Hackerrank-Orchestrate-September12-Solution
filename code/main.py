#!/usr/bin/env python3
"""Level 7 final Buy or Wait? orchestration entry point."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import Dataset, ImageEvidence, Message, load_dataset
from evidence_extraction import (
    EvidenceCache, EvidenceExtractionConfig, EvidenceExtractionError, OpenAIResponsesClient, UsageTracker,
    cache_key, extract_image_evidence, extract_message_evidence, source_from_image, source_from_message, validate_for_source,
)
from evidence_schema import EvidenceClaim
from financial_forecast import build_forecast
from financial_state_resolution import resolve_financial_state
from payment_capacity import evaluate_capacity_from_baseline
from payment_planning import generate_plan_candidates
from recommendation_selection import ALLOWED_METHODS, ALLOWED_STATUSES, OUTPUT_COLUMNS, FinalDecision, decision_row, derive_decision, write_decisions
from recurrence_inference import infer_recurring_series


def evaluation_scope(dataset: Dataset) -> tuple[tuple[Message, ...], tuple[ImageEvidence, ...]]:
    request_ids = {request.request_id for request in dataset.requests}
    user_ids = {request.user_id for request in dataset.requests}
    event_ids = {event.event_id for event in dataset.financial_events if event.user_id in user_ids}
    messages = tuple(sorted((message for message in dataset.messages if message.user_id in user_ids and (message.request_id in request_ids or message.related_event_id in event_ids)), key=lambda item: item.message_id))
    images = tuple(sorted((image for image in dataset.images if image.user_id in user_ids and (image.request_id in request_ids or image.related_event_id in event_ids)), key=lambda item: item.image_id))
    return messages, images


def cache_inventory(messages: tuple[Message, ...], images: tuple[ImageEvidence, ...], cache: EvidenceCache, model: str) -> tuple[int, int, int, int]:
    message_hits = sum(cache.get(cache_key(source_from_message(item), model=model)) is not None for item in messages)
    image_hits = sum(cache.get(cache_key(source_from_image(item), model=model)) is not None for item in images)
    return message_hits, len(messages) - message_hits, image_hits, len(images) - image_hits


def cached_claims(messages: tuple[Message, ...], images: tuple[ImageEvidence, ...], cache: EvidenceCache, model: str) -> tuple[EvidenceClaim, ...]:
    claims: list[EvidenceClaim] = []
    for item in messages:
        payload = cache.get(cache_key(source_from_message(item), model=model))
        if payload is not None:
            claims.extend(validate_for_source(payload, source_from_message(item)).claims)
    for item in images:
        payload = cache.get(cache_key(source_from_image(item), model=model))
        if payload is not None:
            claims.extend(validate_for_source(payload, source_from_image(item)).claims)
    return tuple(claims)


def validate_output(dataset: Dataset, decisions: tuple[FinalDecision, ...]) -> None:
    if len(decisions) != len(dataset.requests):
        raise ValueError(f"Expected {len(dataset.requests)} decisions, got {len(decisions)}")
    request_by_id = dataset.indexes.requests_by_request_id
    if [item.request_id for item in decisions] != [item.request_id for item in dataset.requests]:
        raise ValueError("Output request order/IDs do not exactly match dataset/requests.csv")
    for decision in decisions:
        request = request_by_id[decision.request_id]
        row = decision_row(decision)
        if tuple(row) != OUTPUT_COLUMNS or decision.affordability_status not in ALLOWED_STATUSES or decision.recommended_payment_method not in ALLOWED_METHODS:
            raise ValueError(f"{decision.request_id}: output columns or enums are invalid")
        try:
            value = Decimal(row["amount_safe_to_pay"])
        except InvalidOperation as exc:
            raise ValueError(f"{decision.request_id}: invalid amount_safe_to_pay") from exc
        if "E" in row["amount_safe_to_pay"].upper() or not Decimal("0") <= value <= request.requested_amount:
            raise ValueError(f"{decision.request_id}: amount_safe_to_pay invariant failed")
        if not decision.decision_explanation or "None" in row.values():
            raise ValueError(f"{decision.request_id}: explanation/None leakage")
        if decision.recommended_payment_method == "not_recommended" and (decision.payment_plan != "none" or decision.spending_changes_needed != "none"):
            raise ValueError(f"{decision.request_id}: fallback plan invariant failed")
        if decision.recommended_payment_method == "wait" and decision.earliest_date_for_full_payment is None:
            raise ValueError(f"{decision.request_id}: wait requires earliest date")
        _validate_serialized_plan(decision)
        _validate_serialized_changes(decision)


def _validate_serialized_plan(decision: FinalDecision) -> None:
    if decision.recommended_payment_method == "not_recommended":
        return
    if decision.payment_plan == "none":
        raise ValueError(f"{decision.request_id}: recommendation requires a payment plan")
    legs: list[tuple[date, Decimal]] = []
    for text in decision.payment_plan.split("|"):
        try:
            payment_date_text, amount_text = text.split(":", 1)
            leg = (date.fromisoformat(payment_date_text), Decimal(amount_text))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{decision.request_id}: malformed payment plan") from exc
        if leg[1] <= 0:
            raise ValueError(f"{decision.request_id}: payment legs must be positive")
        legs.append(leg)
    if legs != sorted(legs):
        raise ValueError(f"{decision.request_id}: payment plan is not chronological")
    if decision.recommended_payment_method in {"full_payment", "wait"} and len(legs) != 1:
        raise ValueError(f"{decision.request_id}: {decision.recommended_payment_method} requires one payment")
    if decision.recommended_payment_method == "partial_payment" and len(legs) != 2:
        raise ValueError(f"{decision.request_id}: partial payment requires exactly two legs")


def _validate_serialized_changes(decision: FinalDecision) -> None:
    if decision.spending_changes_needed == "none":
        return
    actions = decision.spending_changes_needed.split("|")
    if len(actions) > 3:
        raise ValueError(f"{decision.request_id}: more than three spending changes")
    event_ids: set[str] = set()
    for action in actions:
        fields = action.split(":")
        valid = (
            len(fields) == 2 and fields[0] == "stop" and bool(fields[1])
        ) or (
            len(fields) == 3 and fields[0] == "reduce_to" and bool(fields[1])
            and _is_positive_decimal(fields[2])
        )
        if not valid or fields[1] in event_ids:
            raise ValueError(f"{decision.request_id}: malformed or conflicting spending change")
        event_ids.add(fields[1])


def _is_positive_decimal(value: str) -> bool:
    try:
        return Decimal(value) > 0
    except InvalidOperation:
        return False


def estimate_runtime_cost(totals: dict[str, int]) -> Decimal:
    """Use the documented GPT-5.6 Terra standard rates with Decimal arithmetic."""
    million = Decimal("1000000")
    return (
        Decimal(totals["input_tokens"]) * Decimal("2.00") / million
        + Decimal(totals["cached_input_tokens"]) * Decimal("0.20") / million
        + Decimal(totals["output_tokens"]) * Decimal("12.00") / million
    )


def write_usage_report(path: Path, *, model: str, usage: UsageTracker, message_hits: int, image_hits: int, message_filled: int, image_filled: int, request_count: int) -> None:
    totals = usage.totals()
    total_tokens = totals["input_tokens"] + totals["output_tokens"]
    current_cost = estimate_runtime_cost(totals)
    historical_cache_entries = message_hits + image_hits
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Runtime Usage Report\n\n"
        "## Final Orchestration Pass\n\n"
        f"- Provider: OpenAI Responses API\n- Model: `{model}`\n"
        f"- New model calls: {totals['calls']}\n- New input tokens: {totals['input_tokens']}\n"
        f"- New cached input tokens: {totals['cached_input_tokens']}\n- New output tokens: {totals['output_tokens']}\n"
        f"- New total tokens: {total_tokens}\n- New average tokens per evaluation request: {Decimal(total_tokens) / Decimal(request_count)}\n"
        f"- New estimated runtime cost: ${current_cost:.8f}\n"
        f"- New estimated cost per evaluation request: ${current_cost / Decimal(request_count):.8f}\n"
        f"- Evidence cache entries reused: {historical_cache_entries}\n- Evidence cache misses filled this pass: {message_filled + image_filled}\n\n"
        "## Historical Evidence Represented By The Final Output\n\n"
        f"- Cached evaluation-relevant sources: {historical_cache_entries} ({message_hits} messages, {image_hits} images)\n"
        f"- Provider calls represented: {historical_cache_entries}; the extraction pipeline performs one Responses API call per source cache miss.\n"
        "- Historical input tokens: unavailable; cache payloads retain validated claims only, not provider usage.\n"
        "- Historical cached input tokens: unavailable; not persisted in cache or local usage metadata.\n"
        "- Historical output tokens: unavailable; not persisted in cache or local usage metadata.\n"
        "- Historical total tokens and average per evaluation request: unavailable because the component token counts cannot be reconstructed.\n"
        "- Historical estimated runtime cost and per-evaluation-request cost: unavailable because token usage cannot be reconstructed.\n\n"
        "Cost formula when usage is available: input_tokens * $2.00/1M + cached_input_tokens * $0.20/1M + output_tokens * $12.00/1M.\n"
        "This report covers solution runtime evidence extraction only, not Codex development usage.\n",
        encoding="utf-8", newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-real-api", action="store_true", help="Fill only cache misses using the configured Responses API key.")
    parser.add_argument("--cache-dir", default=".cache/evidence_extraction")
    args = parser.parse_args()
    dataset = load_dataset(); config = EvidenceExtractionConfig.from_env(cache_dir=args.cache_dir); cache = EvidenceCache(config.cache_dir)
    messages, images = evaluation_scope(dataset)
    message_hits, message_misses, image_hits, image_misses = cache_inventory(messages, images, cache, config.model)
    print(f"provider=OpenAI Responses API model={config.model}")
    print(f"messages total={len(messages)} cache_hits={message_hits} cache_misses={message_misses}")
    print(f"images total={len(images)} cache_hits={image_hits} cache_misses={image_misses}")
    if (message_misses or image_misses) and not args.run_real_api:
        print("Evidence cache is incomplete. Safe rerun command: python code/main.py --run-real-api")
        return 2
    usage = UsageTracker()
    if message_misses or image_misses:
        if not config.api_key:
            print("OPENAI_API_KEY is absent; no provider calls were made. Safe rerun command: python code/main.py --run-real-api")
            return 2
        client = OpenAIResponsesClient(config)
        try:
            for item in messages: extract_message_evidence(item, client=client, config=config, cache=cache, usage_tracker=usage)
            for item in images: extract_image_evidence(item, client=client, config=config, cache=cache, usage_tracker=usage)
        except EvidenceExtractionError as exc:
            print(f"evidence_completion_failed={exc}")
            return 3
    claims = cached_claims(messages, images, cache, config.model)
    state = resolve_financial_state(dataset, claims); recurrence = infer_recurring_series(state)
    decisions: list[FinalDecision] = []; blocker_count = 0; notices = Counter(); no_candidates = 0
    for request in dataset.requests:
        baseline = build_forecast(dataset, state, recurrence.series, user_id=request.user_id, start_date=request.request_date)
        capacity = evaluate_capacity_from_baseline(request, baseline)
        candidates = generate_plan_candidates(dataset, request, capacity, baseline, recurrence.series).candidates
        decision = derive_decision(dataset, request, capacity, candidates); decisions.append(decision)
        blocker_count += len(baseline.blockers); notices.update(item.reason for item in baseline.notices)
        no_candidates += int(decision.selected_candidate_id is None)
    final = tuple(decisions); validate_output(dataset, final)
    write_decisions(REPO_ROOT / "output.csv", final)
    write_usage_report(CODE_DIR / "evaluation" / "usage_report.md", model=config.model, usage=usage, message_hits=message_hits, image_hits=image_hits, message_filled=message_misses, image_filled=image_misses, request_count=len(dataset.requests))
    rows = [decision_row(item) for item in final]
    print(f"output_rows={len(rows)} invariant_validation=passed")
    print(f"status_counts={dict(sorted(Counter(row['affordability_status'] for row in rows).items()))}")
    print(f"method_counts={dict(sorted(Counter(row['recommended_payment_method'] for row in rows).items()))}")
    print(f"spending_change_requests={sum(row['spending_changes_needed'] != 'none' for row in rows)} no_candidate={no_candidates} blockers={blocker_count} notices={dict(notices)}")
    print(f"usage={usage.totals()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
