#!/usr/bin/env python3
"""Populate L2A evidence cache for solved-sample-relevant sources."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import sys


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import Dataset, ImageEvidence, Message, load_dataset  # noqa: E402
from evidence_extraction import (  # noqa: E402
    EvidenceCache,
    EvidenceExtractionError,
    EvidenceExtractionConfig,
    OpenAIResponsesClient,
    UsageTracker,
    cache_key,
    extract_image_evidence,
    extract_message_evidence,
    source_from_image,
    source_from_message,
    validate_for_source,
)
from evidence_schema import EvidenceClaim  # noqa: E402
from financial_state_resolution import resolve_financial_state  # noqa: E402
from payment_capacity import evaluate_request_capacity  # noqa: E402
from recurrence_inference import infer_recurring_series  # noqa: E402


@dataclass(frozen=True)
class SampleCapacitySnapshot:
    request_id: str
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: str | None
    baseline_violates_minimum: bool
    blockers: tuple[str, ...]


def sample_scope(dataset: Dataset) -> tuple[set[str], set[str], set[str]]:
    request_ids = {sample.request.request_id for sample in dataset.sample_requests}
    user_ids = {sample.request.user_id for sample in dataset.sample_requests}
    event_ids = {
        event.event_id
        for event in dataset.financial_events
        if event.user_id in user_ids
    }
    return request_ids, user_ids, event_ids


def select_sample_messages(dataset: Dataset) -> tuple[Message, ...]:
    request_ids, user_ids, event_ids = sample_scope(dataset)
    selected = [
        message
        for message in dataset.messages
        if message.user_id in user_ids
        and (
            message.request_id in request_ids
            or message.related_event_id in event_ids
        )
    ]
    return tuple(sorted(selected, key=lambda item: item.message_id))


def select_sample_images(dataset: Dataset) -> tuple[ImageEvidence, ...]:
    request_ids, user_ids, event_ids = sample_scope(dataset)
    selected = [
        image
        for image in dataset.images
        if image.user_id in user_ids
        and (
            image.request_id in request_ids
            or image.related_event_id in event_ids
        )
    ]
    return tuple(sorted(selected, key=lambda item: item.image_id))


def load_cached_claims_for_sources(
    *,
    messages: tuple[Message, ...],
    images: tuple[ImageEvidence, ...],
    cache: EvidenceCache,
    model: str,
) -> tuple[EvidenceClaim, ...]:
    claims: list[EvidenceClaim] = []
    for message in messages:
        source = source_from_message(message)
        payload = cache.get(cache_key(source, model=model))
        if payload is not None:
            claims.extend(validate_for_source(payload, source).claims)
    for image in images:
        source = source_from_image(image)
        payload = cache.get(cache_key(source, model=model))
        if payload is not None:
            claims.extend(validate_for_source(payload, source).claims)
    return tuple(claims)


def evaluate_samples(dataset: Dataset, claims: tuple[EvidenceClaim, ...]) -> tuple[SampleCapacitySnapshot, ...]:
    state = resolve_financial_state(dataset, claims)
    recurrence = infer_recurring_series(state)
    snapshots: list[SampleCapacitySnapshot] = []
    for sample in dataset.sample_requests:
        result = evaluate_request_capacity(dataset, state, recurrence.series, sample.request)
        snapshots.append(
            SampleCapacitySnapshot(
                request_id=sample.request.request_id,
                amount_safe_to_pay=result.amount_safe_to_pay,
                earliest_date_for_full_payment=(
                    result.earliest_date_for_full_payment.isoformat()
                    if result.earliest_date_for_full_payment is not None
                    else None
                ),
                baseline_violates_minimum=result.baseline_violates_minimum,
                blockers=tuple(blocker.reason for blocker in result.blockers),
            )
        )
    return tuple(snapshots)


def exact_comparison(dataset: Dataset, snapshots: tuple[SampleCapacitySnapshot, ...]) -> dict[str, object]:
    by_request = {item.request_id: item for item in snapshots}
    amount_errors: list[Decimal] = []
    amount_matches = 0
    date_matches = 0
    baseline_violations = 0
    blockers: dict[str, tuple[str, ...]] = {}
    for sample in dataset.sample_requests:
        actual = by_request[sample.request.request_id]
        expected_date = (
            sample.earliest_date_for_full_payment.isoformat()
            if sample.earliest_date_for_full_payment is not None
            else None
        )
        if actual.amount_safe_to_pay == sample.amount_safe_to_pay:
            amount_matches += 1
        if actual.earliest_date_for_full_payment == expected_date:
            date_matches += 1
        if actual.baseline_violates_minimum:
            baseline_violations += 1
        if actual.blockers:
            blockers[actual.request_id] = actual.blockers
        amount_errors.append(abs(actual.amount_safe_to_pay - sample.amount_safe_to_pay))
    ordered_errors = sorted(amount_errors)
    midpoint = len(ordered_errors) // 2
    median_error = ordered_errors[midpoint] if len(ordered_errors) % 2 else (ordered_errors[midpoint - 1] + ordered_errors[midpoint]) / Decimal("2")
    return {
        "amount_matches": amount_matches,
        "date_matches": date_matches,
        "mean_abs_error": sum(amount_errors, Decimal("0")) / Decimal(len(amount_errors)),
        "median_abs_error": median_error,
        "baseline_violations": baseline_violations,
        "blockers": blockers,
    }


def cache_has_source(cache: EvidenceCache, source, model: str) -> bool:  # noqa: ANN001
    return cache.path_for(cache_key(source, model=model)).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".cache/evidence_extraction")
    parser.add_argument("--run-real-api", action="store_true", help="Call the Responses API for cache misses.")
    args = parser.parse_args()

    dataset = load_dataset()
    config = EvidenceExtractionConfig.from_env(cache_dir=args.cache_dir)
    cache = EvidenceCache(config.cache_dir)
    messages = select_sample_messages(dataset)
    images = select_sample_images(dataset)
    before_claims = load_cached_claims_for_sources(messages=messages, images=images, cache=cache, model=config.model)
    before_results = evaluate_samples(dataset, before_claims)

    message_cache_hits = sum(1 for message in messages if cache_has_source(cache, source_from_message(message), config.model))
    image_cache_hits = sum(1 for image in images if cache_has_source(cache, source_from_image(image), config.model))
    message_cache_misses = len(messages) - message_cache_hits
    image_cache_misses = len(images) - image_cache_hits

    print(f"model={config.model}")
    print(f"selected_messages={len(messages)} cache_hits={message_cache_hits} cache_misses={message_cache_misses}")
    print(f"selected_images={len(images)} cache_hits={image_cache_hits} cache_misses={image_cache_misses}")

    usage = UsageTracker()
    if args.run_real_api:
        client = OpenAIResponsesClient(config)
        try:
            for message in messages:
                extract_message_evidence(message, client=client, config=config, cache=cache, usage_tracker=usage)
            for image in images:
                extract_image_evidence(image, client=client, config=config, cache=cache, usage_tracker=usage)
        except EvidenceExtractionError as exc:
            print(f"extraction_failed={exc}")
            return 2
    else:
        print("No API calls made; pass --run-real-api to populate cache misses.")

    after_claims = load_cached_claims_for_sources(messages=messages, images=images, cache=cache, model=config.model)
    after_results = evaluate_samples(dataset, after_claims)
    totals = usage.totals()
    print(
        "usage "
        f"new_api_calls={totals['calls']} input_tokens={totals['input_tokens']} "
        f"output_tokens={totals['output_tokens']} cached_input_tokens={totals['cached_input_tokens']}"
    )
    comparison = exact_comparison(dataset, after_results)
    print(
        "sample_capacity "
        f"amount_exact={comparison['amount_matches']}/25 "
        f"date_exact={comparison['date_matches']}/25 "
        f"mean_abs_error={comparison['mean_abs_error']} "
        f"median_abs_error={comparison['median_abs_error']} "
        f"baseline_violations={comparison['baseline_violations']} "
        f"blockers={comparison['blockers']}"
    )

    before_by_request = {item.request_id: item for item in before_results}
    changed = [
        (item.request_id, before_by_request[item.request_id], item)
        for item in after_results
        if before_by_request[item.request_id] != item
    ]
    print(f"changed_requests={len(changed)}")
    for request_id, before, after in changed:
        print(
            f"- {request_id}: amount {before.amount_safe_to_pay} -> {after.amount_safe_to_pay}; "
            f"earliest {before.earliest_date_for_full_payment} -> {after.earliest_date_for_full_payment}; "
            f"blockers {before.blockers} -> {after.blockers}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
