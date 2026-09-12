#!/usr/bin/env python3
"""Opt-in real Responses API smoke test for Level 2A evidence extraction.

By default this script does not call the API. Pass --run-real-api to manually
exercise a small representative sample after implementation review.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_layer import load_dataset  # noqa: E402
from evidence_extraction import (  # noqa: E402
    EvidenceCache,
    EvidenceExtractionConfig,
    OpenAIResponsesClient,
    UsageTracker,
    cache_key,
    extract_image_evidence,
    extract_message_evidence,
    missing_amount_images,
    source_from_image,
    source_from_message,
    validate_for_source,
)


def select_smoke_sources():
    dataset = load_dataset()
    messages = [
        dataset.indexes.messages_by_message_id["message_01"],
        dataset.indexes.messages_by_message_id["message_02"],
    ]
    images = missing_amount_images(dataset)[:2]
    return dataset, tuple(messages), tuple(images)


def format_claim(claim) -> str:  # noqa: ANN001
    amount = str(claim.amount) if claim.amount is not None else ""
    effective_date = claim.effective_date.isoformat() if claim.effective_date is not None else ""
    external_ref = claim.referenced_external_id or ""
    return (
        f"  - source_type={claim.source_type}; source_id={claim.source_id}; "
        f"related_event_id={claim.related_event_id or ''}; claim_type={claim.claim_type}; "
        f"document_role={claim.document_role or ''}; amount={amount}; currency={claim.currency or ''}; "
        f"effective_date={effective_date}; identifier_role={claim.identifier_role or ''}; "
        f"external_ref={external_ref}"
    )


def cache_origin(cache: EvidenceCache, source, model: str) -> str:  # noqa: ANN001
    return "cache" if cache.path_for(cache_key(source, model=model)).exists() else "api"


def display_cached_source(cache: EvidenceCache, source, model: str) -> bool:  # noqa: ANN001
    key = cache_key(source, model=model)
    cached_payload = cache.get(key)
    if cached_payload is None:
        print(f"{source.source_id}: cache miss")
        return False
    result = validate_for_source(cached_payload, source)
    print(f"{source.source_id}: {len(result.claims)} claims (cache)")
    for claim in result.claims:
        print(format_claim(claim))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-real-api", action="store_true", help="Actually call the OpenAI Responses API.")
    parser.add_argument("--cache-dir", default=".cache/evidence_extraction", help="Extraction cache directory.")
    args = parser.parse_args()

    _, messages, images = select_smoke_sources()
    print(f"Selected {len(messages)} messages and {len(images)} images for smoke testing.")
    print("Message IDs: " + ", ".join(message.message_id for message in messages))
    print("Image IDs: " + ", ".join(image.image_id for image in images))
    config = EvidenceExtractionConfig.from_env(cache_dir=args.cache_dir)
    cache = EvidenceCache(config.cache_dir)
    if not args.run_real_api:
        print("No API calls made. Reading any matching entries from cache.")
        for message in messages:
            display_cached_source(cache, source_from_message(message), config.model)
        for image in images:
            display_cached_source(cache, source_from_image(image), config.model)
        return 0

    usage = UsageTracker()
    client = OpenAIResponsesClient(config)

    for message in messages:
        source = source_from_message(message)
        origin = cache_origin(cache, source, config.model)
        result = extract_message_evidence(message, client=client, config=config, cache=cache, usage_tracker=usage)
        print(f"{message.message_id}: {len(result.claims)} claims ({origin})")
        for claim in result.claims:
            print(format_claim(claim))
    for image in images:
        source = source_from_image(image)
        origin = cache_origin(cache, source, config.model)
        result = extract_image_evidence(image, client=client, config=config, cache=cache, usage_tracker=usage)
        print(f"{image.image_id}: {len(result.claims)} claims ({origin})")
        for claim in result.claims:
            print(format_claim(claim))
    totals = usage.totals()
    print(
        "Usage: "
        f"calls={totals['calls']}, input_tokens={totals['input_tokens']}, "
        f"output_tokens={totals['output_tokens']}, cached_input_tokens={totals['cached_input_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
