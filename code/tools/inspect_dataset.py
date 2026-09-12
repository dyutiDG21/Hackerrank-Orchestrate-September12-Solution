#!/usr/bin/env python3
"""Deterministic dataset-contract inspection for the Buy or Wait? challenge.

This utility reads the participant-facing CSV files under dataset/ and prints a
concise implementation-oriented summary. It never modifies the dataset.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MISSING_MARKERS = {"", "na", "n/a", "null", "none"}

EXPECTED_OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

PRIMARY_KEYS = {
    "financial_profiles.csv": ["user_id"],
    "financial_events.csv": ["event_id"],
    "requests.csv": ["request_id"],
    "sample_requests.csv": ["request_id"],
    "request_payment_options.csv": ["payment_option_id"],
    "messages.csv": ["message_id"],
    "images.csv": ["image_id"],
    "exchange_rates.csv": ["rate_date", "from_currency", "to_currency"],
    "output.csv": ["request_id"],
}

IMPORTANT_NULL_COLUMNS = {
    "financial_profiles.csv": [
        "user_id",
        "home_currency",
        "current_available_balance",
        "minimum_balance_to_keep",
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        "max_installment_months",
    ],
    "financial_events.csv": [
        "event_id",
        "user_id",
        "event_date",
        "settlement_date",
        "amount",
        "currency",
        "event_type",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    ],
    "requests.csv": [
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
    ],
    "sample_requests.csv": [
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "spending_changes_needed",
    ],
    "request_payment_options.csv": [
        "payment_option_id",
        "request_id",
        "payment_method",
        "total_payable_amount",
        "number_of_payments",
        "first_payment_date",
        "payment_frequency_days",
        "financing_fee",
    ],
    "messages.csv": ["message_id", "user_id", "request_id", "related_event_id"],
    "images.csv": ["image_id", "user_id", "request_id", "related_event_id"],
    "exchange_rates.csv": ["rate_date", "from_currency", "to_currency", "rate"],
    "output.csv": EXPECTED_OUTPUT_COLUMNS,
}

CATEGORICAL_HINTS = (
    "status",
    "type",
    "category",
    "currency",
    "method",
    "frequency",
    "recurring",
    "flexible",
    "protected",
    "adjustable",
    "allows",
    "priority",
    "preference",
    "flexibility",
    "source",
)

PIPE_LIST_COLUMNS = {
    "financial_profiles.csv": [
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
    ],
}


@dataclass
class Table:
    name: str
    path: Path
    rows: list[dict[str, str]]
    columns: list[str]


def is_missing(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_MARKERS


def present(value: object) -> bool:
    return not is_missing(value)


def read_csv(path: Path) -> Table:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        return Table(path.name, path, rows, reader.fieldnames or [])


def count_missing(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if is_missing(row.get(column)))


def values(rows: Iterable[dict[str, str]], column: str) -> list[str]:
    return [row.get(column, "") for row in rows if present(row.get(column, ""))]


def counter_for(rows: Iterable[dict[str, str]], column: str) -> Counter[str]:
    return Counter(values(rows, column))


def format_counter(counter: Counter[str], limit: int = 20) -> str:
    if not counter:
        return "none"
    parts = [f"{key}={counter[key]}" for key in sorted(counter, key=lambda item: (-counter[item], item))[:limit]]
    if len(counter) > limit:
        parts.append(f"...+{len(counter) - limit} more")
    return ", ".join(parts)


def distribution(numbers: Iterable[int]) -> str:
    vals = sorted(numbers)
    if not vals:
        return "none"
    return f"min={vals[0]}, max={vals[-1]}, values={format_counter(Counter(vals), 12)}"


def composite_key(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in columns)


def print_section(title: str) -> None:
    print()
    print(f"## {title}")


def inspect_schema(tables: dict[str, Table]) -> None:
    print_section("CSV Inventory")
    for name in sorted(tables):
        table = tables[name]
        print(f"- {name}: rows={len(table.rows)}, columns={', '.join(table.columns)}")


def inspect_primary_keys(tables: dict[str, Table]) -> None:
    print_section("Primary Identifier Uniqueness")
    for name in sorted(tables):
        table = tables[name]
        key_columns = [column for column in PRIMARY_KEYS.get(name, []) if column in table.columns]
        if not key_columns:
            print(f"- {name}: no configured primary key columns present")
            continue
        keys = [composite_key(row, key_columns) for row in table.rows]
        missing = sum(1 for key in keys if any(is_missing(part) for part in key))
        duplicate_groups = sum(1 for _, count in Counter(keys).items() if count > 1)
        print(
            f"- {name}: key=({', '.join(key_columns)}), unique={len(set(keys))}/{len(keys)}, "
            f"missing_key_rows={missing}, duplicate_key_groups={duplicate_groups}"
        )


def inspect_nulls(tables: dict[str, Table]) -> None:
    print_section("Important Missing Values")
    for name in sorted(tables):
        table = tables[name]
        interesting = [column for column in IMPORTANT_NULL_COLUMNS.get(name, []) if column in table.columns]
        missing = [(column, count_missing(table.rows, column)) for column in interesting]
        missing = [(column, count) for column, count in missing if count > 0 or column in table.columns]
        if missing:
            print(f"- {name}: " + ", ".join(f"{column}={count}" for column, count in missing))


def inspect_vocabularies(tables: dict[str, Table]) -> None:
    print_section("Categorical Vocabularies")
    for name in sorted(tables):
        table = tables[name]
        for column in table.columns:
            lower = column.lower()
            if not any(hint in lower for hint in CATEGORICAL_HINTS):
                continue
            counter = counter_for(table.rows, column)
            if counter and len(counter) <= 30:
                print(f"- {name}.{column}: {format_counter(counter)}")
        for column in PIPE_LIST_COLUMNS.get(name, []):
            if column not in table.columns:
                continue
            split_counter: Counter[str] = Counter()
            for raw in values(table.rows, column):
                for item in raw.split("|"):
                    item = item.strip()
                    if item:
                        split_counter[item] += 1
            if split_counter:
                print(f"- {name}.{column} items: {format_counter(split_counter, 30)}")


def inspect_join_integrity(tables: dict[str, Table]) -> None:
    print_section("Join Integrity")
    profiles = tables.get("financial_profiles.csv")
    requests = tables.get("requests.csv")
    samples = tables.get("sample_requests.csv")
    events = tables.get("financial_events.csv")
    options = tables.get("request_payment_options.csv")
    messages = tables.get("messages.csv")
    images = tables.get("images.csv")

    user_ids = set(values(profiles.rows, "user_id")) if profiles and "user_id" in profiles.columns else set()
    request_ids = set(values(requests.rows, "request_id")) if requests and "request_id" in requests.columns else set()
    sample_ids = set(values(samples.rows, "request_id")) if samples and "request_id" in samples.columns else set()
    all_request_ids = request_ids | sample_ids
    event_ids = set(values(events.rows, "event_id")) if events and "event_id" in events.columns else set()

    def missing_refs(table: Table | None, column: str, valid: set[str]) -> tuple[int, int]:
        if not table or column not in table.columns:
            return (0, 0)
        refs = values(table.rows, column)
        return (len(refs), sum(1 for ref in refs if ref not in valid))

    for table_name in ["requests.csv", "sample_requests.csv", "financial_events.csv", "messages.csv", "images.csv"]:
        table = tables.get(table_name)
        total, missing = missing_refs(table, "user_id", user_ids)
        if table and "user_id" in table.columns:
            print(f"- {table_name}.user_id -> profiles: refs={total}, missing={missing}")

    for table_name, valid_ids in [
        ("request_payment_options.csv", all_request_ids),
        ("messages.csv", all_request_ids),
        ("images.csv", all_request_ids),
    ]:
        table = tables.get(table_name)
        total, missing = missing_refs(table, "request_id", valid_ids)
        if table and "request_id" in table.columns:
            print(f"- {table_name}.request_id -> requests+samples: refs={total}, missing={missing}")

    for table_name in ["messages.csv", "images.csv", "financial_events.csv"]:
        table = tables.get(table_name)
        column = "related_event_id" if table_name != "financial_events.csv" else "linked_event_id"
        total, missing = missing_refs(table, column, event_ids)
        if table and column in table.columns:
            print(f"- {table_name}.{column} -> events: refs={total}, missing={missing}")

    if options and "request_id" in options.columns:
        option_request_ids = set(values(options.rows, "request_id"))
        print(f"- evaluation requests without payment options: {len(request_ids - option_request_ids)}")
        print(f"- sample requests without payment options: {len(sample_ids - option_request_ids)}")


def inspect_event_patterns(tables: dict[str, Table]) -> None:
    events = tables.get("financial_events.csv")
    if not events:
        return
    print_section("Financial Event Patterns")
    for column in ["status", "event_type", "category", "direction", "cash_flow_type", "lifecycle_state"]:
        if column in events.columns:
            print(f"- {column}: {format_counter(counter_for(events.rows, column))}")
    if {"status", "event_type"}.issubset(events.columns):
        combo = Counter((row.get("status", ""), row.get("event_type", "")) for row in events.rows)
        print("- status x event_type: " + format_counter(Counter({f"{a}/{b}": c for (a, b), c in combo.items()}), 25))
    if "linked_event_id" in events.columns:
        linked_rows = [row for row in events.rows if present(row.get("linked_event_id"))]
        event_by_id = {row.get("event_id", ""): row for row in events.rows}
        link_patterns = Counter()
        for row in linked_rows:
            target = event_by_id.get(row.get("linked_event_id", ""), {})
            source_status = row.get("status", "unknown")
            target_status = target.get("status", "missing")
            source_type = row.get("event_type", "unknown")
            target_type = target.get("event_type", "missing")
            link_patterns[f"{target_status}/{target_type} -> {source_status}/{source_type}"] += 1
        print(f"- linked_event_id present: {len(linked_rows)}")
        print(f"- linked lifecycle patterns: {format_counter(link_patterns, 20)}")
    duplicate_full_rows = len(events.rows) - len({tuple((column, row.get(column, "")) for column in events.columns) for row in events.rows})
    print(f"- exact duplicate financial event rows: {duplicate_full_rows}")


def inspect_recurring_patterns(tables: dict[str, Table]) -> None:
    events = tables.get("financial_events.csv")
    if not events:
        return
    print_section("Recurring-Event Fields And Patterns")
    recurring_columns = [
        column
        for column in events.columns
        if any(term in column.lower() for term in ["recurr", "frequency", "interval", "day_of", "flexible"])
    ]
    print(f"- recurring-related columns: {', '.join(recurring_columns) if recurring_columns else 'none'}")
    for column in recurring_columns:
        print(f"- {column}: missing={count_missing(events.rows, column)}, values={format_counter(counter_for(events.rows, column), 20)}")
    recurring_rows = [
        row
        for row in events.rows
        if any(present(row.get(column)) and row.get(column, "").lower() not in {"false", "0", "no"} for column in recurring_columns)
    ]
    print(f"- rows flagged by recurring-related fields: {len(recurring_rows)}")
    for column in ["event_type", "category", "status", "flexibility"]:
        if column in events.columns and recurring_rows:
            print(f"- recurring rows by {column}: {format_counter(counter_for(recurring_rows, column), 20)}")
    if "event_type" in events.columns:
        subscriptions = [row for row in events.rows if row.get("event_type") == "subscription"]
        print(f"- subscription event rows: {len(subscriptions)}")
        if subscriptions:
            for column in ["category", "status", "flexibility"]:
                if column in events.columns:
                    print(f"- subscription rows by {column}: {format_counter(counter_for(subscriptions, column), 20)}")
    repeat_columns = ["user_id", "category", "direction", "amount", "currency"]
    if set(repeat_columns).issubset(events.columns):
        repeated_groups = defaultdict(list)
        for row in events.rows:
            if row.get("status") not in {"settled", "scheduled", "pending"}:
                continue
            key = tuple(row.get(column, "") for column in repeat_columns)
            repeated_groups[key].append(row)
        repeat_sizes = [len(rows) for rows in repeated_groups.values() if len(rows) >= 3]
        print(f"- repeated user/category/direction/amount/currency groups with >=3 rows: {len(repeat_sizes)}")
        print(f"- repeated-group size distribution: {distribution(repeat_sizes)}")


def inspect_payment_options(tables: dict[str, Table]) -> None:
    options = tables.get("request_payment_options.csv")
    if not options:
        return
    print_section("Payment Options")
    by_request = defaultdict(int)
    for row in options.rows:
        request_id = row.get("request_id", "")
        if present(request_id):
            by_request[request_id] += 1
    print(f"- option counts per request: {distribution(by_request.values())}")
    for column in [
        "payment_method",
        "payment_type",
        "number_of_payments",
        "payment_frequency_days",
        "payment_amount",
        "financing_fee",
    ]:
        if column in options.columns:
            print(f"- {column}: {format_counter(counter_for(options.rows, column), 25)}")
    if {"payment_method", "number_of_payments"}.issubset(options.columns):
        combo = Counter(f"{row.get('payment_method')} x {row.get('number_of_payments')}" for row in options.rows)
        print(f"- method x number_of_payments: {format_counter(combo, 25)}")


def inspect_currencies(tables: dict[str, Table]) -> None:
    print_section("Currencies And Exchange-Rate Coverage")
    profiles = tables.get("financial_profiles.csv")
    events = tables.get("financial_events.csv")
    rates = tables.get("exchange_rates.csv")
    requests = tables.get("requests.csv")
    options = tables.get("request_payment_options.csv")

    if profiles and "home_currency" in profiles.columns:
        print(f"- profile home_currency: {format_counter(counter_for(profiles.rows, 'home_currency'))}")
    for table in [events, requests, options]:
        if table:
            for column in table.columns:
                if "currency" in column.lower():
                    print(f"- {table.name}.{column}: {format_counter(counter_for(table.rows, column))}")

    if not (profiles and events and rates):
        return
    home_by_user = {row.get("user_id", ""): row.get("home_currency", "") for row in profiles.rows}
    rate_keys = {
        (row.get("rate_date", ""), row.get("from_currency", ""), row.get("to_currency", ""))
        for row in rates.rows
        if {"rate_date", "from_currency", "to_currency"}.issubset(rates.columns)
    }
    print(f"- exchange rate pairs: {format_counter(Counter(f'{src}->{dst}' for _, src, dst in rate_keys), 20)}")
    print(f"- exchange rate dates: {distribution(Counter(date for date, _, _ in rate_keys).values())} rates/day")

    date_column = "settlement_date" if "settlement_date" in events.columns else "event_date"
    foreign_rows = []
    missing_rates = Counter()
    if "currency" in events.columns and date_column in events.columns:
        for row in events.rows:
            currency = row.get("currency", "")
            home = home_by_user.get(row.get("user_id", ""), "")
            if present(currency) and present(home) and currency != home:
                foreign_rows.append(row)
                key = (row.get(date_column, ""), currency, home)
                if key not in rate_keys:
                    missing_rates[key] += 1
    print(f"- foreign-currency financial events: {len(foreign_rows)}")
    print(f"- missing direct event-date rate keys: {sum(missing_rates.values())}")
    if missing_rates:
        print(f"- missing rate key examples: {format_counter(Counter({str(key): count for key, count in missing_rates.items()}), 8)}")


def inspect_evidence(tables: dict[str, Table], dataset_dir: Path) -> None:
    print_section("Messages And Images")
    messages = tables.get("messages.csv")
    images = tables.get("images.csv")
    events = tables.get("financial_events.csv")

    if messages:
        for column in ["user_id", "request_id", "related_event_id"]:
            if column in messages.columns:
                nonempty = len(values(messages.rows, column))
                print(f"- messages with {column}: {nonempty}/{len(messages.rows)}")
        if {"request_id", "related_event_id"}.issubset(messages.columns):
            request_only = sum(1 for row in messages.rows if present(row.get("request_id")) and not present(row.get("related_event_id")))
            event_linked = sum(1 for row in messages.rows if present(row.get("related_event_id")))
            print(f"- message linkage shape: request_only={request_only}, event_linked={event_linked}")

    if images:
        image_dir = dataset_dir / "media" / "images"
        expected_files = [image_dir / f"{row.get('image_id', '')}.png" for row in images.rows if present(row.get("image_id"))]
        existing = sum(1 for path in expected_files if path.exists())
        print(f"- images metadata rows: {len(images.rows)}, png files found={existing}/{len(expected_files)}")
        for column in ["user_id", "request_id", "related_event_id"]:
            if column in images.columns:
                print(f"- images with {column}: {len(values(images.rows, column))}/{len(images.rows)}")

    if events and images and "amount" in events.columns and "event_id" in events.columns:
        image_events = set(values(images.rows, "related_event_id")) if "related_event_id" in images.columns else set()
        missing_amount_events = [row for row in events.rows if is_missing(row.get("amount"))]
        with_image = [row for row in missing_amount_events if row.get("event_id", "") in image_events]
        without_image = [row for row in missing_amount_events if row.get("event_id", "") not in image_events]
        print(f"- financial events with missing amount: {len(missing_amount_events)}")
        print(f"- missing-amount events with linked image evidence: {len(with_image)}")
        print(f"- missing-amount events without linked image evidence: {len(without_image)}")
        if with_image and "event_type" in events.columns:
            print(f"- missing-amount image-backed event types: {format_counter(counter_for(with_image, 'event_type'))}")


def inspect_duplicates_and_conflicts(tables: dict[str, Table]) -> None:
    print_section("Duplicate And Conflict Signals")
    for name in sorted(tables):
        table = tables[name]
        full_rows = [tuple(row.get(column, "") for column in table.columns) for row in table.rows]
        exact_duplicates = len(full_rows) - len(set(full_rows))
        if exact_duplicates:
            print(f"- {name}: exact_duplicate_rows={exact_duplicates}")

    events = tables.get("financial_events.csv")
    if events and {"user_id", "event_date", "amount", "currency", "event_type"}.issubset(events.columns):
        grouped = defaultdict(list)
        for row in events.rows:
            key = (
                row.get("user_id", ""),
                row.get("event_date", ""),
                row.get("amount", ""),
                row.get("currency", ""),
                row.get("event_type", ""),
            )
            grouped[key].append(row)
        duplicate_like = [rows for rows in grouped.values() if len(rows) > 1]
        status_conflicts = sum(1 for rows in duplicate_like if len({row.get("status", "") for row in rows}) > 1)
        linked_groups = sum(1 for rows in duplicate_like if any(present(row.get("linked_event_id")) for row in rows))
        print(f"- financial_events duplicate-like groups by user/date/amount/currency/type: {len(duplicate_like)}")
        print(f"- duplicate-like groups with status conflicts: {status_conflicts}")
        print(f"- duplicate-like groups with linked_event_id: {linked_groups}")


def inspect_samples(tables: dict[str, Table]) -> None:
    sample = tables.get("sample_requests.csv")
    if not sample:
        return
    print_section("Solved Sample Request Characteristics")
    print(f"- solved sample rows: {len(sample.rows)}")
    for column in [
        "request_type",
        "allows_partial_payment",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
    ]:
        if column in sample.columns:
            counter = counter_for(sample.rows, column)
            if column in {"payment_plan", "spending_changes_needed"}:
                none_count = counter.get("none", 0)
                non_none = len(sample.rows) - none_count - count_missing(sample.rows, column)
                print(f"- {column}: none={none_count}, non_none={non_none}, missing={count_missing(sample.rows, column)}")
            else:
                print(f"- {column}: {format_counter(counter, 20)}")
    if {"affordability_status", "recommended_payment_method"}.issubset(sample.columns):
        combo = Counter(
            f"{row.get('affordability_status')} -> {row.get('recommended_payment_method')}" for row in sample.rows
        )
        print(f"- status -> method: {format_counter(combo, 20)}")
    if {"recommended_payment_method", "payment_plan"}.issubset(sample.columns):
        blank_plan_by_method = Counter(
            row.get("recommended_payment_method", "unknown")
            for row in sample.rows
            if is_missing(row.get("payment_plan"))
        )
        print(f"- blank payment_plan by method: {format_counter(blank_plan_by_method, 20)}")
    if {"recommended_payment_method", "spending_changes_needed"}.issubset(sample.columns):
        changes_by_method = Counter(
            row.get("recommended_payment_method", "unknown")
            for row in sample.rows
            if present(row.get("spending_changes_needed"))
        )
        print(f"- nonblank spending_changes_needed by method: {format_counter(changes_by_method, 20)}")
    if {"amount_safe_to_pay", "requested_amount"}.issubset(sample.columns):
        full_safe = 0
        partial_safe = 0
        zero_safe = 0
        for row in sample.rows:
            try:
                safe = float(row.get("amount_safe_to_pay", ""))
                requested = float(row.get("requested_amount", ""))
            except ValueError:
                continue
            full_safe += int(safe == requested)
            partial_safe += int(0 < safe < requested)
            zero_safe += int(safe == 0)
        print(f"- amount_safe_to_pay relation: full={full_safe}, partial={partial_safe}, zero={zero_safe}")


def validate_output_template(tables: dict[str, Table]) -> None:
    output = tables.get("output.csv")
    if not output:
        return
    print_section("Output Template")
    print(f"- columns match required order: {output.columns == EXPECTED_OUTPUT_COLUMNS}")
    print(f"- rows in dataset/output.csv: {len(output.rows)}")


def load_tables(dataset_dir: Path) -> dict[str, Table]:
    csv_paths = sorted(dataset_dir.glob("*.csv"), key=lambda path: path.name)
    return {path.name: read_csv(path) for path in csv_paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to the dataset directory. Defaults to <repo root>/dataset.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else repo_root / "dataset"
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    tables = load_tables(dataset_dir)
    print(f"# Dataset Contract Inspection")
    print(f"dataset_dir={dataset_dir}")
    inspect_schema(tables)
    inspect_primary_keys(tables)
    inspect_nulls(tables)
    inspect_vocabularies(tables)
    inspect_join_integrity(tables)
    inspect_event_patterns(tables)
    inspect_recurring_patterns(tables)
    inspect_payment_options(tables)
    inspect_currencies(tables)
    inspect_evidence(tables, dataset_dir)
    inspect_duplicates_and_conflicts(tables)
    inspect_samples(tables)
    validate_output_template(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
