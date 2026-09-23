# Buy or Wait? — Hybrid AI Financial Decision Engine

## Overview

Deciding whether a purchase is affordable takes more than comparing its price
with a current account balance. A safe answer has to account for future income
and expenses, pending transactions, recurring obligations, minimum cash
reserves, available payment options, and financial evidence found in messages
or images.

Buy or Wait? is a hybrid AI financial reasoning system that reconstructs a
person's financial state, forecasts near-term cash flow, and recommends safe
ways to handle requested purchases.

Originally built for HackerRank Orchestrate 2026 · Ranked 357th out of ~3,000 participants globally.

## Why I Built It This Way

Unstructured financial evidence benefits from AI perception: messages, receipts,
and payroll documents rarely arrive as clean database records. Financial
arithmetic and safety decisions benefit from determinism, traceability, exact
constraints, and reproducibility.

The central design principle is:

> **LLM for perception; deterministic code for financial decisions.**

The system boundary is:

> **Use AI to understand evidence. Use deterministic software to decide what is safe.**

The model extracts structured evidence from unstructured sources. All
financial-state reconstruction, cash forecasting, payment safety checks, and
recommendation selection are deterministic and Decimal-based.

## Architecture

```text
typed ingestion
  -> multimodal evidence extraction
  -> financial-state resolution
  -> recurrence inference
  -> 90-day cash-flow forecast
  -> safe payment capacity
  -> candidate plan generation
  -> deterministic ranking and output
```

Each stage has one clear job: turn raw records into typed financial facts,
validate model-derived evidence before deterministic financial rules use it,
resolve only supported changes, then simulate cash flow before recommending an
action. The detailed modules keep joins, lifecycle resolution, recurrence,
planning, and output validation separate while sharing the same core financial
semantics.

## Evidence Extraction

Relevant messages and images are processed with GPT-5.6 Terra through the
OpenAI Responses API. The model returns typed financial claims, such as an
amount, currency, effective date, status, document role, or identifier role.

Model output is evidence, not truth. Claims are deterministically validated
before they can affect financial state, retain source provenance, and are cached
per source. Malformed claims are contained at claim level: for example, a value
such as `12%` in a monetary field is rejected rather than coerced into money or
allowed to abort sibling claims from the same response.

## Financial-State Resolution

Financial events are resolved field by field, preserving original event
provenance and a trace for each applied mutation. The resolver handles explicit
cancellations, settlements, amendments, linked lifecycle records, pending
transactions, and missing amounts supported by evidence.

Important protections include:

- pending debits remain reserved; pending credits are not available cash;
- missing amounts never become zero by default;
- foreign currency conversion uses only the supplied exact-date,
  `from_currency -> to_currency` exchange-rate rows;
- document metadata cannot overwrite cash semantics;
- statuses such as `Delivered` are not treated as financial settlement states;
- receipt, invoice, or document dates do not automatically replace a known
  cash settlement date;
- an amount confirmation can fill a missing amount without replacing a known
  event currency; explicit amount amendments retain amendment behavior.

## Recurrence And Forecasting

The recurrence layer detects weekly, biweekly, and monthly series when history
supports a deterministic cadence. Ambiguous or irregular patterns are retained
as non-projectable rather than turned into speculative future cash flows.

The 90-day forecast is the single safety oracle for planning. It avoids
double-counting a confirmed explicit salary occurrence and a synthetic salary
projection for the same payroll cycle. It reserves pending debits, excludes
pending credits from available cash, and requires the projected balance to stay
at or above `minimum_balance_to_keep` after every movement.

## Payment Planning

The capacity layer derives two baseline fields before optional spending changes:

- `amount_safe_to_pay`: the maximum safe amount on the request date;
- `earliest_date_for_full_payment`: the earliest date a one-off full payment is
  safe within the forecast horizon.

The planner then generates and verifies candidates for full payment, eligible
partial payment, supplied installment plans, waiting, and permitted
spending-change scenarios. Installment schedules are validated exactly as
supplied, never invented. Full payment, partial payment, installments, waiting,
and spending-change scenarios all pass through the same 90-day forecast engine,
giving the system one consistent definition of financial safety.

## Deterministic Recommendation Ranking

Valid candidates are ranked deterministically in this order:

1. Completion by the desired date
2. No spending changes
3. Minimum total paid
4. Earlier start
5. Fewer payments
6. Lowest `payment_option_id`

The selected candidate is serialized into the final recommendation fields only
after schedule, amount, ordering, and output invariants pass.

## Engineering Lessons And Limitations

Exact-description recurrence is deliberately conservative. Irregular habitual
spending, such as groceries or dining with rotating descriptions, may not have
enough stable cadence evidence to be projected.

A broader category-level forecasting experiment was evaluated for this gap. It
improved some public-sample metrics but also introduced additional incorrect
projections and degraded overall reliability, so it was reverted rather than
overfitting public examples. The current engine favors transparent, supported
recurrence evidence over a more aggressive spending heuristic.

## Testing And Validation

Focused regression tests cover the major layers: typed parsing, Decimal and FX
handling, evidence validation, lifecycle resolution, recurrence, forecasting,
capacity, payment planning, recommendation ranking, and final-output
validation. The end-to-end pipeline validates one output row for each of the
250 evaluation requests.

## What I Would Improve Next

- Better calibrated modeling for irregular habitual spending without
  double-counting or systematically over-projecting expense patterns.
- Persist model and provider usage metadata alongside evidence-cache entries.
- Add richer per-request forecast traces for debugging and auditability.

## Project Structure

```text
code/
├── data_layer.py                   # Typed CSV ingestion and indexes
├── event_normalization.py          # Cash-state and exact-date FX normalization
├── evidence_schema.py              # Typed evidence claims and validation
├── evidence_extraction.py          # Responses API client, cache, usage tracking
├── financial_state_resolution.py   # Lifecycle and field-level evidence resolution
├── recurrence_inference.py         # Supported recurring-series inference
├── financial_forecast.py           # 90-day balance simulation and scenarios
├── payment_capacity.py             # Safe-now capacity and earliest full date
├── payment_planning.py             # Valid plan-candidate generation
├── recommendation_selection.py     # Deterministic candidate ranking and output
├── main.py                         # Full orchestration entry point
├── tools/                          # Focused validation and inspection scripts
└── evaluation/                     # Runtime usage report
```

## Running The Project

Requires Python 3.11+ and the standard library. The evidence cache lives at
`.cache/evidence_extraction`.

To run from an already complete cache without provider calls:

```bash
python code/main.py
```

If evidence cache entries are missing, provide `OPENAI_API_KEY` through the
process environment and allow the pipeline to fill only those misses:

```bash
python code/main.py --run-real-api
```

Never put API keys in source code or commit secrets to the repository. A
successful run writes `output.csv` at the repository root and updates
`code/evaluation/usage_report.md`.

## Hackathon Context

This project began as an implementation of the HackerRank Orchestrate
September 2026 "Buy or Wait?" challenge. The original task contract, input
schema, and decision rules remain in [problem_statement.md](./problem_statement.md).
