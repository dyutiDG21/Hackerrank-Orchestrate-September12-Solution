# Runtime Usage Report

## Final Orchestration Pass

- Provider: OpenAI Responses API
- Model: `gpt-5.6-terra`
- New model calls: 0
- New input tokens: 0
- New cached input tokens: 0
- New output tokens: 0
- New total tokens: 0
- New average tokens per evaluation request: 0
- New estimated runtime cost: $0.00000000
- New estimated cost per evaluation request: $0.00000000
- Evidence cache entries reused: 137
- Evidence cache misses filled this pass: 0

## Historical Evidence Represented By The Final Output

- Cached evaluation-relevant sources: 137 (126 messages, 11 images)
- Provider calls represented: 137; the extraction pipeline performs one Responses API call per source cache miss.
- Historical input tokens: unavailable; cache payloads retain validated claims only, not provider usage.
- Historical cached input tokens: unavailable; not persisted in cache or local usage metadata.
- Historical output tokens: unavailable; not persisted in cache or local usage metadata.
- Historical total tokens and average per evaluation request: unavailable because the component token counts cannot be reconstructed.
- Historical estimated runtime cost and per-evaluation-request cost: unavailable because token usage cannot be reconstructed.

Cost formula when usage is available: input_tokens * $2.00/1M + cached_input_tokens * $0.20/1M + output_tokens * $12.00/1M.
This report covers solution runtime evidence extraction only, not Codex development usage.
