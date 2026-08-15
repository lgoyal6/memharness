# memharness - baselines-n20

Generated 2026-08-14T18:45:45Z. Dataset `locomo10.json` sha256 `79fa87e90f040813…`, 419 turns, 20 questions (of 152 eligible).

Identical across every row: answer model `llama3.1:8b` @ temp 0.0 seed 20260814, judge `llama3.1:8b`, top_k 10, and the answer/judge prompts in `config.yaml`.

## Accuracy, cost, latency

| System | Acc (LLM judge) | Acc (deterministic) | Ingest tokens | Tokens/query | Tokens/solved | p50 query ms | p95 query ms |
|---|---|---|---|---|---|---|---|
| full_context | 30.0% (6/20) | 0.0% (0/20) | 0 | 2,106 | 7,020 | 4564 | 10577 |
| recency_window | 10.0% (2/20) | 5.0% (1/20) | 0 | 1,065 | 10,652 | 460 | 628 |
| bm25 | 50.0% (10/20) | 25.0% (5/20) | 0 | 565 | 1,130 | 2347 | 2890 |

## Derived dollars

Token counts above are measured. Dollars below are those counts times the `pricing` block in `config.yaml` (`gpt-4o-mini + text-embedding-3-small list price`) and are DERIVED, not billed. This run cost $0: everything ran on local Ollama.

| System | $ ingest (DERIVED) | $ all queries (DERIVED) | $/solved-task (DERIVED) |
|---|---|---|---|
| full_context | $0.00000 | $0.00682 | $0.00114 |
| recency_window | $0.00000 | $0.00324 | $0.00162 |
| bm25 | $0.00000 | $0.00175 | $0.00018 |

## Ingest side

| System | Ingest wall s | Ingest LLM/embed calls | Mean context chars/query | p50 retrieve ms | p95 retrieve ms |
|---|---|---|---|---|---|
| full_context | 0.0 | 0 | 73,892 | 0.3 | 2.5 |
| recency_window | 0.0 | 0 | 3,703 | 0.0 | 0.3 |
| bm25 | 0.0 | 0 | 1,754 | 1.0 | 7.8 |
