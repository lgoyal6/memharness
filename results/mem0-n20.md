# memharness - mem0-n20

Generated 2026-08-14T18:59:57Z. Dataset `locomo10.json` sha256 `79fa87e90f040813…`, 419 turns, 20 questions (of 152 eligible).

Identical across every row: answer model `llama3.1:8b` @ temp 0.0 seed 20260814, judge `llama3.1:8b`, top_k 10, and the answer/judge prompts in `config.yaml`.

## Accuracy, cost, latency

| System | Acc (LLM judge) | Acc (deterministic) | Ingest tokens | Tokens/query | Tokens/solved | p50 query ms | p95 query ms |
|---|---|---|---|---|---|---|---|
| mem0 | 65.0% (13/20) | 30.0% (6/20) | 76,450 | 302 | 6,345 | 1204 | 1506 |

## Derived dollars

Token counts above are measured. Dollars below are those counts times the `pricing` block in `config.yaml` (`gpt-4o-mini + text-embedding-3-small list price`) and are DERIVED, not billed. This run cost $0: everything ran on local Ollama.

| System | $ ingest (DERIVED) | $ all queries (DERIVED) | $/solved-task (DERIVED) |
|---|---|---|---|
| mem0 | $0.01514 | $0.00091 | $0.00123 |

## Ingest side

| System | Ingest wall s | Ingest LLM/embed calls | Mean context chars/query | p50 retrieve ms | p95 retrieve ms |
|---|---|---|---|---|---|
| mem0 | 791.1 | 60 | 956 | 76.7 | 97.7 |
