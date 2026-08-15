# memharness

An apples-to-apples agent-memory benchmark. One dataset, one answer prompt, one answering
model, one judge, one metric, applied identically to every system in the table.

The reason it exists: agent-memory numbers are usually produced by whoever is being
measured, on a harness nobody else can run. This is a harness anybody can run. Every knob
that can move a number is a line in `config.yaml`, not a constant inside the Python.

It also reports something published memory tables tend to leave out: the **ingest-side
token cost**. Retrieval accuracy and search latency are the usual columns. The tokens a
memory system burns writing to itself are what you actually pay for, and they are
invisible in most comparisons.

## What is measured

| Axis | How |
|---|---|
| Accuracy | Two scorers, both computed for every system: a pluggable LLM judge, and a deterministic normalized-token-F1 + containment scorer that needs no model at all |
| Cost | Real token counts. `meter.py` wraps the Ollama client and records `prompt_eval_count` / `eval_count` on **every** call, including the calls Mem0 makes inside its own extraction pipeline. Dollars are those measured counts times a price sheet in `config.yaml`, and are labelled DERIVED |
| Latency | Per-question wall clock, split into retrieve and answer, reported p50 and p95 |

## Systems

| System | What it is |
|---|---|
| `mem0` | Mem0 OSS (`pip install mem0ai`), fully local: Ollama LLM, Ollama embedder, local Qdrant. No mem0 cloud, no API key |
| `full_context` | The entire transcript in the prompt. The honest ceiling baseline |
| `recency_window` | The last N turns. What you get with no memory system at all |
| `bm25` | Lexical retrieval over the raw transcript. No LLM anywhere in the retrieval loop |

## Run it

Prerequisites: [Ollama](https://ollama.com) running locally, and [uv](https://docs.astral.sh/uv/).

```bash
# from the repo root
ollama serve &
ollama pull llama3.1:8b
ollama pull nomic-embed-text

uv venv --python 3.12 work/.venv
VIRTUAL_ENV=work/.venv uv pip install -r memharness/requirements.txt
work/.venv/bin/python -m spacy download en_core_web_sm

mkdir -p work/data
curl -sL -o work/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

cd memharness && ../work/.venv/bin/python run.py
```

`config.yaml` already holds the exact slice, models, seed and prompts of the published run,
so the bare command above reproduces it. `--n`, `--systems` and `--label` override the
config for quick experiments.

Results land in `memharness/results/<label>.json` (per-question predictions and a full
call-by-call token log) and `memharness/results/<label>.md` (the tables).

## Swapping things out

Everything below is a one-line edit in `config.yaml`:

- **Judge model** (`models.judge`) - any Ollama model. To use a hosted judge, replace the
  `make_chat` callable in `run.py`; `judge.llm_judge` takes the chat function as an argument
  precisely so the judge is not welded to a provider.
- **Answering model** (`models.answer`), temperature, seed, `num_predict`.
- **Dataset slice** (`dataset.conversation_ids`, `n_questions`, `sample`,
  `exclude_categories`). `sha256_expected` pins the exact data file so a run cannot
  silently drift onto different data.
- **Retrieval budget** (`retrieval.top_k`) - applied identically to every retrieval system.
- **Price sheet** (`pricing`) - re-prices the derived `$` column without re-running anything.
- **Mem0 role mapping** (`mem0.role_mapping`) - an explicit knob, because how you map
  conversation participants onto `user`/`assistant` roles is exactly the kind of choice
  that quietly changes a system's score in a head-to-head. It should be written down.

## Honesty notes

- **Token counts are measured. Dollars are derived.** The run itself costs $0; it is all
  local Ollama. The `$` columns are measured tokens times the configured price sheet, and
  are marked DERIVED everywhere they appear. Verify the sheet against the provider's
  current pricing page before quoting the dollar figures.
- **The absolute accuracy numbers are a property of the backbone.** They were produced with
  `llama3.1:8b` answering, which is a small model. They are not comparable to any published
  number produced on a different backbone. What is comparable is the **relative** standing
  of the four systems, because they share the backbone, the prompt, the seed, and the judge.
- **An abstention (`NO ANSWER`) is never scored correct**, under either scorer, for any
  system. Small judges will otherwise happily mark an abstention CORRECT.
- **LoCoMo category 5 is excluded** because those items carry no `answer` field. Mem0's and
  Zep's published evaluations both drop it too.
- **The published run measured Mem0 in its dense-only configuration.** Mem0's hybrid keyword
  path needs `mem0ai[extras]`, whose `fastembed` model download stalled on this machine, so
  `fastembed` was uninstalled and the run forced offline with `HF_HUB_OFFLINE=1`. That is a
  handicap on the `mem0` row and its accuracy should be read as a floor. To remove it:
  `uv pip install "mem0ai[extras]"`, drop `HF_HUB_OFFLINE`, rerun. Benchmarking a system in
  a configuration its authors would not recognise is the main way these comparisons go
  wrong, so the deviation is stated here rather than buried.

## Files

```
config.yaml   every knob, in one place
run.py        the runner: load, slice, ingest, retrieve, answer, judge, tabulate
systems.py    the four systems behind one two-method interface
judge.py      deterministic scorer + pluggable LLM judge
meter.py      metered Ollama client; real token counts for every call, mem0's included
```

Dataset: [snap-research/locomo](https://github.com/snap-research/locomo), `data/locomo10.json`.
