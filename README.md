# memharness

An agent-memory benchmark where every knob that could change a result lives in one config
file and is applied identically to every system, plus a token meter that counts what a
memory system spends **ingesting** a conversation, not only what it spends answering
questions about it.

Published comparisons in this space measure the author's own system directly and quote
competitors from their papers. The dataset, prompt, answering model, judge and metric all
differ across the rows being compared, so the numbers are not wrong so much as not
comparable. This runs four systems through one pipeline with one configuration.

---

## Results

LoCoMo, one conversation of 419 turns, 20 questions sampled from 152 eligible.
llama3.1:8b answering and judging, nomic-embed-text embedding, everything local.
Judge is an LLM; `det` is exact-match. Seed 0, temperature 0.

| system | acc | det | ingest tokens | query tokens | per question | p50 retrieve | mean context |
|---|---:|---:|---:|---:|---:|---:|---:|
| full context | 0.30 | 0.00 | 0 | 42,117 | 2,106 | 0.33 ms | 73,892 ch |
| recency window | 0.10 | 0.05 | 0 | 21,304 | 1,065 | 0.02 ms | 3,703 ch |
| BM25 | 0.50 | 0.25 | 0 | 11,302 | 565 | 0.97 ms | 1,754 ch |
| **mem0 2.0.18** | **0.65** | **0.30** | **76,450** | **6,039** | **302** | **76.66 ms** | **956 ch** |

**n=20. The accuracy column cannot support a ranking claim and is not offered as one.**
Confidence intervals at this size overlap for every pair. mem0 also ran dense-only, because
the hybrid keyword path needed a download that would not complete offline, so 0.65 is a
floor rather than a result. The columns that carry weight here are the cost and latency
ones, which are counted rather than judged.

---

## What the numbers say

### Two things that support mem0's own argument, more strongly than mem0 states it

**1. Stuffing the full transcript is not merely expensive, it is worse.** Full context scored
**0.30 against mem0's 0.65** while spending **7x more tokens per question**. mem0's public
material argues the cost side of this (roughly 6,900 tokens per retrieval against 25,000+
for full context). The accuracy side is the stronger half of the same argument and gets less
airtime. Handing a model 73,892 characters made it worse at answering than handing it 956.

**2. mem0 won on the tightest context of any system tested.** 956 characters mean, against
BM25's 1,754 and full context's 73,892. Best accuracy on **77x less context** than the
full-transcript baseline and **1.8x less** than lexical retrieval. Whatever the extraction
pipeline is doing, it is not winning by retrieving more, which is the interesting way to win.

### Three things the published accounting does not cover

**3. Ingest costs 12.7x the entire query workload.** 76,450 tokens to read one 419-turn
conversation, against 6,039 tokens for all twenty questions combined. Per-retrieval figures
describe the small half of the bill. This is not a criticism of the number, it is a
different number that nobody publishes.

Where it lands, on a dollars-per-solved-task basis against BM25:

```
mem0 fixed cost      76,450 tokens, paid once per conversation
mem0 marginal          302 tokens per question   (BM25: 565)
crossover           ~220 questions asked against that same conversation
```

Under roughly 220 questions per conversation, the retrieval saving is real and the total is
not. Over it, mem0 wins and keeps winning. **The number that decides which regime a buyer is
in has not been published by anyone, including mem0.** A support product answering four
questions per ticket and a research agent interrogating one corpus for a month sit on
opposite sides of that line, and today they read the same marketing page.

**4. Retrieval is 76.66 ms at p50 and 97.69 ms at p95, against BM25's 0.97 ms.** Roughly
**79x**. mem0's research page reports that median latency "stays flat at +1 ms", which
concerns memory-decay operations rather than the retrieve path. For an interactive agent,
77 ms on every turn before the model has produced a single token is a real budget item, and
it is absent from the public numbers.

**5. Ingest is 19 LLM calls and 41 embedding calls for one conversation**, with extraction
output accounting for **14,745 of the 76,450 tokens**, about 27%. The extraction step writes
a great deal relative to what it reads. The call count is architectural and holds regardless
of which model serves it, which makes it the number that decides whether ingestion can sit
in a request path or has to be a background job.

### One finding that applies to everybody's benchmark, not just mem0's

**6. Swapping an LLM judge for exact match does not shift scores by a constant.** It
reorders the size of the gaps:

| system | LLM judge | exact match | change |
|---|---:|---:|---|
| full context | 0.30 | **0.00** | everything |
| recency window | 0.10 | 0.05 | halved |
| BM25 | 0.50 | 0.25 | halved |
| mem0 | 0.65 | 0.30 | halved |

Three systems halve. Full context goes to **zero**: on twenty questions it did not produce
a single exactly-correct answer, while an LLM judge credited it with six. Long-context
answers are verbose and approximately right, which is exactly what an LLM judge rewards and
exact match refuses. Any leaderboard in this space using an LLM judge is measuring a blend
of retrieval quality and answer style, and the blend is not uniform across architectures.
That is worth knowing before comparing two published numbers scored by different judges.

---

## How it works

```
config.yaml ── one file: dataset slice, prompt, answering model, temperature,
     │         seed, judge, top_k, role mapping, pricing
     ▼
run.py ── for each system:
     │
     ├── INGEST  feed the conversation in, meter every call the system makes
     │            internally  (mem0: 19 LLM + 41 embed;  BM25: an index build;
     │            full context and recency: nothing)
     │
     ├── QUERY   for each question: retrieve → build context → answer
     │            meter every call, record wall time per stage
     │
     └── SCORE   exact match and an LLM judge, both recorded, neither discarded
     ▼
results/*.json ── per-question rows plus every individual model call
```

Four systems behind one interface, so nothing about the pipeline can differ between them:

- **full context** puts the entire transcript in the prompt. The expensive control.
- **recency window** keeps the last N messages. The cheap control.
- **BM25** lexical retrieval over raw turns. Pure Python, no model, no ingest cost.
- **mem0** the real package, extraction pipeline and all.

Everything that could be tilted in someone's favour is a named line in `config.yaml` rather
than a default buried in code. `role_mapping` is the clearest example: how you map a
transcript's speakers onto chat roles silently moves a competitor's score, so it is an
explicit knob with its options written down.

### The token meter

The interesting part. mem0 does `from ollama import Client` at module import time, so
replacing `ollama.Client` after the fact does not catch it. `meter.py` therefore rebinds the
already-imported names too:

```python
ollama.Client = MeteredClient
import mem0.embeddings.ollama as _e
import mem0.llms.ollama as _l
_l.Client = MeteredClient
_e.Client = MeteredClient
```

That is what makes the ingest column possible. Without it you can only meter the calls your
own harness makes, which is precisely why published figures cover the query path and stop
there. Every call any system makes, including inside mem0's own fact-extraction pipeline,
lands in `results/*.json` with its stage, model, prompt tokens, completion tokens and wall
time.

### How mem0 is wired in

The public package, unmodified, from PyPI:

```python
from mem0 import Memory                     # mem0ai 2.0.18
self.mem = Memory.from_config({...})        # Ollama LLM + Ollama embedder + local Qdrant
self.mem.add(messages=msgs, user_id=uid, infer=True)   # ingest, extraction path on
self.mem.search(query=q, top_k=k, filters={"user_id": uid})
```

`infer=True` is mem0's default extraction path, which is the thing being measured. No cloud,
no API key, no network egress. The whole run is reproducible on a laptop for zero dollars,
and that is only possible because mem0 ships something an outsider can actually run. Most
systems in this category cannot be independently benchmarked at all.

---

## Run it

```bash
uv venv --python 3.12 ../work/.venv
uv pip install --python ../work/.venv/bin/python -r requirements.txt
../work/.venv/bin/python -m spacy download en_core_web_sm

cd memharness && ../work/.venv/bin/python run.py                 # defaults from config.yaml
../work/.venv/bin/python run.py --systems mem0 --n 152           # the full eligible set
../work/.venv/bin/python run.py --systems bm25,full_context      # baselines only
```

Needs Ollama running locally with `llama3.1:8b` and `nomic-embed-text`. Change `--n` and
everything else follows from the config.

---

## Limitations, stated plainly

- **n=20 supports no accuracy ranking.** Intervals overlap for every pair. Rerun at `--n 152`
  before quoting the accuracy column at anyone.
- **One conversation.** 419 turns, one LoCoMo sample. The crossover arithmetic depends on
  conversation length; a shorter conversation is cheaper to ingest and crosses over sooner.
- **mem0 ran dense-only.** The hybrid keyword path needed a download that would not complete
  offline, so 0.65 is a floor.
- **Local 8B weights answer and judge.** A frontier model would change every accuracy number
  here. The cost and latency structure is architectural and would survive; the accuracy
  column would not.
- **The judge is the same model family as the answerer**, which is a known source of bias.
  Exact match is reported alongside precisely so the judge is not the only witness.
- **Wall-clock ingest of 782 s is model-dependent** and says more about local 8B throughput
  than about mem0. The 19 LLM calls and 41 embedding calls are the architectural figures;
  the seconds are not.
