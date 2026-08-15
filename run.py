"""memharness - one dataset, one prompt, one judge, one metric, four memory systems.

    python run.py [--config config.yaml] [--systems mem0,bm25]

Writes results/<label>.json and results/<label>.md.
"""

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import yaml

os.environ.setdefault("MEM0_TELEMETRY", "False")  # no phoning home from a benchmark

import meter  # noqa: E402

meter.install()
from meter import METER  # noqa: E402

import judge as J  # noqa: E402
import systems as S  # noqa: E402

HERE = Path(__file__).parent


# ---------------------------------------------------------------- dataset

def load_turns_and_qa(cfg):
    path = (HERE / cfg["dataset"]["path"]).resolve()
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    wanted = set(cfg["dataset"]["conversation_ids"])
    convs = [c for c in data if c["sample_id"] in wanted]
    if not convs:
        sys.exit(f"no conversation matched {wanted} in {path}")

    turns, qa = [], []
    for conv in convs:
        c = conv["conversation"]
        a, b = c["speaker_a"], c["speaker_b"]
        sids = sorted(
            (m.group(1) for k in c if (m := re.fullmatch(r"session_(\d+)", k))), key=int
        )
        for sid in sids:
            date = c.get(f"session_{sid}_date_time", "")
            for t in c[f"session_{sid}"]:
                txt = t.get("text", "")
                if not txt:
                    continue
                turns.append(
                    {
                        "session": sid,
                        "date": date,
                        "speaker": t["speaker"],
                        "is_a": t["speaker"] == a,
                        "dia_id": t.get("dia_id", ""),
                        "render": f"[{date}] {t['speaker']}: {txt}",
                    }
                )
        for q in conv["qa"]:
            if q.get("category") in cfg["dataset"]["exclude_categories"]:
                continue
            if "answer" not in q:
                continue
            qa.append(
                {
                    "question": q["question"],
                    "gold": str(q["answer"]),
                    "category": q.get("category"),
                }
            )
    return turns, qa, (a, b), sha, str(path)


def sample_qa(qa, cfg):
    n, mode = cfg["dataset"]["n_questions"], cfg["dataset"]["sample"]
    rng = random.Random(cfg["run"]["seed"])
    if mode == "first":
        return qa[:n]
    if mode == "random":
        return rng.sample(qa, min(n, len(qa)))
    buckets = {}
    for q in qa:
        buckets.setdefault(q["category"], []).append(q)
    for v in buckets.values():
        rng.shuffle(v)
    out, cats = [], sorted(buckets)
    i = 0
    while len(out) < min(n, len(qa)):
        c = cats[i % len(cats)]
        if buckets[c]:
            out.append(buckets[c].pop())
        i += 1
        if all(not buckets[c] for c in cats):
            break
    return out


# ---------------------------------------------------------------- llm

def make_chat(cfg, spec):
    import ollama

    client = ollama.Client(host=cfg["backend"]["base_url"])

    def chat(prompt):
        r = client.chat(
            model=spec["model"],
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": spec["temperature"],
                "num_predict": spec["num_predict"],
                "seed": spec["seed"],
                "top_p": 1.0,
            },
        )
        return r.message.content or ""

    return chat


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[k]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--systems", default=None, help="comma list, overrides config")
    ap.add_argument("--n", type=int, default=None, help="question count, overrides config")
    ap.add_argument("--label", default=None, help="output label, overrides config")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / args.config).read_text())
    if args.systems:
        cfg["systems"] = args.systems.split(",")
    if args.n:
        cfg["dataset"]["n_questions"] = args.n
    if args.label:
        cfg["run"]["label"] = args.label

    turns, qa_all, (spk_a, spk_b), sha, dpath = load_turns_and_qa(cfg)
    exp = cfg["dataset"].get("sha256_expected")
    if exp and exp != sha:
        sys.exit(f"dataset sha mismatch: expected {exp}, got {sha}")
    qa = sample_qa(qa_all, cfg)

    answer = make_chat(cfg, cfg["models"]["answer"])
    judge_chat = make_chat(cfg, cfg["models"]["judge"])
    ans_tmpl, judge_tmpl = cfg["prompts"]["answer"], cfg["prompts"]["judge"]
    thr = cfg["scoring"]["deterministic_f1_threshold"]

    print(f"dataset {dpath}\n  sha256 {sha}\n  turns {len(turns)}  "
          f"questions {len(qa)}/{len(qa_all)} eligible", flush=True)

    results = {}
    for sname in cfg["systems"]:
        sys_obj = S.REGISTRY[sname](cfg)
        print(f"\n== {sname} ==", flush=True)

        t0 = time.perf_counter()
        with METER.scope(sname, "ingest"):
            sys_obj.ingest(turns)
        ingest_ms = (time.perf_counter() - t0) * 1000
        print(f"  ingest {ingest_ms / 1000:.1f}s  {METER.totals(sname, 'ingest')}", flush=True)

        rows = []
        for i, q in enumerate(qa, 1):
            t = time.perf_counter()
            with METER.scope(sname, "retrieve"):
                ctx = sys_obj.retrieve(q["question"])
            ret_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            with METER.scope(sname, "answer"):
                pred = answer(
                    ans_tmpl.format(
                        speaker_a=spk_a, speaker_b=spk_b, context=ctx, question=q["question"]
                    )
                )
            ans_ms = (time.perf_counter() - t) * 1000

            det_ok, f1 = J.deterministic(pred, q["gold"], thr)
            with METER.scope(sname, "judge"):
                llm_ok = J.llm_judge(judge_chat, judge_tmpl, q["question"], q["gold"], pred)

            rows.append(
                {
                    "question": q["question"],
                    "category": q["category"],
                    "gold": q["gold"],
                    "prediction": pred.strip(),
                    "context_chars": len(ctx),
                    "retrieve_ms": ret_ms,
                    "answer_ms": ans_ms,
                    "query_ms": ret_ms + ans_ms,
                    "deterministic_correct": det_ok,
                    "token_f1": f1,
                    "llm_judge_correct": llm_ok,
                }
            )
            print(f"  [{i:>3}/{len(qa)}] {'J' if llm_ok else '.'}"
                  f"{'D' if det_ok else '.'} {ret_ms + ans_ms:7.0f}ms  {pred.strip()[:50]!r}",
                  flush=True)

        results[sname] = {"ingest_ms": ingest_ms, "rows": rows}

    # ------------------------------------------------------------ metrics
    price = cfg["pricing"]
    summary = {}
    for sname, r in results.items():
        rows = r["rows"]
        n = len(rows)
        solved_j = sum(x["llm_judge_correct"] for x in rows)
        solved_d = sum(x["deterministic_correct"] for x in rows)
        ing = METER.totals(sname, "ingest")
        q_in = METER.totals(sname, "retrieve")["prompt_tokens"] + METER.totals(sname, "answer")["prompt_tokens"]
        q_out = METER.totals(sname, "retrieve")["completion_tokens"] + METER.totals(sname, "answer")["completion_tokens"]
        q_embed = METER.totals(sname, "retrieve")["embed_tokens"]

        def dollars(inp, outp, emb):
            return (
                inp / 1e6 * price["input_per_mtok"]
                + outp / 1e6 * price["output_per_mtok"]
                + emb / 1e6 * price["embed_per_mtok"]
            )

        ing_usd = dollars(ing["prompt_tokens"] - ing["embed_tokens"], ing["completion_tokens"], ing["embed_tokens"])
        qry_usd = dollars(q_in - q_embed, q_out, q_embed)
        total_tokens = ing["prompt_tokens"] + ing["completion_tokens"] + q_in + q_out
        summary[sname] = {
            "n": n,
            "accuracy_llm_judge": solved_j / n,
            "accuracy_deterministic": solved_d / n,
            "solved_llm_judge": solved_j,
            "solved_deterministic": solved_d,
            "mean_token_f1": statistics.mean(x["token_f1"] for x in rows),
            "ingest_wall_s": r["ingest_ms"] / 1000,
            "ingest_llm_calls": ing["calls"],
            "ingest_prompt_tokens": ing["prompt_tokens"],
            "ingest_completion_tokens": ing["completion_tokens"],
            "query_prompt_tokens": q_in,
            "query_completion_tokens": q_out,
            "mean_context_chars": statistics.mean(x["context_chars"] for x in rows),
            "p50_query_ms": pct([x["query_ms"] for x in rows], 50),
            "p95_query_ms": pct([x["query_ms"] for x in rows], 95),
            "p50_retrieve_ms": pct([x["retrieve_ms"] for x in rows], 50),
            "p95_retrieve_ms": pct([x["retrieve_ms"] for x in rows], 95),
            "tokens_per_solved_task": (total_tokens / solved_j) if solved_j else None,
            "usd_ingest_DERIVED": ing_usd,
            "usd_query_total_DERIVED": qry_usd,
            "usd_per_solved_task_DERIVED": ((ing_usd + qry_usd) / solved_j) if solved_j else None,
        }

    out = {
        "label": cfg["run"]["label"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"path": dpath, "sha256": sha, "turns": len(turns),
                    "questions_used": len(qa), "questions_eligible": len(qa_all)},
        "config": cfg,
        "summary": summary,
        "per_question": {k: v["rows"] for k, v in results.items()},
        "calls": [c.__dict__ for c in METER.calls],
    }
    odir = HERE / cfg["run"]["out_dir"]
    odir.mkdir(exist_ok=True)
    (odir / f"{cfg['run']['label']}.json").write_text(json.dumps(out, indent=1))
    (odir / f"{cfg['run']['label']}.md").write_text(render_md(out))
    print("\n" + render_md(out))


def render_md(out):
    s, cfg = out["summary"], out["config"]
    L = []
    L.append(f"# memharness - {out['label']}\n")
    L.append(f"Generated {out['generated_utc']}. "
             f"Dataset `{Path(out['dataset']['path']).name}` sha256 `{out['dataset']['sha256'][:16]}…`, "
             f"{out['dataset']['turns']} turns, {out['dataset']['questions_used']} questions "
             f"(of {out['dataset']['questions_eligible']} eligible).\n")
    L.append(f"Identical across every row: answer model `{cfg['models']['answer']['model']}` "
             f"@ temp {cfg['models']['answer']['temperature']} seed {cfg['models']['answer']['seed']}, "
             f"judge `{cfg['models']['judge']['model']}`, top_k {cfg['retrieval']['top_k']}, "
             f"and the answer/judge prompts in `config.yaml`.\n")
    L.append("## Accuracy, cost, latency\n")
    L.append("| System | Acc (LLM judge) | Acc (deterministic) | Ingest tokens | Tokens/query | Tokens/solved | p50 query ms | p95 query ms |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in s.items():
        tps = f"{v['tokens_per_solved_task']:,.0f}" if v["tokens_per_solved_task"] else "n/a"
        tpq = (v["query_prompt_tokens"] + v["query_completion_tokens"]) / v["n"]
        L.append(
            f"| {k} | {v['accuracy_llm_judge']:.1%} ({v['solved_llm_judge']}/{v['n']}) "
            f"| {v['accuracy_deterministic']:.1%} ({v['solved_deterministic']}/{v['n']}) "
            f"| {v['ingest_prompt_tokens'] + v['ingest_completion_tokens']:,} "
            f"| {tpq:,.0f} | {tps} | {v['p50_query_ms']:.0f} | {v['p95_query_ms']:.0f} |"
        )
    L.append("\n## Derived dollars\n")
    L.append(f"Token counts above are measured. Dollars below are those counts times the "
             f"`pricing` block in `config.yaml` (`{cfg['pricing']['sheet_name']}`) and are "
             f"DERIVED, not billed. This run cost $0: everything ran on local Ollama.\n")
    L.append("| System | $ ingest (DERIVED) | $ all queries (DERIVED) | $/solved-task (DERIVED) |")
    L.append("|---|---|---|---|")
    for k, v in s.items():
        ups = f"${v['usd_per_solved_task_DERIVED']:.5f}" if v["usd_per_solved_task_DERIVED"] else "n/a"
        L.append(f"| {k} | ${v['usd_ingest_DERIVED']:.5f} | ${v['usd_query_total_DERIVED']:.5f} | {ups} |")
    L.append("\n## Ingest side\n")
    L.append("| System | Ingest wall s | Ingest LLM/embed calls | Mean context chars/query | p50 retrieve ms | p95 retrieve ms |")
    L.append("|---|---|---|---|---|---|")
    for k, v in s.items():
        L.append(f"| {k} | {v['ingest_wall_s']:.1f} | {v['ingest_llm_calls']} "
                 f"| {v['mean_context_chars']:,.0f} | {v['p50_retrieve_ms']:.1f} | {v['p95_retrieve_ms']:.1f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
