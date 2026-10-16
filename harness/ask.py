#!/usr/bin/env python3
"""Answer benchmark questions with a fixed model under one context condition
(no tools). Faithful to SWE-QA's `llm_direct` protocol: the answering
instruction is the upstream system message verbatim, the user turn is
"Repository: X\\n Question: Y", temperature 0 (upstream default). The only thing
that changes between conditions is what else the model is given:

  direct  nothing but the repository name (SWE-QA's own baseline)
  inject  the repository's AOCI volumes (Root + Meta + Code) as text in the
          system prompt, behind a preamble (v1: "your knowledge" — the pilot-1
          artefact; v2: "additional file-level context, answer at the asked
          granularity, no bare paths")

Runs --workers requests at once (default 7, 1..20). Every record carries the
full usage block, the parameters in effect, and `retried` when the symmetric
degenerate-answer guard fired (empty / too short / cut off → one nudge).

Usage:
  ask.py --repo-name xarray --questions benchmarks/swe-qa/xarray.jsonl \
         --condition inject --repo-root repos/xarray --out runs/x/answers-inject.jsonl [--preamble v2]
  ask.py ... --condition direct --out runs/x/answers-direct.jsonl
  ask.py ... --dry-run
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import llm  # noqa: E402

# SWE-QA Experiment/Script/llm_direct/main.py, verbatim.
UPSTREAM_SYSTEM = (
    "You are a direct answer generator. Provide ONLY the direct answer to the question. "
    "Do not include explanations, citations, references, or any additional content. "
    "Give the most concise and direct response possible. "
    "\nProvide the final answer concisely and directly, without code snippets, extra explanations or commentary."
)

INDEX_PREAMBLE_V1 = (
    "The following is the AOCI cognition index of the repository: one line per source file "
    "(name[Tag]: F:function | R:relations | A:interfaces | S:constraints). "
    "Use it as your knowledge of the repository when answering.\n\n"
)

INDEX_PREAMBLE_V2 = (
    "Below is the AOCI cognition index of this repository: one line per file — name[Tag]: F:function | "
    "R:relations | A:interfaces | S:constraints. It is a file-level map provided as additional context alongside "
    "what you already know about this repository. Use it to locate the relevant files and constraints, then answer "
    "the question at the level of detail it asks for (function, method, mechanism), naming the files involved; "
    "do not answer with only a file path.\n\n"
)
PREAMBLES = {"v1": INDEX_PREAMBLE_V1, "v2": INDEX_PREAMBLE_V2}
FIELD_WORDS = {"F": "F:function", "R": "R:relations", "A": "A:interfaces", "S": "S:constraints"}


def index_preamble_v2(fields="TFRAS"):
    """The V2 preamble for a given field set (AOCI ablation on the answering leg, C batch 2026-09-22). 'TFRAS'
    is INDEX_PREAMBLE_V2 byte for byte (the sealed experiment-3 wording). An ablated artefact gets the same
    sentences with only the format clause rewritten, so the model is never told to expect a stripped field;
    names-only also drops "and constraints" from the use clause, since there are none to locate."""
    if fields == "TFRAS":
        return INDEX_PREAMBLE_V2
    if fields == "N":
        fmt = "name (the repository-relative path), grouped by directory"; use = "Use it to locate the relevant files"
    else:
        name = "name[Tag]" if "T" in fields else "name"
        fmt = "%s: %s" % (name, " | ".join(FIELD_WORDS[f] for f in "FRAS" if f in fields))
        use = "Use it to locate the relevant files and constraints" if "S" in fields else "Use it to locate the relevant files"
    return ("Below is the AOCI cognition index of this repository: one line per file — %s. It is a file-level map "
            "provided as additional context alongside what you already know about this repository. %s, then answer "
            "the question at the level of detail it asks for (function, method, mechanism), naming the files involved; "
            "do not answer with only a file path.\n\n" % (fmt, use))

# --preamble guided: closed-book port of the DeepSWE usage protocol (guided.md b15c7d07/a15e8d9e; the
# agent-side port lives in agent_ask.py). No tools exist here, so the read-before-relying clauses become
# index-grounding clauses: every claim anchored in specific entries, silence acknowledged, S/R re-checked
# before finishing. Same head+index+tail structure; no concrete entity tokens in the instructions (G7).
ASK_GUIDED_HEAD = (
    "# REPOSITORY INDEX — build a global understanding first\n\n"
    "Below is a pre-built cognitive index of this repository: module responsibilities, key symbols, their "
    "relationships, and recorded constraints (one line per file — name[Tag]: F:function | R:relations | "
    "A:interfaces | S:constraints). Read it through first to build a global understanding of the repository "
    "and of the areas relevant to the question, then answer. You cannot open files in this setting; the "
    "index is your only source about this repository.\n\n"
)
ASK_GUIDED_TAIL = (
    "\n\n# USAGE PROTOCOL — global understanding first, every claim grounded in the index\n\n"
    "1. Use the index above to build global understanding: the repository's structure, and the "
    "responsibilities and relationships of the modules relevant to the question.\n"
    "2. You cannot read files here: ground every claim in specific index entries (their F/R/A/S content). "
    "Do not assert anything about this repository that no index entry supports.\n"
    "3. Where the index is silent, say so rather than inventing details.\n"
    "4. Before you consider the answer finished, go back to the index entries of every file you rely on: "
    "check your answer against the constraints recorded in their S fields, and check whether the files "
    "named in their R fields change the answer.\n"
    "5. Name the concrete files involved and the specific recorded behaviour; do not answer with only a "
    "file path.\n"
)

VOLUME_FILES = ("aoci.txt", "aoci.meta.txt", "aoci.code.txt")


def load_index(repo_root):
    parts = []
    for name in VOLUME_FILES:
        pth = os.path.join(repo_root, name)
        if os.path.exists(pth):
            parts.append(f"===== {name} =====\n" + open(pth, encoding="utf-8").read())
    if not parts:
        sys.exit(f"no AOCI volumes under {repo_root} (expected {', '.join(VOLUME_FILES)})")
    return "\n".join(parts)


def build_system(condition, index_text, preamble="v2", fields="TFRAS"):
    blocks = [{"type": "text", "text": UPSTREAM_SYSTEM}]
    if condition == "inject":
        if fields != "TFRAS" and preamble != "v2":
            sys.exit("ablated artefacts run with --preamble v2 (the other preambles name all four fields)")
        if preamble == "guided":
            txt = ASK_GUIDED_HEAD + index_text + ASK_GUIDED_TAIL
        elif preamble == "v2":
            txt = index_preamble_v2(fields) + index_text
        else:
            txt = PREAMBLES[preamble] + index_text
        blocks.append({"type": "text", "text": txt,
                       "cache_control": {"type": "ephemeral"}})  # honoured on the Anthropic route only
    return blocks


def user_turn(repo_name, question):
    return f"Repository: {repo_name}\n Question: {question}"


def answer_one(prof, system, repo_name, q, params):
    """One question → record fields. Applies the degenerate-answer guard once."""
    t0 = time.time()
    rec = {"answer": "", "stop_reason": None, "served_model": None, "usage": None, "retried": None, "error": None}
    kw = dict(max_tokens=params["max_tokens"], temperature=params["temperature"],
              effort=params["effort"] if prof.get("protocol") == "anthropic" else None)
    try:
        r = llm.complete(prof, system, user_turn(repo_name, q), **kw)
        rec.update(answer=r["text"], stop_reason=r["stop_reason"], served_model=r["model"], usage=dict(r["usage"]))
        gar = common.refusal_reason(r["text"])
        if gar:  # refusal / special-token garble is not an answer: one bounded re-run in a FRESH conversation
            r = llm.complete(prof, system, user_turn(repo_name, q), **kw)
            for k in rec["usage"]:
                rec["usage"][k] += r["usage"].get(k, 0)
            rec.update(answer=r["text"], stop_reason=r["stop_reason"], retried=gar)
        reason = common.degenerate_reason(r["text"])
        if reason:
            msgs = [{"role": "user", "content": user_turn(repo_name, q)},
                    {"role": "assistant", "content": r["text"] or "(empty)"},
                    {"role": "user", "content": common.NUDGE}]
            r2 = llm.complete(prof, system, msgs, **dict(kw, max_tokens=max(kw["max_tokens"], 8192)))
            for k in rec["usage"]:
                rec["usage"][k] += r2["usage"].get(k, 0)
            rec.update(retried=reason, stop_reason=r2["stop_reason"])
            if r2["text"].strip():
                rec["answer"] = r2["text"]
        if rec["stop_reason"] == "refusal":
            rec["error"] = "refusal"
    except Exception as e:  # noqa: BLE001 — record and continue; the ledger is the diagnosis
        rec["error"] = f"{e.__class__.__name__}: {str(e)[:300]}"
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--condition", choices=["direct", "inject", "aoci"], required=True,
                    help="'aoci' is accepted as an alias of 'inject'")
    ap.add_argument("--repo-root", default=None, help="repository with aoci.txt/aoci.meta.txt/aoci.code.txt (inject)")
    ap.add_argument("--index-artifact", default=None,
                    help="inject: deliver this built artefact (sidecar-verified) instead of the raw volumes; "
                         "stamps index_sha256/index_chars on every record so acceptance can assert delivery")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default=None, help="endpoint/model profile (default: profiles.json default)")
    ap.add_argument("--preamble", default="v2", choices=sorted(list(PREAMBLES) + ["guided"]),
                    help="'guided' = closed-book port of the DeepSWE usage protocol (head+index+tail)")
    ap.add_argument("--fields", default="TFRAS", help="field set the injected artefact carries (AOCI ablation): TFRAS = sealed index, N = names-only; must match the artefact sidecar")
    ap.add_argument("--max-tokens", type=int, default=8192, help="adaptive thinking counts against this on OpenAI-compatible routes")
    ap.add_argument("--temperature", type=float, default=0.0, help="upstream llm_direct default is 0; pass -1 for the endpoint default")
    ap.add_argument("--effort", default=None, help="Anthropic route only (low|medium|high)")
    ap.add_argument("--workers", type=int, default=common.DEFAULT_WORKERS, help=f"parallel requests, 1..{common.MAX_WORKERS}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=None, help="stop launching questions once the running estimate exceeds this")
    ap.add_argument("--allow-param-drift", action="store_true", help="resume onto records produced with different settings")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    condition = "inject" if a.condition == "aoci" else a.condition
    if condition == "inject" and not (a.repo_root or a.index_artifact):
        sys.exit("--repo-root or --index-artifact is required for --condition inject")
    prof = llm.load_profile(a.profile)
    workers = common.clamp_workers(a.workers)
    params = {"max_tokens": a.max_tokens, "temperature": None if a.temperature < 0 else a.temperature,
              "effort": a.effort, "preamble": a.preamble if condition == "inject" else None,
              "fields": a.fields if condition == "inject" else None}

    questions = common.load_questions(a.questions, a.limit)
    index_text, index_fp = "", None
    if condition == "inject":
        if a.index_artifact:
            index_text, index_fp = common.load_artifact(a.index_artifact)
            side = json.load(open(a.index_artifact[:-4] + ".json", encoding="utf-8"))
            if side.get("fields", "TFRAS") != a.fields:            # 侧车说这份制品带哪些字段;--fields 必须与之一致,否则前言会撒谎
                sys.exit("--fields %s does not match the artefact sidecar (fields=%s)" % (a.fields, side.get("fields", "TFRAS")))
            index_fp["fields"] = a.fields
        else:
            if a.fields != "TFRAS": sys.exit("--fields needs --index-artifact (a built, sidecar-stamped artefact)")
            index_text = load_index(a.repo_root)
    system = build_system(condition, index_text, a.preamble, a.fields)
    common.check_param_drift(a.out, {"params": params, "model": prof["model"], "profile": prof["name"], "kind": condition},
                             allow=a.allow_param_drift)
    done = {q for q, r in common.load_records(a.out, common.valid_answer).items() if common.valid_answer(r)}
    todo = [q for q in questions if q["question"] not in done]
    print(f"condition={condition} profile={prof['name']} protocol={prof.get('protocol')} model={prof['model']} "
          f"workers={workers} temperature={params['temperature']} questions={len(questions)} done={len(done)} "
          f"todo={len(todo)} index_bytes={len(index_text.encode())}", flush=True)
    if a.dry_run:
        print("dry-run: system blocks", [(b['type'], len(b['text'])) for b in system], "| first user turn:",
              repr(user_turn(a.repo_name, todo[0]['question'])[:120]) if todo else "<none>")
        return
    if not prof["_has_key"]:
        sys.exit(f"key missing for profile {prof['name']}: set {prof['api_key_env']} in config/secrets.env or export it")
    try:  # one tiny real call before the loop: wrong key / model id / endpoint fail here, not on question 1
        llm.complete(prof, [{"type": "text", "text": "You are a terse assistant."}], "Reply with exactly: OK", max_tokens=8)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"preflight failed for profile {prof['name']}: {e.__class__.__name__}: {str(e)[:300]}")

    run_dir = os.path.dirname(os.path.abspath(a.out))
    arm = os.path.basename(a.out)[len("answers-"):-len(".jsonl")] if os.path.basename(a.out).startswith("answers-") else condition
    common.write_manifest(run_dir, {"arms": {arm: {
        "kind": condition, "profile": prof["name"], "model": prof["model"], "protocol": prof.get("protocol"),
        "params": params, "workers": workers, "questions_sha256": common.sha256_file(a.questions)[:16],
        "system_sha256": common.sha256_text("\n".join(b["text"] for b in system))[:16],
        "index": index_fp if index_fp else (common.index_fingerprint(a.repo_root) if condition == "inject" else None)}}})

    import analyze
    price = analyze.price_for(analyze.load_prices(), prof["model"])
    budget = common.Budget(a.max_usd, (price["in"], price["out"]) if price else None)

    def work(q):
        rec = {"qid": common.qid(q["question"]), "repo": a.repo_name, "condition": arm, "kind": condition,
               "profile": prof["name"], "model": prof["model"], "protocol": prof.get("protocol"),
               "params": params, "question": q["question"], "reference": q.get("answer"),
               "index_sha256": index_fp["artifact_sha256"] if index_fp else None,
               "index_chars": index_fp["chars"] if index_fp else 0,
               "steps": None, "tool_calls": None, "files_read": None, "trace": None}
        if budget.exceeded():
            rec.update(answer="", error=f"budget: est ${budget.spent:.2f} >= cap ${a.max_usd}", usage=None, wall_s=0)
            return rec
        rec.update(answer_one(prof, system, a.repo_name, q["question"], params))
        budget.add(rec.get("usage"))
        return rec

    def on_result(q, rec, i, n):
        if "question" not in rec:  # fn raised: wrap into a record so resume can retry it
            rec = {"qid": common.qid(q["question"]), "question": q["question"], "condition": arm, "answer": "", **rec}
        common.append_jsonl(a.out, rec)
        u = rec.get("usage") or {}
        print(f"[{i}/{n}] {rec.get('wall_s')}s in={u.get('input')} out={u.get('output')} "
              f"{('retried=' + rec['retried']) if rec.get('retried') else ''} {rec.get('error') or ''}", flush=True)

    common.run_parallel(todo, work, workers=workers, on_result=on_result, label="ask", progress_path=common.progress_path_for(a.out))


if __name__ == "__main__":
    main()
