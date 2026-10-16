#!/usr/bin/env python3
"""Score answers with SWE-QA's LLM-as-a-Judge rubric.

The rubric text is the upstream `Benchmark construction/score/llm-as-a-judge.py`
prompt verbatim (five dimensions, 1-20 each, total 100). Upstream scores each
answer five times and takes the majority; --votes controls that here (use 1
for pilots, 5 for the reported run). The judge model is a parameter — upstream
used Claude Sonnet 4.5; whatever is used must be disclosed alongside results,
and it should not be the model that produced the answers where avoidable.

Prompt shape follows upstream: the rubric is the single *user* message (no
system message); upstream also asked for a 1024-token thinking budget, which
OpenAI-compatible gateways ignore — leave --max-tokens room for it instead.
Runs --workers answers at once (default 7, 1..20; upstream used 16-48).

Usage:
  judge.py --answers runs/x/answers-inject.jsonl --out runs/x/scores-inject.jsonl [--votes 5] [--profile opus5]
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import llm  # noqa: E402

DIMS = ("correctness", "completeness", "relevance", "clarity", "reasoning")

RUBRIC = """You are a STRICT and RIGOROUS evaluator. You must rate the candidate answer STRICTLY against the reference answer. Be CONSERVATIVE with high scores - only award high scores (16-20) when the candidate answer is truly excellent and closely matches the reference answer in quality and content.

CRITICAL EVALUATION PRINCIPLES:
1. Compare the candidate answer DIRECTLY with the reference answer point by point
2. Any deviation, omission, or inaccuracy should result in score reduction
3. High scores (16-20) should be RARE - reserve them only for answers that are nearly perfect
4. Be strict about factual accuracy - even minor errors should lower the correctness score
5. Missing key points from the reference answer should significantly reduce completeness score
6. Vague or imprecise language should lower clarity scores
7. When in doubt between two score ranges, choose the LOWER one

Evaluation Criteria and Scoring Guidelines (each scored 1 to 20, total score 100):
        1. Correctness (STRICT - penalize any inaccuracies):
            20 — ONLY if completely correct with ALL core points and details accurate, matching reference answer precisely
            16-19 — Mostly correct but must have only TRIVIAL inaccuracies; any noticeable error reduces to 15 or below
            12-15 — Partially correct; has some errors or omissions that affect understanding; main points may be accurate but details are wrong
            8-11 — Several errors or ambiguities that significantly affect understanding of core information
            4-7 — Many errors; misleading or fails to convey key information correctly
            1-3 — Serious errors; completely wrong or misleading
        2. Completeness (STRICT - penalize missing information):
            20 — ONLY if covers ALL key points from reference answer without ANY omission; must match reference in depth
            16-19 — Covers most key points but missing some non-trivial information; minor omissions are acceptable
            12-15 — Missing several important key points; content is noticeably incomplete compared to reference
            8-11 — Important information largely missing; content is one-sided or superficial
            4-7 — Covers very little relevant information; seriously incomplete
            1-3 — Covers almost no relevant information; completely incomplete
        3. Relevance (STRICT - penalize off-topic content):
            20 — ONLY if content is fully focused on question topic with NO irrelevant information whatsoever
            16-19 — Mostly focused but may have minor peripheral information; any significant off-topic content reduces score
            12-15 — Generally on topic but contains some off-topic content that detracts from answer
            8-11 — Topic not sufficiently focused; contains considerable off-topic or tangential content
            4-7 — Content deviates from topic; includes excessive irrelevant information
            1-3 — Majority of content irrelevant to the question
        4. Clarity (STRICT - penalize unclear expression):
            20 — ONLY if language is exceptionally fluent, clear, and precise; very easy to understand without any ambiguity
            16-19 — Mostly fluent and clear but may have minor unclear points; any significant ambiguity reduces score
            12-15 — Generally clear but some expressions are unclear or not concise; may require effort to understand
            8-11 — Expression somewhat awkward; has ambiguity or lacks fluency that hinders understanding
            4-7 — Language obscure; sentences are not smooth; significantly hinders understanding
            1-3 — Expression confusing; very difficult to understand
        5. Reasoning (STRICT - penalize weak logic):
            20 — ONLY if reasoning is exceptionally clear, logical, and well-structured; argumentation is excellent and matches reference quality
            16-19 — Reasoning is clear and logical with solid argumentation; minor logical gaps may exist
            12-15 — Reasoning generally reasonable but has noticeable logical jumps or organization issues
            8-11 — Reasoning is average; has logical jumps or organization problems that affect understanding
            4-7 — Reasoning unclear; lacks logical order; difficult to follow
            1-3 — No clear reasoning; logic is chaotic

INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,20], corresponding
    to the evaluation scores:
        {{
        "correctness": <1-20>,
        "completeness": <1-20>,
        "relevance": <1-20>,
        "clarity": <1-20>,
        "reasoning": <1-20>
        }}

SCORING INSTRUCTIONS:
- Read the reference answer carefully and identify ALL key points, details, and structure
- Compare the candidate answer systematically against the reference answer
- For each criterion, start with a conservative score and only increase if the candidate truly deserves it
- If the candidate answer is significantly shorter, less detailed, or less precise than the reference, reduce scores accordingly
- If the candidate answer contains information not in the reference (unless it's clearly relevant and accurate), consider reducing relevance score
- When scoring, ask yourself: "Does this candidate answer match the quality and completeness of the reference answer?" If not, reduce scores
- Average or mediocre answers should receive scores in the 8-15 range, not higher
- Only truly excellent answers that closely match the reference should receive 16-20 scores

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON. Be strict and conservative with your scores."""


def parse_scores(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    def _valid(block):
        try:
            d = json.loads(block)
            s = {k: int(d[k]) for k in DIMS}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return s if all(1 <= v <= 20 for v in s.values()) else None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = _valid(m.group(0))
    if s:
        return s
    # 2026-10-16 (r5): for one sphinx question the judge keeps writing a malformed object (duplicated "relevance", no
    # "clarity") and then, on some calls, a corrected object after "Wait, I need to output valid JSON"; the greedy match
    # spans both and fails. Accept the LAST well-formed five-field object in the text; nothing else changes.
    for block in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        s = _valid(block)
        if s:
            return s
    return None


def aggregate(votes):
    out = {}
    for k in DIMS:
        vals = [v[k] for v in votes]
        c = Counter(vals).most_common()
        # majority when one exists, else median (upstream: majority voting)
        out[k] = c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else int(statistics.median(vals))
    out["total"] = sum(out[k] for k in DIMS)
    return out


def supports_effort(model):
    """Anthropic's output_config.effort exists from the 4.6 / 5 generation on; Sonnet 4.5 (the experiment-3 judge, now
    reached through CloseAI) rejects it with 400 (C batch smoke, 2026-09-22)."""
    import re
    return not re.search(r"(?:^|-)(?:3|4)-[0-5](?:-|$)|4\.[0-5]", model)


QUOTA_ERRORS = []


def judge_one(prof, r, votes, max_tokens):
    prompt = RUBRIC.format(question=r["question"], reference=r["reference"], candidate=r["answer"])
    got, raw, usage = [], [], {"input": 0, "output": 0}
    # 2026-10-14 (r4): the judge occasionally emits a malformed score object for one specific answer (a duplicated
    # key and a missing one) on most calls; the old "votes × 2 attempts" budget then left 1-2 valid votes forever.
    # Keep asking until `votes` VALID votes exist, bounded at 3× that many calls; aggregation is unchanged.
    calls = 0
    max_calls = votes * int(os.environ.get("JUDGE_MAX_CALLS_FACTOR", "3"))   # r5: raise for the one answer the judge keeps garbling
    while len(got) < votes and calls < max_calls:
        for attempt in range(2):  # one schema retry, as upstream tolerates fenced JSON
            calls += 1
            try:
                res = llm.complete(prof, [], prompt, max_tokens=max_tokens,
                                   effort="low" if prof.get("protocol") == "anthropic" and supports_effort(prof["model"]) else None,
                                   thinking={"type": "disabled"} if prof.get("protocol") == "anthropic" else None)
            except Exception as e:  # noqa: BLE001
                raw.append(f"ERROR {e.__class__.__name__}: {str(e)[:200]}")
                if "额度" in str(e) or "PermissionDenied" in e.__class__.__name__ or "Error code: 403" in str(e):
                    QUOTA_ERRORS.append(str(e)[:120])      # fail fast at the end of the pass instead of burning every vote
                time.sleep(5)
                continue
            usage["input"] += res["usage"]["input"]
            usage["output"] += res["usage"]["output"]
            raw.append(res["text"])
            sc = parse_scores(res["text"])
            if sc:
                got.append(sc)
                break
            if calls >= max_calls:
                break
    return {"votes_requested": votes, "votes": len(got), "raw": raw, "usage": usage,
            "scores": aggregate(got) if got else None, "error": None if got else "no valid vote"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default=None, help="judge endpoint/model profile (default: profiles.json default)")
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048, help="must leave room for adaptive thinking on OpenAI-compatible routes")
    ap.add_argument("--workers", type=int, default=common.DEFAULT_WORKERS, help=f"parallel answers, 1..{common.MAX_WORKERS}")
    ap.add_argument("--allow-self-judge", action="store_true", help="permit judge model == answering model (pilots only)")
    ap.add_argument("--rubric-file", default=None, help="override the built-in rubric (judge-robustness re-judging); must keep {question}/{reference}/{candidate} slots")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.rubric_file:
        global RUBRIC
        txt = open(a.rubric_file, encoding="utf-8").read()
        for slot in ("{question}", "{reference}", "{candidate}"):
            if slot not in txt:
                sys.exit(f"rubric file missing slot {slot}")
        RUBRIC = txt
        import hashlib
        print(f"rubric override: {a.rubric_file} sha256={hashlib.sha256(txt.encode()).hexdigest()[:16]}", flush=True)
    prof = llm.load_profile(a.profile)
    workers = common.clamp_workers(a.workers)

    answers = common.load_records(a.answers, common.valid_answer)
    rows = [r for r in answers.values() if common.valid_answer(r)]
    # 2026-10-14 (r4): a row counts as scored only when it carries every requested vote and no error — the gptnb
    # quota outage left rows with 0-4 votes that the old "any scores" test silently kept; resume now re-judges them
    # (the new row is appended; consumers take the last record per question).
    scored = {q for q, r in common.load_records(a.out, common.valid_score).items()
              if common.valid_score(r) and not r.get("error") and ((r.get("votes") or 0) >= a.votes or r.get("empty_answer"))}
    todo = [r for r in rows if r["question"] not in scored]
    answer_models = sorted({r.get("model") for r in rows if r.get("model")})
    print(f"judge profile={prof['name']} model={prof['model']} votes={a.votes} workers={workers} "
          f"answers={len(answers)} valid={len(rows)} scored={len(scored)} todo={len(todo)} answer_models={answer_models}", flush=True)
    if any(m and m.split('/')[-1] == prof['model'].split('/')[-1] for m in answer_models):
        msg = "judge model equals the answering model (self-preference risk)"
        if not a.allow_self_judge:
            sys.exit(f"refusing: {msg}; pass --allow-self-judge for a pilot")
        print("WARNING:", msg, flush=True)
    if a.dry_run:
        if todo:
            p = RUBRIC.format(question=todo[0]["question"], reference=todo[0]["reference"], candidate=todo[0]["answer"])
            print(f"dry-run: first prompt {len(p)} chars; calls that would be made: {len(todo) * a.votes}")
        return
    if not prof["_has_key"]:
        raise SystemExit(f"key missing for profile {prof['name']}: set {prof['api_key_env']} in config/secrets.env")

    run_dir = os.path.dirname(os.path.abspath(a.out))
    base = os.path.basename(a.out)
    arm = base[len("scores-"):-len(".jsonl")] if base.startswith("scores-") else "?"
    common.write_manifest(run_dir, {"judge": {arm: {"profile": prof["name"], "model": prof["model"], "votes": a.votes,
                                                     "max_tokens": a.max_tokens, "workers": workers,
                                                     "rubric_sha256": common.sha256_text(RUBRIC)[:16],
                                                     "answer_models": answer_models}}})

    # An EMPTY final answer is a scored zero, not a skipped record: dropping it would silently
    # delete the arm's worst outcomes and inflate its mean (2026-08-28: oh/oh-cg lost 20 empty
    # finals to the valid_answer filter and the leaderboard's complete-run count exposed it).
    empties = [r for r in answers.values()
               if not common.valid_answer(r) and not r.get("error") and not (r.get("answer") or "").strip()
               and r["question"] not in scored]
    for r in empties:
        common.append_jsonl(a.out, {"qid": common.qid(r["question"]), "question": r["question"],
                                    "condition": r.get("condition"), "answer_model": r.get("model"),
                                    "judge_profile": prof["name"], "judge_model": prof["model"],
                                    "scores": {**{k: 0 for k in DIMS}, "total": 0}, "votes": 0,
                                    "empty_answer": True, "raw": None, "usage": None, "error": None})
    if empties:
        print(f"empty finals scored 0 by rule: {len(empties)}", flush=True)

    def work(r):
        rec = {"qid": common.qid(r["question"]), "question": r["question"], "condition": r.get("condition"),
               "answer_model": r.get("model"), "judge_profile": prof["name"], "judge_model": prof["model"]}
        rec.update(judge_one(prof, r, a.votes, a.max_tokens))
        return rec

    def on_result(r, rec, i, n):
        if "question" not in rec:
            rec = {"qid": common.qid(r["question"]), "question": r["question"], "condition": r.get("condition"), "scores": None, **rec}
        common.append_jsonl(a.out, rec)
        print(f"[{i}/{n}] total={rec['scores']['total'] if rec.get('scores') else 'FAIL'} votes={rec.get('votes')} {rec.get('error') or ''}", flush=True)
        if len(QUOTA_ERRORS) >= 5:
            raise SystemExit(f"judge route refused {len(QUOTA_ERRORS)} calls (quota/403): {QUOTA_ERRORS[0]} — stopping this pass; resume after the route is restored")

    common.run_parallel(todo, work, workers=workers, on_result=on_result, label="judge", progress_path=common.progress_path_for(a.out, stage='judge'))


if __name__ == "__main__":
    main()
