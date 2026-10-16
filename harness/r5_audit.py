#!/usr/bin/env python3
"""r5 closing audit: every check a reader could run against the delivered data, printed PASS/FAIL (2026-10-16).

  python3 harness/r5_audit.py [--runs-glob "runs/sweqa-r5-*-k10"] [--banks benchmarks/swe-qa]

Checks per repo: question bank == 48 distinct official questions; workspace commit == official pin; 3 arms × 10 draws,
last record per question answers every question once; no error records; no empty finals; one answer model; the
cognition turn (rules → overview chunks → attestation) present on every oh-aoci record; acceptance gate clean; every
discarded record re-answered cleanly; judge uniformity (claude-sonnet-4-5, votes 5, one rubric); every judged row has
5 votes; run_dir volumes == manifest volumes; arm means and exact 10v10 p recomputed here == runlevel-stats.json;
superseded passes (answered under the wrong chunk_tokens) present only under _superseded-* and never in the live files.
"""
import argparse
import collections
import glob
import hashlib
import itertools
import json
import os
import re
import statistics as st

ARMS = ("oh", "oh-aoci", "oh-cg")
PAIRS = (("H1", "oh-aoci", "oh"), ("H2", "oh-aoci", "oh-cg"), ("H3", "oh-cg", "oh"))
PINS = {"astropy": "0a041d38fb", "django": "14fc2e9703", "flask": "85c5d93cbd", "matplotlib": "a5e1f6086e", "pylint": "44740e5e5c",
        "pytest": "5989efe3ef", "requests": "46e939b552", "scikit-learn": "adb1ae76d7", "sphinx": "6c9e3209c4", "sqlfluff": "db9801b7eb",
        "sympy": "3c817ed8ab", "xarray": "40119bf35c", "conan": "52f43d98fe", "reflex": "fe0f946dc0", "streamlink": "ab1f365600"}


def jl(p):
    out = []
    for l in open(p, encoding="utf-8"):
        try:
            out.append(json.loads(l))
        except Exception:  # noqa: BLE001
            pass
    return out


def last(rows):
    d = {}
    for r in rows:
        d[r.get("question")] = r
    return d


def total_of(r):
    t = r.get("total")
    if t is None and r.get("scores"):
        t = r["scores"].get("total")
    return None if t is None else float(t)


def exact_p(a, b):
    obs = st.mean(b) - st.mean(a)
    pool = a + b
    tot = sum(pool)
    cnt = n = 0
    for comb in itertools.combinations(range(len(pool)), len(a)):
        s = sum(pool[i] for i in comb)
        d = (tot - s) / len(b) - s / len(a)
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
        n += 1
    return obs, cnt / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/sweqa-r5-*-k10")
    ap.add_argument("--banks", default="benchmarks/swe-qa")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fails = 0

    def check(ok, msg):
        nonlocal fails
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            fails += 1

    for run in sorted(glob.glob(os.path.join(root, a.runs_glob))):
        repo = os.path.basename(run).replace("sweqa-r5-", "").replace("-k10", "")
        print(f"===== {repo} ({os.path.relpath(run, root)})")
        man = json.load(open(os.path.join(run, "manifest.json")))
        bank = jl(os.path.join(root, a.banks, f"{repo}.jsonl"))
        bank_q = {r["question"] for r in bank}
        check(len(bank) == 48 and len(bank_q) == 48, f"question bank: {len(bank)} rows, {len(bank_q)} distinct")
        commit = str(man.get("repo_commit") or "")
        check(commit.startswith(PINS[repo]), f"workspace commit {commit[:12]} == official pin {PINS[repo]}")
        per_arm_draws = collections.Counter()
        errs = invalid = cog_bad = cog_n = 0
        models = set()
        for f in sorted(glob.glob(os.path.join(run, "answers-*.jsonl"))):
            m = re.match(r"^answers-(oh|oh-aoci|oh-cg)-(\d+)\.jsonl$", os.path.basename(f))   # exact arm names (G62)
            if not m:
                continue
            rows = last(jl(f))
            per_arm_draws[m.group(1)] += 1
            if set(rows) != bank_q:
                check(False, f"{os.path.basename(f)}: 48 official questions answered once each (last record): {len(rows)}")
            for r in rows.values():
                models.add(r.get("model"))
                if r.get("error"):
                    errs += 1
                if not (r.get("answer") or "").strip():
                    invalid += 1
                if m.group(1) == "oh-aoci":
                    c = r.get("cognition") or {}
                    cog_n += 1
                    if not (c.get("rules_calls", 0) >= 1 and c.get("overview_calls", 0) >= 1 and c.get("completed_seen") and c.get("attestation_submitted")):
                        cog_bad += 1
        check(dict(per_arm_draws) == {x: 10 for x in ARMS}, f"3 arms × 10 draws: {dict(per_arm_draws)}")
        check(errs == 0, f"error records among last-per-question: {errs}")
        check(invalid == 0, f"empty final answers: {invalid}")
        check(models == {"deepseek-flash"}, f"answer models: {sorted(models)}")
        check(cog_bad == 0, f"cognition turn complete (rules, overview to completed, attestation) on every oh-aoci record: {cog_n - cog_bad}/{cog_n}")
        acc_path = os.path.join(run, "acceptance.json")
        if os.path.exists(acc_path):
            acc = json.load(open(acc_path))
            rows = acc if isinstance(acc, list) else (acc.get("passes") or acc.get("rows") or list(acc.values()))
            hard = [x["pass"] for x in rows if isinstance(x, dict) and x.get("hard_fail")]
            check(not hard, f"acceptance gate (last run): hard failures {hard}")
        else:
            check(False, "acceptance.json present")
        dpath = os.path.join(run, "discards.jsonl")
        for d in (jl(dpath) if os.path.exists(dpath) else []):
            rr = last(jl(os.path.join(run, f"answers-{d['pass']}.jsonl"))).get(d["question"])
            check(bool(rr and not rr.get("error") and (rr.get("answer") or "").strip()), f"discard {d['pass']}#{d['index']} re-answered cleanly")
        sup = glob.glob(os.path.join(run, "_superseded-*", "answers-*.jsonl"))
        if sup:
            print(f"      superseded passes kept aside (not in live files): {len(sup)} files, {sum(len(jl(f)) for f in sup)} records")
        jm, votes_req, rubrics = set(), set(), set()
        full = part = 0
        for f in sorted(glob.glob(os.path.join(run, "scores-*.jsonl"))):
            m = re.match(r"^scores-(oh|oh-aoci|oh-cg)-(\d+)\.jsonl$", os.path.basename(f))
            if not m:
                continue
            for r in last(jl(f)).values():
                jm.add(r.get("judge_model"))
                votes_req.add(r.get("votes_requested"))
                if (r.get("votes") or 0) >= 5 and not r.get("error") and total_of(r) is not None:
                    full += 1
                else:
                    part += 1
        for arm, j in (man.get("judge") or {}).items():
            rubrics.add(j.get("rubric_sha256"))
        check(jm == {"claude-sonnet-4-5"} and votes_req == {5}, f"judge model {sorted(jm, key=str)} votes_requested {sorted(votes_req, key=str)}")
        check(len(rubrics) == 1, f"one rubric across passes: {rubrics}")
        check(part == 0 and full == 1440, f"judged rows with 5 votes: {full}/{full + part} (incomplete {part})")
        idx = man.get("index") or {}
        vols = {}
        for name in ("aoci.txt", "aoci.meta.txt", "aoci.code.txt"):
            p = os.path.join(run, name)
            if os.path.exists(p):
                vols[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        recorded = {k: (v.get("sha256") or "") for k, v in (idx.get("volumes") or {}).items()}
        check(bool(vols) and all(recorded.get(k) == v for k, v in vols.items()), f"run_dir volumes == manifest index volumes {vols}")
        draws = collections.defaultdict(dict)
        for f in sorted(glob.glob(os.path.join(run, "scores-*.jsonl"))):
            m = re.match(r"^scores-(oh|oh-aoci|oh-cg)-(\d+)\.jsonl$", os.path.basename(f))
            if not m:
                continue
            vals = [total_of(r) for r in last(jl(f)).values() if total_of(r) is not None and (r.get("votes") or 0) >= 5]
            if vals:
                draws[m.group(1)][int(m.group(2))] = st.mean(vals)
        dm = {arm: [v for _, v in sorted(d.items())] for arm, d in draws.items()}
        rl_path = os.path.join(run, "runlevel-stats.json")
        rl = json.load(open(rl_path)) if os.path.exists(rl_path) else {}
        for arm in ARMS:
            if len(dm.get(arm, [])) != 10:
                check(False, f"{arm}: 10 judged draws (have {len(dm.get(arm, []))})")
                continue
            mine = st.mean(dm[arm])
            theirs = (rl.get("arms") or {}).get(arm, {}).get("mean")
            check(theirs is not None and abs(mine - theirs) < 0.02, f"{arm} mean {mine:.2f} == runlevel_stats {theirs}")
        if all(len(dm.get(arm, [])) == 10 for arm in ARMS):
            for label, b, bb in PAIRS:
                obs, p = exact_p(dm[bb], dm[b])
                print(f"      {label} {b} − {bb} = {obs:+.2f}  exact p={p:.4f}")
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAIL(S)'}")


if __name__ == "__main__":
    main()
