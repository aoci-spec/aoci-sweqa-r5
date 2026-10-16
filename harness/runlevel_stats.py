#!/usr/bin/env python3
"""Run-level exact permutation over per-draw arm means, for any SWE-QA-style run.

One observation = one draw's arm-pass mean over its judged questions (k draws per arm).
A k vs k contrast is tested by the exact two-sided permutation over C(2k, k) splits
(k=10 -> 184,756). Category rows use the bank's explicit category axis when present
(common.question_categories), falling back to nothing rather than a wrong heuristic.

Usage:
  runlevel_stats.py runs/<run> \
      --pairs "agent-inject:agent=H1,oh-inject:oh=H2" \
      [--cat-pairs "agent-inject:agent"] [--json out.json]

Pair syntax B:A=label means the reported delta is mean(B) - mean(A).
"""
import argparse
import itertools
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def exact_perm(a, b):
    """Two-sided exact permutation p for mean(b)-mean(a) over all C(len(a)+len(b), len(a)) splits."""
    obs = st.mean(b) - st.mean(a)
    pool = list(a) + list(b)
    total = sum(pool)
    cnt = tot = 0
    for comb in itertools.combinations(range(len(pool)), len(a)):
        s = sum(pool[i] for i in comb)
        d = (total - s) / len(b) - s / len(a)
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
        tot += 1
    return obs, cnt / tot


def load_draws(run_dir, qcat=None, cat=None, dim="total"):
    """{base: [per-draw means in draw order]}; optionally one bank category and/or one judge dimension."""
    acc = {}
    for f in sorted(os.listdir(run_dir)):
        m = re.match(r"scores-(.+)-(\d+)\.jsonl$", f)
        if not m:
            continue
        base, d = m.group(1), int(m.group(2))
        tots = []
        # 2026-10-14 (r4): judge resumes APPEND a re-scored row, so a file can hold two rows for one question;
        # take the last record per question (same rule as the gate, the judge and r4_report) — never both.
        for r in common.load_records(os.path.join(run_dir, f), common.valid_score).values():
            if not r.get("scores"):
                continue
            if cat is not None and (qcat or {}).get(r.get("question")) != cat:
                continue
            tots.append(r["scores"][dim])
        if tots:
            acc.setdefault(base, {})[d] = st.mean(tots)
    return {b: [v[d] for d in sorted(v)] for b, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pairs", required=True, help="comma list of B:A=label (delta = mean(B)-mean(A))")
    ap.add_argument("--cat-pairs", default="", help="comma list of B:A pairs to also break down per bank category")
    ap.add_argument("--dim", default="total", help="judge score field to analyse (e.g. correctness for the P0 reweight check)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run_dir = a.run.rstrip("/")
    man = common.read_manifest(run_dir) or {}
    qpath = ((man.get("spec") or {}).get("questions")) or ""
    qmap, axis = common.question_categories(qpath)

    draws = load_draws(run_dir, dim=a.dim)
    print(f"=== 臂均值(k 轮 run 级均值;括号=轮间 sd) · {run_dir} ===")
    for b in sorted(draws):
        v = draws[b]
        print(f"  {b:20} {st.mean(v):6.2f} ({st.pstdev(v):.2f})  k={len(v)}")

    out = {"run": run_dir, "arms": {b: {"mean": st.mean(v), "sd": st.pstdev(v), "k": len(v), "draws": v}
                                   for b, v in draws.items()}, "pairs": [], "cat_pairs": {}}
    print("\n=== run 级精确置换(两侧;k=10v10 -> C(20,10)=184,756) ===")
    for spec in [x for x in a.pairs.split(",") if ":" in x]:
        ba, _, label = spec.partition("=")
        B, _, A = ba.partition(":")
        if B not in draws or A not in draws:
            print(f"  {label or ba}: SKIP (missing arm {B if B not in draws else A})")
            continue
        obs, p = exact_perm(draws[A], draws[B])
        out["pairs"].append({"label": label or ba, "b": B, "a": A, "delta": obs, "p": p})
        print(f"  {label or ba:24} {B} − {A} = {obs:+6.2f}   p={p:.4f}")

    for ba in [x for x in a.cat_pairs.split(",") if ":" in x]:
        B, _, A = ba.partition(":")
        print(f"\n=== 类别分解(探索性;run 级) · {B} − {A} ===")
        rows = []
        for c in axis:
            da = load_draws(run_dir, qmap, c, dim=a.dim)
            if B in da and A in da and len(da[B]) == len(da[A]):
                obs, p = exact_perm(da[A], da[B])
                rows.append({"cat": c, "delta": obs, "p": p})
                print(f"  {c:8} Δ={obs:+6.2f}  p={p:.4f}")
        out["cat_pairs"][ba] = rows

    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
