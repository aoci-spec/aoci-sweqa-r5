#!/usr/bin/env python3
"""r5 flat tables from the judged runs (last record per question, rows with 5 valid votes only).

  python3 harness/r5_flat.py [--runs-glob "runs/sweqa-r5-*-k10"] --out-dir runs/r5-tables
Writes SWEQA-R5-scores-flat.csv (repo, arm, run, qid, category, total, five dimensions, votes, answer_chars, files_read,
zero_source, forced, cognition_attested) and SWEQA-R5-arm-run-means.csv (repo, arm, run, n_q, mean_total) — the direct
input of the run-level permutation tests — plus SWEQA-R5-parity-flat.csv for the bare single-run parity rows.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

ARMS = ("oh", "oh-aoci", "oh-cg")
DIMS = ("correctness", "completeness", "relevance", "clarity", "reasoning")
CATS = {"what", "why", "where", "how"}


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


def category(q):
    w = (q or "").strip().split()[:1]
    w = w[0].lower().strip("?,.:") if w else ""
    return w if w in CATS else "other"


def total_of(r):
    t = r.get("total")
    if t is None and r.get("scores"):
        t = r["scores"].get("total")
    return None if t is None else float(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/sweqa-r5-*-k10")
    ap.add_argument("--out-dir", default="runs/r5-tables")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, a.out_dir), exist_ok=True)
    flat, means = [], []
    for run in sorted(glob.glob(os.path.join(root, a.runs_glob))):
        repo = os.path.basename(run).replace("sweqa-r5-", "").replace("-k10", "")
        for sf in sorted(glob.glob(os.path.join(run, "scores-*.jsonl"))):
            m = re.match(r"^scores-(oh|oh-aoci|oh-cg)-(\d+)\.jsonl$", os.path.basename(sf))
            if not m:
                continue
            arm, draw = m.group(1), int(m.group(2))
            ans = last(jl(os.path.join(run, f"answers-{arm}-{draw}.jsonl")))
            rows = []
            for r in last(jl(sf)).values():
                if r.get("error") or (r.get("votes") or 0) < 5 or total_of(r) is None:
                    continue
                ar = ans.get(r.get("question")) or {}
                cog = ar.get("cognition") or {}
                row = {"repo": repo, "arm": arm, "run": draw, "qid": r.get("qid"), "category": category(r.get("question")),
                       "total": total_of(r), **{d: (r.get("scores") or {}).get(d) for d in DIMS}, "votes": r.get("votes"),
                       "answer_chars": len(ar.get("answer") or ""), "files_read": len(ar.get("files_read") or []),
                       "zero_source": int(not (ar.get("files_read") or [])), "forced": int(bool(ar.get("forced") or ar.get("hit_cap"))),
                       "cognition_attested": ("" if arm != "oh-aoci" else int(bool(cog.get("attestation_submitted"))))}
                rows.append(row)
            flat.extend(rows)
            if rows:
                means.append({"repo": repo, "arm": arm, "run": draw, "n_q": len(rows), "mean_total": round(st.mean(r["total"] for r in rows), 4)})
    with open(os.path.join(root, a.out_dir, "SWEQA-R5-scores-flat.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    with open(os.path.join(root, a.out_dir, "SWEQA-R5-arm-run-means.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "arm", "run", "n_q", "mean_total"])
        w.writeheader(); w.writerows(means)
    par = []
    for run in sorted(glob.glob(os.path.join(root, "runs", "sweqa-r5p-*-k1"))):
        m = re.match(r"^sweqa-r5p-(gpt51|glm46)-(.+)-k1$", os.path.basename(run))
        if not m:
            continue
        sf = next((x for x in (os.path.join(run, "scores-oh.jsonl"), os.path.join(run, "scores-oh-1.jsonl")) if os.path.exists(x)), None)
        if not sf:
            continue
        for r in last(jl(sf)).values():
            empty_by_rule = r.get("votes_requested") is None and total_of(r) == 0 and not r.get("error")
            if (r.get("error") or (r.get("votes") or 0) < 5 or total_of(r) is None) and not empty_by_rule:
                continue
            par.append({"model": m.group(1), "repo": m.group(2), "qid": r.get("qid"), "category": category(r.get("question")),
                        "total": total_of(r), **{d: (r.get("scores") or {}).get(d) for d in DIMS}, "votes": r.get("votes"),
                        "empty_answer_scored_zero": int(empty_by_rule)})
    if par:
        with open(os.path.join(root, a.out_dir, "SWEQA-R5-parity-flat.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(par[0].keys()))
            w.writeheader(); w.writerows(par)
    print(f"flat rows {len(flat)} | arm-run means {len(means)} | parity rows {len(par)} → {a.out_dir}")


if __name__ == "__main__":
    main()
