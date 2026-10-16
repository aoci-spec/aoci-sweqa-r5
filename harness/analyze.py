#!/usr/bin/env python3
"""Analyse a run directory: per-arm scores, effort, cost, locate metrics, and
paired comparisons with bootstrap CIs.

Arms are discovered from runs/<id>/answers-<arm>.jsonl (+ scores-<arm>.jsonl).
The manifest (manifest.json) is echoed so the numbers travel with what
produced them.

Per arm: n, errors, retried; judge total (mean/median/sd), what/how/why/where
means (SWE-QA's category axis = the question's first word), judge dimensions;
steps/tool calls/files read (agent arms); tokens, est. $ per question, wall;
locate metrics against the file paths named in the reference answer:
  recall_read  share of reference paths the agent opened with its own tools (lower bound)
  recall_map   share of reference paths the MAP named in its output (map arms only; a mention, not a
               read — never add it to recall_read)
  recall_ans   share of reference paths named in the answer (all arms)

Pairs (--pairs b:a means "b minus a"; default: every arm minus --baseline, plus
agent-inject:agent when both exist): mean Δ with a bootstrap 95% CI,
win/tie/loss, per-category Δ, first-tool distribution and step deltas for
agent pairs, and the largest swings with what each arm read.

Usage: analyze.py runs/pilot-2 [--pairs agent-aoci:agent,aoci-v2:direct] [--baseline direct] [--json out.json]
"""
import argparse
import glob
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DIMS = ("correctness", "completeness", "relevance", "clarity", "reasoning")
CATS = ("what", "how", "why", "where")
PATH_RE = re.compile(r"\b((?:[\w.-]+/)+[\w.-]+\.(?:py|pyi|pyx|ts|tsx|js|jsx|go|rs|java|kt|c|cc|cpp|h|hpp|cs|rb|php|rst|md|yml|yaml|toml|json))\b")


def load_prices():
    p = os.path.join(common.ROOT, "config", "prices.json")
    try:
        d = json.load(open(p, encoding="utf-8"))
        return {k: v for k, v in d.items() if not k.startswith("_")}
    except (OSError, json.JSONDecodeError):
        return {}


def price_for(prices, model):
    if not model:
        return None
    return prices.get(model) or prices.get(model.split("/")[-1])


def cost_usd(prices, model, u):
    p = price_for(prices, model)
    if not p or not u:
        return None
    return ((u.get("input") or 0) * p["in"] + (u.get("output") or 0) * p["out"]
            + (u.get("cache_read") or 0) * p.get("cache_read", 0) + (u.get("cache_write") or 0) * p.get("cache_write", 0)) / 1e6


def ref_paths(text):
    return set(PATH_RE.findall(text or ""))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def fmt(x, w=6, d=1):
    return f"{x:{w}.{d}f}" if x == x else " " * (w - 3) + "n/a"


def boot_ci(d, iters=10000, seed=0):
    if not d:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    n = len(d)
    ms = sorted(sum(rnd.choices(d, k=n)) / n for _ in range(iters))
    return (ms[int(0.025 * iters)], ms[int(0.975 * iters) - 1])


_QMAP, _AXIS = {}, ()   # explicit bank categories (set in main from the manifest's questions file)


def set_category_axis(questions_path):
    global _QMAP, _AXIS
    _QMAP, _AXIS = common.question_categories(questions_path or "")


def category(q):
    c = _QMAP.get(q)
    if c:
        return c
    w = q.strip().split()[0].lower().strip("?,.:") if q.strip() else ""
    return w if w in CATS else "other"


def load_run(run):
    arms = {}
    for p in sorted(glob.glob(os.path.join(run, "answers-*.jsonl"))):
        arm = os.path.basename(p)[len("answers-"):-len(".jsonl")]
        ans = common.load_records(p, common.valid_answer)
        sp = os.path.join(run, f"scores-{arm}.jsonl")
        sc = common.load_records(sp, common.valid_score) if os.path.exists(sp) else {}
        arms[arm] = {"answers": ans, "scores": {q: r for q, r in sc.items() if common.valid_score(r)}}
    return arms


def vote_spread(rec):
    """SD of the per-vote totals for one judged answer (None if <2 parsable votes)."""
    tots = []
    for raw in rec.get("raw") or []:
        try:
            import judge as _j
            sc = _j.parse_scores(raw)
        except Exception:  # noqa: BLE001
            sc = None
        if sc:
            tots.append(sum(sc[d] for d in DIMS))
    return st.pstdev(tots) if len(tots) >= 2 else None


def arm_summary(arm, data, prices):
    ans, sc = data["answers"], data["scores"]
    rows = list(ans.values())
    valid = [r for r in rows if common.valid_answer(r)]
    totals = [sc[q]["scores"]["total"] for q in sc]
    spreads = [vote_spread(sc[q]) for q in sc]
    bycat = defaultdict(list)
    for q in sc:
        bycat[category(q)].append(sc[q]["scores"]["total"])
    dims = {d: mean([sc[q]["scores"][d] for q in sc]) for d in DIMS}
    rr, rs, ra = [], [], []
    for r in valid:
        refs = ref_paths(r.get("reference"))
        if not refs:
            continue
        if r.get("files_read") is not None:
            rr.append(len(refs & set(r["files_read"])) / len(refs))          # opened with its own tools
        named = r.get("files_map_named")
        if named is not None:
            rs.append(len(refs & set(named)) / len(refs))                    # named by the map (not read)
        ra.append(len(refs & ref_paths(r.get("answer"))) / len(refs))
    peaks = [max((x.get("input") or 0) for x in (r.get("rounds") or [{}])) if r.get("rounds") else None for r in valid]
    us = [r.get("usage") or {} for r in valid]
    model = next((r.get("model") for r in rows if r.get("model")), None)
    costs = [cost_usd(prices, model, u) for u in us]
    agent = any(r.get("steps") is not None for r in valid)
    tot = {k: sum((u.get(k) or 0) for u in us) for k in ("input", "output", "cache_read", "cache_write")}
    return {
        "arm": arm, "model": model, "kind": next((r.get("kind") for r in rows if r.get("kind")), None),
        "n_answers": len(rows), "n_valid": len(valid), "errors": sum(1 for r in rows if r.get("error")),
        "retried": sum(1 for r in valid if r.get("retried")), "min_steps_nudged": sum(1 for r in valid if r.get("min_steps_nudged")),
        "n_scored": len(totals), "judge_vote_sd": mean([x for x in spreads if x is not None]) if any(x is not None for x in spreads) else None,
        "score_mean": mean(totals), "score_median": st.median(totals) if totals else float("nan"),
        "score_sd": st.pstdev(totals) if totals else float("nan"),
        "by_category": {c: mean(bycat.get(c, [])) for c in (_AXIS or CATS)}, "dims": dims,
        "steps": mean([r.get("steps") for r in valid]) if agent else None,
        "tool_calls": mean([r.get("tool_calls") for r in valid]) if agent else None,
        "files_read": mean([len(r.get("files_read") or []) for r in valid]) if agent else None,
        "ans_chars": mean([len(r.get("answer") or "") for r in valid]),
        "map_calls": mean([r.get("map_calls") for r in valid]) if any(r.get("map_calls") is not None for r in valid) else None,
        "map_chars": mean([r.get("map_chars") for r in valid]) if any(r.get("map_chars") is not None for r in valid) else None,
        "map_used_share": (sum(1 for r in valid if (r.get("map_calls") or 0) > 0) / len(valid)) if valid and any(r.get("map_calls") is not None for r in valid) else None,
        "files_via_map": mean([len(r.get("files_map_named") or []) for r in valid]) if any(r.get("files_map_named") is not None for r in valid) else None,
        "peak_in_tok": mean([p for p in peaks if p is not None]) if any(p is not None for p in peaks) else None,
        "in_tok": mean([u.get("input") for u in us]), "out_tok": mean([u.get("output") for u in us]),
        "cache_read_tok": mean([u.get("cache_read") for u in us]),
        "tokens_total": tot, "cost_total": (sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None),
        "cost_per_q": mean(costs) if any(c is not None for c in costs) else None,
        "wall_s": mean([r.get("wall_s") for r in valid]),
        "recall_read": mean(rr) if rr else None, "recall_seen": mean(rs) if rs else None,
        "recall_ans": mean(ra) if ra else None,
    }


def first_tool(r):
    t = r.get("trace") or []
    return t[0]["tool"] if t else "(none)"


def pair_summary(b, a, B, A):
    """B minus A over common scored questions."""
    qs = sorted(set(B["scores"]) & set(A["scores"]))
    d = [B["scores"][q]["scores"]["total"] - A["scores"][q]["scores"]["total"] for q in qs]
    lo, hi = boot_ci(d)
    w, l = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
    import math
    nn = w + l
    sign_p = min(1.0, 2 * sum(math.comb(nn, k) for k in range(0, min(w, l) + 1)) / 2 ** nn) if nn else float("nan")
    sd_d = st.pstdev(d) if len(d) > 1 else float("nan")
    md = mean(d)
    need_n = ((1.96 + 0.84) * sd_d / abs(md)) ** 2 if d and md and sd_d == sd_d else float("nan")
    out = {"pair": f"{b} - {a}", "n": len(qs), "mean_delta": md, "ci95": [lo, hi],
           "median_delta": st.median(d) if d else float("nan"), "sd_delta": sd_d,
           "win": w, "tie": len(d) - nn, "loss": l, "sign_test_p": sign_p, "n_for_80pct_power": need_n,
           "by_category": {}, "largest": []}
    bycat = defaultdict(list)
    for q, x in zip(qs, d):
        bycat[category(q)].append(x)
    for c in (_AXIS or CATS):
        xs = bycat.get(c, [])
        clo, chi = boot_ci(xs, iters=4000)
        out["by_category"][c] = {"n": len(xs), "delta": mean(xs), "ci95": [clo, chi]}
    agentB = any(r.get("steps") is not None for r in B["answers"].values())
    agentA = any(r.get("steps") is not None for r in A["answers"].values())
    if agentA and agentB:
        out["first_tool"] = {a: dict(Counter(first_tool(A["answers"][q]) for q in qs if q in A["answers"])),
                             b: dict(Counter(first_tool(B["answers"][q]) for q in qs if q in B["answers"]))}
        sb = [(B["answers"][q].get("steps") or 0) - (A["answers"][q].get("steps") or 0) for q in qs if q in A["answers"] and q in B["answers"]]
        out["steps"] = {"fewer": sum(1 for x in sb if x < 0), "same": sum(1 for x in sb if x == 0), "more": sum(1 for x in sb if x > 0)}
        def cap_of(arm_data):
            r0 = next(iter(arm_data["answers"].values()), {})
            p = r0.get("params") or {}
            return p.get("max_steps") or p.get("max_iterations")

        def hit(arm_data, q):
            """Uniform rule for both hosts: a forced round, or the step budget reached."""
            r = arm_data["answers"].get(q) or {}
            cap = cap_of(arm_data)
            return bool(r.get("forced")) or (bool(cap) and (r.get("steps") or 0) >= cap)
        out["hit_cap"] = {a: sum(1 for q in qs if hit(A, q)), b: sum(1 for q in qs if hit(B, q))}
    # paired efficiency deltas — the axis a map is actually sold on; same questions, same pairing
    def paired(get):
        xs = [(get(B["answers"].get(q) or {}), get(A["answers"].get(q) or {})) for q in qs]
        xs = [(b, a_) for b, a_ in xs if b is not None and a_ is not None]
        if not xs:
            return None
        dd = [b - a_ for b, a_ in xs]
        lo, hi = boot_ci(dd, iters=4000)
        base = mean([a_ for _, a_ in xs])
        return {"n": len(dd), "delta": mean(dd), "ci95": [lo, hi], "median": st.median(dd),
                "base": base, "ratio": (mean([b for b, _ in xs]) / base) if base else None}

    out["efficiency"] = {
        "input_tokens": paired(lambda r: (r.get("usage") or {}).get("input")),
        "output_tokens": paired(lambda r: (r.get("usage") or {}).get("output")),
        "peak_input_tokens": paired(lambda r: max((x.get("input") or 0) for x in r["rounds"]) if r.get("rounds") else None),
        "tool_calls": paired(lambda r: r.get("tool_calls")),
        "wall_s": paired(lambda r: r.get("wall_s")),
        "answer_chars": paired(lambda r: len(r.get("answer") or "") or None),
    }
    order = sorted(zip(d, qs))
    picks = order[:3] + [t for t in order[-3:][::-1] if t not in order[:3]]
    for x, q in picks:
        out["largest"].append({"delta": x, "question": q[:120],
                               a: {"read": (A["answers"].get(q) or {}).get("files_read"), "steps": (A["answers"].get(q) or {}).get("steps"),
                                   "retried": (A["answers"].get(q) or {}).get("retried")},
                               b: {"read": (B["answers"].get(q) or {}).get("files_read"), "steps": (B["answers"].get(q) or {}).get("steps"),
                                   "retried": (B["answers"].get(q) or {}).get("retried")}})
    return out


def print_report(run, manifest, arms_sum, pairs):
    print(f"run: {run}")
    if manifest:
        print(f"manifest: harness={manifest.get('harness_sha')} created={manifest.get('created')} "
              f"repo={manifest.get('repo')} commit={str(manifest.get('repo_commit') or '')[:12]} "
              f"index_entries={(manifest.get('index') or {}).get('entries')}")
        for arm, m in (manifest.get("arms") or {}).items():
            print(f"  arm {arm}: kind={m.get('kind')} model={m.get('model')} params={m.get('params')}")
        for arm, m in (manifest.get("judge") or {}).items():
            print(f"  judge {arm}: model={m.get('model')} votes={m.get('votes')}")
    print()
    hdr = (f"{'arm':14} {'n':>3} {'err':>3} {'rty':>3} | {'score':>6} {'sd':>5} {'what':>6} {'how':>6} {'why':>6} {'where':>6} | "
           f"{'steps':>5} {'calls':>5} {'files':>5} {'+map':>5} | {'in_tok':>8} {'peak_in':>8} {'out_tok':>7} {'ans_ch':>7} {'$/q':>6} {'wall':>6} | {'rec_rd':>6} {'rec_mp':>6} {'rec_an':>6}")
    print(hdr)
    print("-" * len(hdr))
    for s in arms_sum:
        c = s["by_category"]
        print(f"{s['arm']:14} {s['n_valid']:3d} {s['errors']:3d} {s['retried']:3d} | {fmt(s['score_mean'])} {fmt(s['score_sd'],5)} "
              f"{fmt(c['what'])} {fmt(c['how'])} {fmt(c['why'])} {fmt(c['where'])} | "
              f"{fmt(s['steps'] if s['steps'] is not None else float('nan'),5)} {fmt(s['tool_calls'] if s['tool_calls'] is not None else float('nan'),5)} "
              f"{fmt(s['files_read'] if s['files_read'] is not None else float('nan'),5)} "
              f"{fmt(s['files_via_map'] if s['files_via_map'] is not None else float('nan'),5)} | "
              f"{fmt(s['in_tok'],8,0)} {fmt(s['peak_in_tok'] if s['peak_in_tok'] is not None else float('nan'),8,0)} "
              f"{fmt(s['out_tok'],7,0)} {fmt(s['ans_chars'],7,0)} "
              f"{fmt(s['cost_per_q'] if s['cost_per_q'] is not None else float('nan'),6,3)} {fmt(s['wall_s'])} | "
              f"{fmt(s['recall_read'] if s['recall_read'] is not None else float('nan'),6,2)} "
              f"{fmt(s['recall_seen'] if s['recall_seen'] is not None else float('nan'),6,2)} "
              f"{fmt(s['recall_ans'] if s['recall_ans'] is not None else float('nan'),6,2)}")
    print("dims:", "  ".join(f"{s['arm']}=" + "/".join(f"{s['dims'][d]:.1f}" for d in DIMS) for s in arms_sum if s["n_scored"]))
    print("judge vote SD (mean over questions):", "  ".join(f"{s['arm']}={s['judge_vote_sd']:.1f}" for s in arms_sum if s.get("judge_vote_sd") is not None) or "n/a (1 vote)")
    mp = [f"{s['arm']}={s['map_calls']:.1f} calls/{s['map_chars']:.0f}ch (used on {100*s['map_used_share']:.0f}% of questions)"
          for s in arms_sum if s.get("map_calls") is not None]
    if mp:
        print("map tools:", "  ".join(mp))
    nud = [f"{s['arm']}={s['min_steps_nudged']}" for s in arms_sum if s.get("min_steps_nudged")]
    if nud:
        print("min_steps nudges:", "  ".join(nud))
    print("totals:", "  ".join(f"{s['arm']}: in={s['tokens_total']['input']:,} out={s['tokens_total']['output']:,}"
                               + (f" ${s['cost_total']:.2f}" if s['cost_total'] is not None else "") for s in arms_sum))
    for p in pairs:
        print(f"\n== {p['pair']}: n={p['n']} Δ={p['mean_delta']:+.1f} 95%CI=[{p['ci95'][0]:+.1f},{p['ci95'][1]:+.1f}] "
              f"median={p['median_delta']:+.1f} win/tie/loss={p['win']}/{p['tie']}/{p['loss']} sign-test p={p['sign_test_p']:.3f} "
              f"| n for 80% power at this Δ ≈ {p['n_for_80pct_power']:.0f}")
        print("   by category: " + "  ".join(f"{c}: {v['delta']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}] (n={v['n']})"
                                             for c, v in p["by_category"].items() if v["n"]))
        eff = p.get("efficiency") or {}
        cells = []
        for k in ("input_tokens", "peak_input_tokens", "output_tokens", "tool_calls", "wall_s", "answer_chars"):
            v = eff.get(k)
            if v:
                cells.append(f"{k}: {v['delta']:+.0f} [{v['ci95'][0]:+.0f},{v['ci95'][1]:+.0f}]"
                             + (f" ({v['ratio']:.2f}x)" if v.get("ratio") else ""))
        if cells:
            print("   paired efficiency (b−a): " + "  ".join(cells))
        if "first_tool" in p:
            print(f"   first tool: {p['first_tool']}")
            print(f"   steps (b−a): fewer/same/more = {p['steps']['fewer']}/{p['steps']['same']}/{p['steps']['more']}; hit cap: {p['hit_cap']}")
        for L in p["largest"]:
            keys = [k for k in L if k not in ("delta", "question")]
            print(f"   {L['delta']:+.0f}  {L['question']}")
            for k in keys:
                if L[k]["steps"] is None and L[k]["read"] is None:
                    if L[k]["retried"]:
                        print(f"         {k:14} retried={L[k]['retried']}")
                    continue
                print(f"         {k:14} read={L[k]['read']} steps={L[k]['steps']} retried={L[k]['retried']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pairs", default="", help="comma list of b:a (b minus a)")
    ap.add_argument("--baseline", default=None, help="default comparison base (default: 'direct' if present)")
    ap.add_argument("--json", default=None, help="write the full analysis to this path")
    a = ap.parse_args()
    arms = load_run(a.run)
    if not arms:
        sys.exit(f"no answers-*.jsonl under {a.run}")
    prices = load_prices()
    manifest = common.read_manifest(a.run)
    set_category_axis(((manifest or {}).get("spec") or {}).get("questions"))
    arms_sum = [arm_summary(arm, d, prices) for arm, d in arms.items()]
    pairs_spec = [tuple(x.split(":")) for x in a.pairs.split(",") if ":" in x]
    if not pairs_spec:
        base = a.baseline or ("direct" if "direct" in arms else next(iter(arms)))
        pairs_spec = [(b, base) for b in arms if b != base]
        for b, x in (("agent-inject", "agent"), ("agent-aoci", "agent"), ("inject", "direct"), ("aoci-v2", "direct")):
            if b in arms and x in arms and (b, x) not in pairs_spec:
                pairs_spec.append((b, x))
    pairs = [pair_summary(b, x, arms[b], arms[x]) for b, x in pairs_spec if b in arms and x in arms and arms[b]["scores"] and arms[x]["scores"]]
    print_report(a.run, manifest, arms_sum, pairs)
    if a.json:
        json.dump({"run": a.run, "manifest": manifest, "arms": arms_sum, "pairs": pairs},
                  open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
