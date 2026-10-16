#!/usr/bin/env python3
"""SWE-QA r5 (2026-10-16) merged report: 15 repos × three OpenHands arms × k=10 (+ parity rows).

Per repo the preregistered test is runlevel_stats.py (exact 10v10 permutation). This script adds the merged layer and
the side panels, as r4_report.py did, plus the r5 preregistered exploratory cuts:
  * 15-repo stratified permutation (repo = stratum; MC N draws) on per-draw arm means — H1 AOCI−裸, H2 AOCI−CG, H3 CG−裸,
  * category rows (What/Why/Where/How by the question's first word), pooled the same way,
  * H5: effect by index size — strata by the number of overview chunks the AOCI arm had to read (≤2 / 3–5 / ≥6),
  * H6: the test-referencing questions (question or reference answer names tests/… or test_*.py) vs the rest,
  * parity rows: bare OpenHands single run with a paper-table model vs the paper's Table 3 OpenHands number,
  * behaviour + cognition-turn panels, discards and superseded passes, so every re-answered record is disclosed.

  python3 harness/r5_report.py [--runs-glob "runs/sweqa-r5-*-k10"] [--mc 100000] [--json out.json] [--md out.md]
Repos that do not yet have 10 fully judged draws for every arm are listed as incomplete and left out of the merged tests.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import statistics as st

ARMS = ("oh", "oh-aoci", "oh-cg")
PAIRS = (("H1 AOCI − 裸", "oh-aoci", "oh"), ("H2 AOCI − CodeGraph", "oh-aoci", "oh-cg"), ("H3 CodeGraph − 裸", "oh-cg", "oh"))
CATS = {"what": "what", "why": "why", "where": "where", "how": "how"}
PARITY = {"gpt51": ("GPT-5.1 (CloseAI relay, reasoning none)", 70.79, (12.26, 11.60, 16.82, 15.59, 14.52)),
          "glm46": ("GLM-4.6 (OpenRouter → Novita bf16)", 70.15, (11.91, 12.70, 16.90, 13.82, 14.82))}
DIMS = ("correctness", "completeness", "relevance", "clarity", "reasoning")
TEST_RE = re.compile(r"(^|[\s`'\"/(])tests?/|test_[a-z0-9_]+\.py|conftest\.py")


def load_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def last_per_question(rows):
    """Resumed passes append re-answers; gate, judge and stats take the LAST record per question."""
    d = {}
    for r in rows:
        d[r.get("question")] = r
    return d


def category(q):
    w = (q or "").strip().split()[:1]
    return CATS.get(w[0].lower().strip("?,.:") if w else "", "other")


def total_of(r):
    t = r.get("total")
    if t is None and r.get("scores"):
        t = r["scores"].get("total")
    return None if t is None else float(t)


def score_files(run_dir):
    for f in sorted(glob.glob(os.path.join(run_dir, "scores-*.jsonl"))):
        m = re.match(r"^scores-(oh|oh-aoci|oh-cg)-(\d+)\.jsonl$", os.path.basename(f))   # exact arm names (G62)
        if m:
            yield m.group(1), int(m.group(2)), f


def draw_means(run_dir, keep=None):
    """{arm: [mean over judged questions of draw 1..k]}; keep(question_text) filters the questions (category / test subset)."""
    out = collections.defaultdict(dict)
    for arm, draw, f in score_files(run_dir):
        rows = last_per_question(load_jsonl(f))
        vals = [total_of(r) for r in rows.values()
                if total_of(r) is not None and not r.get("error") and (r.get("votes") or 0) >= 5
                and (keep is None or keep(r.get("question")))]
        if vals:
            out[arm][draw] = st.mean(vals)
    return {a: [v for _, v in sorted(d.items())] for a, d in out.items()}


def complete(arms, k=10):
    return all(a in arms and len(arms[a]) == k for a in ARMS)


def stratified_perm(strata, b, a, mc, seed=20261016):
    rng = random.Random(seed)

    def stat(assign):
        return sum(st.mean(xb) - st.mean(xa) for xb, xa in assign.values()) / len(assign)
    obs = stat({r: (arms[b], arms[a]) for r, arms in strata.items()})
    cnt = 0
    pools = {r: arms[b] + arms[a] for r, arms in strata.items()}
    for _ in range(mc):
        assign = {}
        for r, pool in pools.items():
            p = pool[:]
            rng.shuffle(p)
            h = len(p) // 2
            assign[r] = (p[:h], p[h:])
        if abs(stat(assign)) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (mc + 1)


def answer_rows(run_dir, arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(run_dir, "answers-*.jsonl"))):
        m = re.match(r"^answers-(oh|oh-aoci|oh-cg)-(\d+)\.jsonl$", os.path.basename(f))
        if not m or m.group(1) != arm:
            continue
        rows.extend(last_per_question(load_jsonl(f)).values())
    return [r for r in rows if not r.get("error")]


def chunk_count(run_dir):
    """Overview chunks the AOCI arm read per question (mode over its records); None when the arm has no cognition trail."""
    counts = collections.Counter()
    for r in answer_rows(run_dir, "oh-aoci"):
        for c in (r.get("cognition") or {}).get("calls") or []:
            m = re.search(r'"chunk_count":(\d+)', str(c.get("obs_head") or ""))
            if m:
                counts[int(m.group(1))] += 1
                break
    if counts:
        return counts.most_common(1)[0][0]
    # a repo whose whole index fits one delivery reports no chunk_count: one chunk when the cognition turn completed
    return 1 if any(((r.get("cognition") or {}).get("completed_seen")) for r in answer_rows(run_dir, "oh-aoci")) else None


def size_stratum(chunks):
    if chunks is None:
        return "unknown"
    return "≤2 块" if chunks <= 2 else ("3–5 块" if chunks <= 5 else "≥6 块")


def behaviour(run_dir):
    out = {}
    for arm in ARMS:
        ok = answer_rows(run_dir, arm)
        if not ok:
            continue
        n = len(ok)

        def mean(key, sub=None):
            return st.mean([((r.get(sub) or {}).get(key) if sub else r.get(key)) or 0 for r in ok])
        b = {"n": n, "steps": mean("steps"), "tool_calls": mean("tool_calls"),
             "files_read": st.mean([len(r.get("files_read") or []) for r in ok]), "map_calls": mean("map_calls"),
             "forced_pct": 100 * sum(1 for r in ok if r.get("forced") or r.get("hit_cap")) / n,
             "zero_source_pct": 100 * sum(1 for r in ok if not (r.get("files_read") or [])) / n,
             "dirty": sum(1 for r in ok if r.get("workspace_dirty")), "contam": sum(1 for r in ok if r.get("contamination")),
             "wall_s": mean("wall_s"), "input_tokens": mean("input", "usage"), "cache_read": mean("cache_read", "usage"),
             "answer_chars": st.mean([len(r.get("answer") or "") for r in ok])}
        cog = [r.get("cognition") for r in ok if r.get("cognition")]
        if cog:
            att = collections.Counter((((c.get("attestation") or {}).get("model_attestation")
                                        or (c.get("attestation") or {}).get("cognition_state") or "unknown")) for c in cog)
            b["cognition"] = {"n": len(cog), "llm_calls": st.mean([c.get("llm_calls") or 0 for c in cog]),
                              "overview_calls": st.mean([c.get("overview_calls") or 0 for c in cog]),
                              "completed_pct": 100 * sum(1 for c in cog if c.get("completed_seen")) / len(cog),
                              "attestation_submitted_pct": 100 * sum(1 for c in cog if c.get("attestation_submitted")) / len(cog),
                              "final_state": dict(collections.Counter(c.get("final_state") or "?" for c in cog)),
                              "verdict": dict(att), "wall_s": st.mean([c.get("wall_s") or 0 for c in cog])}
        out[arm] = b
    return out


def test_questions(repo, root):
    """question_text -> True when the question or its reference answer names a test file (HF bank, same text)."""
    bank = os.path.join(root, "benchmarks", "swe-qa", f"{repo}.jsonl")
    flags = {}
    if os.path.exists(bank):
        for r in load_jsonl(bank):
            flags[r.get("question")] = bool(TEST_RE.search((r.get("question") or "") + " " + (r.get("answer") or "")))
    return flags


def parity(root, tag, repos):
    per = {}
    dims = collections.defaultdict(list)
    for repo in repos:
        rd = os.path.join(root, "runs", f"sweqa-r5p-{tag}-{repo}-k1")
        f = next((x for x in (os.path.join(rd, "scores-oh.jsonl"), os.path.join(rd, "scores-oh-1.jsonl")) if os.path.exists(x)), None)
        if not f:
            continue
        # 5 valid votes, or an empty final answer scored 0 by rule (votes_requested None): the paper's single run keeps it
        rows = [r for r in last_per_question(load_jsonl(f)).values()
                if total_of(r) is not None and not r.get("error")
                and ((r.get("votes") or 0) >= 5 or (r.get("votes_requested") is None and total_of(r) == 0))]
        if len(rows) == 48:
            per[repo] = st.mean(total_of(r) for r in rows)
            for d in DIMS:
                dims[d].extend(float((r.get("scores") or {}).get(d, 0)) for r in rows)
    return per, {d: st.mean(v) for d, v in dims.items()} if dims else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/sweqa-r5-*-k10")
    ap.add_argument("--mc", type=int, default=100000)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = sorted(glob.glob(os.path.join(root, a.runs_glob)))
    repos = {os.path.basename(r).replace("sweqa-r5-", "").replace("-k10", ""): r for r in runs}
    strata_all = {repo: draw_means(rd) for repo, rd in repos.items()}
    done = {r: v for r, v in strata_all.items() if complete(v)}
    pending = sorted(set(strata_all) - set(done))
    chunks = {repo: chunk_count(rd) for repo, rd in repos.items()}
    report = {"repos": {}, "merged": {}, "categories": {}, "size": {}, "tests": {}, "parity": {}, "behaviour": {}, "discards": {},
              "superseded": {}, "pending": pending, "chunks": chunks}
    lines = ["# SWE-QA r5 合并报告(15 仓 × 三臂 × k=10;GitHub 复现包)", ""]
    if pending:
        lines.append(f"> 未判完(不进合并检验):{', '.join(pending)}")
        lines.append("")
    lines.append("## 臂均值(每仓 10 轮均值的均值;判官总分 0–100;块 = AOCI 臂每题读完整份索引的 overview 块数)")
    lines.append("| 仓 | 块 | " + " | ".join(ARMS) + " | " + " | ".join(l.split(" ", 1)[1] for l, _, _ in PAIRS) + " |")
    lines.append("|---|---|" + "---|" * (len(ARMS) + len(PAIRS)))
    for repo, arms in strata_all.items():
        report["repos"][repo] = {arm: {"draws": len(v), "mean": st.mean(v), "sd": st.pstdev(v)} for arm, v in arms.items()}
        report["repos"][repo]["chunks"] = chunks[repo]
        cells = [f"{st.mean(arms[arm]):.2f}" if arm in arms else "—" for arm in ARMS]
        deltas = [f"{st.mean(arms[b]) - st.mean(arms[bb]):+.2f}" if b in arms and bb in arms else "—" for _, b, bb in PAIRS]
        lines.append(f"| {repo}{'' if repo in done else ' (未完)'} | {chunks[repo] or '—'} | " + " | ".join(cells + deltas) + " |")
    lines.append("")
    lines.append(f"## 预登记对比:{len(done)} 仓分层置换(MC {a.mc:,},仓等权;逐仓精确置换见各 run 的 runlevel-stats.json)")
    lines.append("| 对比 | 合并 Δ | p | 同向仓数 |")
    lines.append("|---|---|---|---|")
    for label, b, bb in PAIRS:
        obs, p = stratified_perm(done, b, bb, a.mc)
        same = sum(1 for x in done.values() if (st.mean(x[b]) - st.mean(x[bb])) * (1 if obs >= 0 else -1) > 0)
        report["merged"][label] = {"delta": obs, "p": p, "same_direction": same, "n_repos": len(done),
                                   "per_repo": {r: st.mean(x[b]) - st.mean(x[bb]) for r, x in done.items()}}
        lines.append(f"| {label} | **{obs:+.2f}** | {p:.4f} | {same}/{len(done)} |")
    lines.append("")
    lines.append("## 题类(首词 What/Why/Where/How;分层置换,探索性)")
    lines.append("| 题类 | " + " | ".join(l for l, _, _ in PAIRS) + " |")
    lines.append("|---|" + "---|" * len(PAIRS))
    for cat in ("what", "why", "where", "how"):
        cs = {repo: draw_means(rd, lambda q, c=cat: category(q) == c) for repo, rd in repos.items() if repo in done}
        cells = []
        report["categories"][cat] = {}
        for label, b, bb in PAIRS:
            usable = {r: arms for r, arms in cs.items() if complete(arms)}
            if not usable:
                cells.append("—")
                continue
            obs, p = stratified_perm(usable, b, bb, max(20000, a.mc // 5))
            report["categories"][cat][label] = {"delta": obs, "p": p, "n_repos": len(usable)}
            cells.append(f"{obs:+.2f} (p={p:.3f})")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## H5 效应随索引规模(按 AOCI 臂读完整份索引的块数分层;分层置换,探索性)")
    lines.append("| 规模层 | 仓 | " + " | ".join(l for l, _, _ in PAIRS) + " |")
    lines.append("|---|---|" + "---|" * len(PAIRS))
    groups = collections.defaultdict(list)
    for repo in done:
        groups[size_stratum(chunks[repo])].append(repo)
    for g in ("≤2 块", "3–5 块", "≥6 块", "unknown"):
        if g not in groups:
            continue
        sub = {r: done[r] for r in groups[g]}
        cells = []
        report["size"][g] = {"repos": groups[g]}
        for label, b, bb in PAIRS:
            obs, p = stratified_perm(sub, b, bb, max(20000, a.mc // 5))
            report["size"][g][label] = {"delta": obs, "p": p}
            cells.append(f"{obs:+.2f} (p={p:.3f})")
        lines.append(f"| {g} | {', '.join(groups[g])} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## H6 指向测试文件的题(题干或参考答案点名 tests/ 或 test_*.py;分层置换,探索性)")
    lines.append("| 子集 | 题数 | " + " | ".join(l for l, _, _ in PAIRS) + " |")
    lines.append("|---|---|" + "---|" * len(PAIRS))
    flags = {repo: test_questions(repo, root) for repo in repos}
    for name, want in (("指向测试", True), ("其余", False)):
        cs = {repo: draw_means(rd, lambda q, fl=flags[repo], w=want: fl.get(q, False) == w) for repo, rd in repos.items() if repo in done}
        usable = {r: arms for r, arms in cs.items() if complete(arms)}
        nq = sum(1 for repo in done for v in flags[repo].values() if v == want)
        cells = []
        report["tests"][name] = {"n_questions": nq, "n_repos": len(usable)}
        for label, b, bb in PAIRS:
            if not usable:
                cells.append("—")
                continue
            obs, p = stratified_perm(usable, b, bb, max(20000, a.mc // 5))
            report["tests"][name][label] = {"delta": obs, "p": p}
            cells.append(f"{obs:+.2f} (p={p:.3f})")
        lines.append(f"| {name} | {nq} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 对齐检查(裸 OpenHands,官方口径单跑;SWE-QA 论文 Table 3 OpenHands 行,总分 = 五维 1–20 之和)")
    lines.append("| 模型 | 已判仓 | 15 仓等权均分 | SWE-QA 论文 | 差 | 五维(正确/完整/相关/清晰/推理) | SWE-QA 论文五维 |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag, (desc, paper, pdims) in PARITY.items():
        per, dims = parity(root, tag, list(repos))
        if not per:
            continue
        merged = st.mean(per.values())
        report["parity"][tag] = {"desc": desc, "per_repo": per, "merged": merged, "paper": paper, "dims": dims}
        lines.append(f"| {desc} | {len(per)} | {merged:.2f} | {paper:.2f} | {merged - paper:+.2f} | "
                     + " / ".join(f"{dims.get(d, 0):.2f}" for d in DIMS) + " | " + " / ".join(f"{x:.2f}" for x in pdims) + " |")
    lines.append("")
    lines.append("## 行为与认知轮面板(每仓每臂 480 题;续答取末条)")
    for repo, rd in repos.items():
        beh = behaviour(rd)
        report["behaviour"][repo] = beh
        lines.append(f"### {repo}")
        lines.append("| 臂 | 步数 | 调用 | 读文件 | 图调用 | 顶格% | 零源码% | 脏区 | 污染 | 墙钟 s | 输入 token | 缓存 | 答案字符 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            b = beh.get(arm)
            if not b:
                continue
            lines.append(f"| {arm} | {b['steps']:.1f} | {b['tool_calls']:.1f} | {b['files_read']:.1f} | {b['map_calls']:.2f} | {b['forced_pct']:.0f} | "
                         f"{b['zero_source_pct']:.0f} | {b['dirty']} | {b['contam']} | {b['wall_s']:.0f} | {b['input_tokens']:.0f} | {b['cache_read']:.0f} | {b['answer_chars']:.0f} |")
        c = (beh.get("oh-aoci") or {}).get("cognition")
        if c:
            lines.append(f"- oh-aoci 认知轮:n={c['n']},LLM 调用 {c['llm_calls']:.1f},overview {c['overview_calls']:.1f},completed {c['completed_pct']:.0f}%,"
                         f"认证提交 {c['attestation_submitted_pct']:.0f}%,终态 {c['final_state']},判定 {c['verdict']},墙钟 {c['wall_s']:.0f} s")
        dpath = os.path.join(rd, "discards.jsonl")
        if os.path.exists(dpath):
            ds = load_jsonl(dpath)
            report["discards"][repo] = ds
            lines.append(f"- 丢弃并续答 {len(ds)} 条:" + "; ".join(f"{d.get('pass')}#{d.get('index')}({'dirty' if d.get('workspace_dirty') else 'error'})" for d in ds))
        sup = []
        for sd in sorted(glob.glob(os.path.join(rd, "_superseded-*"))):
            n = sum(len(load_jsonl(f)) for f in glob.glob(os.path.join(sd, "answers-*.jsonl")))
            sup.append(f"{os.path.basename(sd)}: {n} 条")
        if sup:
            report["superseded"][repo] = sup
            lines.append("- 作废重答(不入统计):" + "; ".join(sup))
        lines.append("")
    md = "\n".join(lines)
    print(md)
    if a.json:
        json.dump(report, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md + "\n")


if __name__ == "__main__":
    main()
