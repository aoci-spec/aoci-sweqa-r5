#!/usr/bin/env python3
"""r5 — SWE-QA all 15 repos: write the main three-arm k=10 specs and the parity (bare OpenHands, single run) specs.

  python3 harness/r5_make_specs.py            # writes specs/sweqa-r5-<repo>-k10.json and specs/sweqa-r5p-<tag>-<repo>-k1.json

Main spec = r4's spec shape (run.py) with: three arms oh / oh-aoci / oh-cg, draws 10, repo_root repos-r5/<repo> (code-only
Managed Scope: full profile + observe ** + index **/*.py|pyx|pxd|pyi), questions benchmarks/swe-qa/<repo>.jsonl (HF split as-is).
Parity spec = one bare arm, draws 1, the paper-row model profile; same judge.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = ["astropy", "conan", "django", "flask", "matplotlib", "pylint", "pytest", "reflex", "requests",
         "scikit-learn", "sphinx", "sqlfluff", "streamlink", "sympy", "xarray"]
PARITY = {"gpt51": ("closeai-gpt51", "GPT-5.1 (gpt-5.1-2025-11-13 via the CloseAI relay, reasoning_effort none); paper Table 3 OpenHands 70.79"),
          "glm46": ("openrouter-glm46-novita", "GLM-4.6 via OpenRouter pinned to Novita bf16; paper Table 3 OpenHands 70.15")}

MAIN_COMMENT = (
    "R5 — SWE-QA ALL 15 REPOS (2026-10-16; published as a reproduction package). "
    "Host: OpenHands 1.1.0 default agent (terminal + file_editor + task_tracker), max 10 reasoning-action iterations, temperature 0 — the paper's protocol. "
    "Three arms, k=10 interleaved: oh (bare) / oh-aoci (AOCI served by aoci-code 0.1.0-rc20 MCP, nine tools; every question first runs a COGNITION TURN with its own "
    "budget (<=30 iterations): aoci_rules -> aoci_overview through every next_cursor to completed=true (chunk_tokens=24000) -> attestation, then the paper's 10-iteration "
    "question turn) / oh-cg (CodeGraph 1.5.0 served as an MCP tool from a read-only mirror). No mixed arm. "
    "Index scope (all 15 repos, uniform): Managed Scope full profile + observe ** (order 0) + index **/*.py, **/*.pyx, **/*.pxd, **/*.pyi (order 10-13) + exclude AGENTS.md: "
    "Python/Cython sources INCLUDING tests get Entries (173/720 questions reference test files), non-code files are observed; this keeps every whole index under the rc20 "
    "cognition budget (whole_index.max_tokens 400k; django 2,845 entries is the largest). Entries authored by Claude Sonnet 5.5 (CloseAI) with s3_author under the rc20 quota "
    "profile and written through the governed path (aoci_maintain -> aoci_update_entry) until verify reports aligned; the four r4 repos are re-indexed under this scope and re-run "
    "(r4 used the full profile, so r4 numbers are not mixed in). Answer model DeepSeek-V4.1-Flash (official deepseek-flash, thinking disabled, 1M context). "
    "Judge: claude-sonnet-4-5 x5 votes per answer, per-dimension majority (median if no majority), five dimensions 1-20 = 100, rubric sha 750916e2 (the benchmark's); only rows with 5 valid votes count. "
    "PREREGISTERED (frozen before the first answer): H1 oh-aoci - oh > 0; H2 oh-aoci - oh-cg (two-sided); H3 oh-cg - oh (two-sided). Per repo: run-level exact permutation 10 v 10 "
    "(184,756 splits, two-sided). Merged: 15-repo stratified permutation (MC 1e5, equal repo weights). Exploratory: question category by first word; effect vs index size "
    "(<=2 / 3-5 / >=8 overview chunks); the 173 test-referencing questions. Absolute scores are not comparable with r4 or batch one. "
    "Acceptance gate before judging: 0 leaks, 0 network successes, delivery sha asserted per record, workspace-modified records discarded and re-answered (audited in discards.jsonl)."
)
PARITY_COMMENT = (
    "R5 PARITY CHECK (2026-10-16): bare OpenHands 1.1.0 default agent, 10 iterations, temperature 0, ONE run per question (the paper's protocol), same judge as the main batch "
    "(claude-sonnet-4-5 x5 majority, 1-20 x5 = 100). Purpose: show this pipeline reproduces the published Table 3 OpenHands row for a paper model before AOCI is added. "
    "Acceptance band: merged 15-repo mean within 3 points of the paper's number and all five dimensions in the same direction; outside the band -> investigate "
    "(iteration cap, SDK version, reasoning parameters, provider precision) before any main-batch claim. No AOCI, no CodeGraph in this run. Model: {model_note}."
)


def main():
    tpl = json.load(open(os.path.join(ROOT, "specs", "sweqa-r4-xarray-k10.json"), encoding="utf-8"))
    base_arm = next(a for a in tpl["arms"] if a["name"] == "oh")
    os.makedirs(os.path.join(ROOT, "specs"), exist_ok=True)
    written = []
    for r in REPOS:
        spec = {"run": f"sweqa-r5-{r}-k10", "repo": r, "repo_root": f"repos-r5/{r}", "questions": f"benchmarks/swe-qa/{r}.jsonl",
                "draws": 10, "limit": 0, "workers": tpl["workers"], "concurrent_passes": tpl["concurrent_passes"],
                "arms": [dict(base_arm, name=n, kind=n) for n in ("oh", "oh-aoci", "oh-cg")],
                "judge": dict(tpl["judge"]), "require_complete_index": True, "_comment": MAIN_COMMENT}
        p = os.path.join(ROOT, "specs", f"sweqa-r5-{r}-k10.json")
        json.dump(spec, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); written.append(p)
        for tag, (prof, note) in PARITY.items():
            pspec = {"run": f"sweqa-r5p-{tag}-{r}-k1", "repo": r, "repo_root": f"repos-r5/{r}", "questions": f"benchmarks/swe-qa/{r}.jsonl",
                     "draws": 1, "limit": 0, "workers": 16, "concurrent_passes": 1,
                     "arms": [dict(base_arm, name="oh", kind="oh", profile=prof)],
                     "judge": dict(tpl["judge"]), "require_complete_index": False,
                     "_comment": PARITY_COMMENT.format(model_note=note)}
            pp = os.path.join(ROOT, "specs", f"sweqa-r5p-{tag}-{r}-k1.json")
            json.dump(pspec, open(pp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); written.append(pp)
    print(f"wrote {len(written)} specs")


if __name__ == "__main__":
    main()
