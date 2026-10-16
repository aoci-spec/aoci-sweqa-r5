# SWE-QA × AOCI, all 15 repositories (r5, 2026-10-16)

A full-benchmark rerun of [SWE-QA](https://github.com/peng-weihan/SWE-QA-Bench) (ACL 2026 Findings; 720 repository-level
code questions over 15 Python projects) measuring what an [AOCI](https://github.com/aoci-spec/aoci-code) repository
cognition index does for an agent that has to answer them. Three arms, same host, same budget, k = 10 runs per arm:

| arm | what the agent gets |
|---|---|
| `oh` | bare OpenHands 1.1.0 default agent (terminal / file_editor / task_tracker), max 10 reasoning–action iterations, temperature 0 — the benchmark paper's own configuration |
| `oh-aoci` | the same, plus the AOCI index served as MCP tools (aoci-code 0.1.0-rc20). Before every question the agent runs the product's cognition turn: `aoci_rules` → `aoci_overview` through every `next_cursor` until the whole index is delivered → attestation; then the paper's 10-iteration question turn |
| `oh-cg` | the same, plus [CodeGraph](https://pypi.org/project/codegraph/) 1.5.0 served as an MCP tool from a read-only mirror |

Answer model: DeepSeek-V4.1-Flash (official `deepseek-flash`, thinking disabled). Judge: the benchmark's own rubric with
Claude Sonnet 4.5, five votes per answer, per-dimension majority (median when no majority); five dimensions × 1–20 = 100.
Every number below comes from rows with five valid votes; resumed passes take the last record per question.

This batch is published as a stand-alone reproduction package.

## Results

### Per repository (mean over 10 runs of the 48-question mean; "chunks" = overview chunks the AOCI arm had to read to load the whole index)

| repo | chunks | oh | oh-aoci | oh-cg | AOCI − bare | AOCI − CodeGraph | CodeGraph − bare |
|---|---|---|---|---|---|---|---|
| astropy | 5 | 76.15 | 77.35 | 76.61 | +1.20 | +0.74 | +0.45 |
| conan | 5 | 69.21 | 70.76 | 70.08 | +1.55 | +0.68 | +0.87 |
| django | 10 | 77.69 | 80.69 | 78.29 | +3.01 | +2.41 | +0.60 |
| flask | 1 | 74.70 | 75.66 | 76.24 | +0.96 | −0.58 | +1.54 |
| matplotlib | 4 | 81.25 | 83.72 | 81.92 | +2.47 | +1.80 | +0.67 |
| pylint | 8 | 67.59 | 65.46 | 69.38 | −2.12 | −3.92 | +1.79 |
| pytest | 2 | 76.83 | 81.32 | 79.26 | +4.49 | +2.06 | +2.43 |
| reflex | 2 | 79.03 | 78.59 | 79.41 | −0.44 | −0.82 | +0.38 |
| requests | 1 | 81.87 | 79.97 | 83.03 | −1.90 | −3.06 | +1.16 |
| scikit-learn | 5 | 79.73 | 82.11 | 81.44 | +2.37 | +0.66 | +1.71 |
| sphinx | 3 | 72.81 | 76.99 | 74.77 | +4.18 | +2.22 | +1.95 |
| sqlfluff | 3 | 74.39 | 76.92 | 75.13 | +2.53 | +1.79 | +0.74 |
| streamlink | 2 | 74.78 | 74.28 | 75.34 | −0.50 | −1.05 | +0.55 |
| sympy | 8 | 69.64 | 70.41 | 71.22 | +0.76 | −0.82 | +1.58 |
| xarray | 2 | 83.38 | 85.28 | 84.07 | +1.90 | +1.21 | +0.69 |

Per-repository significance (exact 10 v 10 run-level permutation, two-sided) is in `runs/<repo>/runlevel-stats.txt`.

### Preregistered comparisons (15-repo stratified permutation, repo = stratum, equal repo weights, 100,000 draws)

| comparison | merged Δ | p | repos in the same direction |
|---|---|---|---|
| H1 AOCI − bare | **+1.36** | < 0.0001 | 11 / 15 |
| H2 AOCI − CodeGraph | **+0.22** | 0.21 | 9 / 15 |
| H3 CodeGraph − bare | **+1.14** | < 0.0001 | 15 / 15 |

Read it as: the index helps, by about 1.4 points on the 100-point judge scale, consistently across most repositories; a
code graph served the same way helps by a similar amount; averaged over all 15 repositories the two cannot be told apart.

### Exploratory cuts (same test, declared in the spec before the first answer)

By question type (first word):

| type | AOCI − bare | AOCI − CodeGraph | CodeGraph − bare |
|---|---|---|---|
| what | −0.39 (p = 0.24) | −1.84 (p < 0.001) | +1.44 (p < 0.001) |
| why | +3.38 (p < 0.001) | +1.33 (p < 0.001) | +2.05 (p < 0.001) |
| where | +1.61 (p < 0.001) | +1.31 (p < 0.001) | +0.30 (p = 0.28) |
| how | +0.91 (p = 0.002) | +0.26 (p = 0.39) | +0.65 (p = 0.007) |

By index size (overview chunks the AOCI arm reads per question):

| size | repos | AOCI − bare | AOCI − CodeGraph | CodeGraph − bare |
|---|---|---|---|---|
| ≤ 2 chunks | flask, pytest, reflex, requests, streamlink, xarray | +0.75 (p = 0.009) | −0.37 (p = 0.15) | +1.12 (p < 0.001) |
| 3–5 chunks | astropy, conan, matplotlib, scikit-learn, sphinx, sqlfluff | +2.38 (p < 0.001) | +1.32 (p < 0.001) | +1.07 (p < 0.001) |
| ≥ 6 chunks | django, pylint, sympy | +0.55 (p = 0.29) | −0.78 (p = 0.17) | +1.32 (p < 0.001) |

Questions that name a test file (`tests/…`, `test_*.py`) in the question or the reference answer — 173 of 720:

| subset | questions | AOCI − bare | AOCI − CodeGraph | CodeGraph − bare |
|---|---|---|---|---|
| test-referencing | 173 | +3.31 (p < 0.001) | +2.25 (p < 0.001) | +1.07 (p < 0.001) |
| the rest | 547 | +0.89 (p < 0.001) | −0.25 (p = 0.24) | +1.14 (p < 0.001) |

**Post-hoc cut, not preregistered** — repositories with at least 50k lines of Python (13 of 15; requests at 11k and
flask at 18k are the only ones below; a bare agent reads such a repository in two or three files, so no map can add
information). Same stratified permutation, 13 strata, equal weights; this cut was chosen after seeing the per-repository
results, so read it as a hypothesis for the next run, not as a finding of this one.

| scope (13 repos ≥ 50k LOC) | AOCI | CodeGraph | bare | AOCI − bare | AOCI − CodeGraph | CodeGraph − bare |
|---|---|---|---|---|---|---|
| total | 77.22 | 76.68 | 75.58 | +1.65 (p < 0.001, 10/13) | +0.54 (p = 0.005, 9/13) | +1.11 (p < 0.001, 13/13) |
| why | 80.10 | 78.17 | 76.22 | +3.88 (p < 0.001) | +1.93 (p < 0.001) | +1.95 (p < 0.001) |
| where | 76.23 | 74.94 | 74.53 | +1.70 (p < 0.001) | +1.29 (p = 0.002) | +0.41 (p = 0.19) |
| how | 80.29 | 79.60 | 79.15 | +1.14 (p = 0.001) | +0.70 (p = 0.034) | +0.45 (p = 0.095) |
| what | 72.26 | 73.85 | 72.36 | −0.10 (p = 0.78) | −1.59 (p < 0.001) | +1.49 (p < 0.001) |
| completeness (1–20) | 13.60 | 13.14 | 12.74 | +0.86 (p < 0.001) | +0.46 (p < 0.001) | +0.40 (p < 0.001) |
| reasoning (1–20) | 14.68 | 14.36 | 14.01 | +0.68 (p < 0.001) | +0.32 (p < 0.001) | +0.36 (p < 0.001) |
| relevance (1–20) | 17.69 | 17.97 | 17.97 | −0.28 (p < 0.001) | −0.28 (p < 0.001) | 0.00 (p = 0.99) |
| correctness (1–20) | 15.23 | 15.24 | 14.99 | +0.23 (p < 0.001) | −0.01 (p = 0.84) | +0.24 (p < 0.001) |
| clarity (1–20) | 16.02 | 15.97 | 15.86 | +0.16 (p < 0.001) | +0.05 (p = 0.08) | +0.11 (p < 0.001) |

Leave-one-repository-out on the 13: AOCI − CodeGraph stays between +0.38 and +0.91; the weakest case (django left out)
has p = 0.054.

### Behaviour behind the numbers

Across the 15 repositories the AOCI arm reads fewer source files than the bare arm and answers without opening any
source file in 20–40 % of questions (pylint 38 %, the one repository where it loses); its answers are 1.3–1.5× longer. The
cognition turn delivers the complete index on every question (100 % `completed`, 100 % attestation submitted) but the
model almost never passes the product's attestation challenge (final state `delivery_verified` on 99 % of questions):
with this answer model the protocol is a delivery mechanism, not verified understanding. Full panels per repository and
arm are in `tables/SWEQA-R5-REPORT.md`.

### Parity check — does this pipeline reproduce the paper's numbers?

Bare OpenHands, one run per question (the paper's protocol), a model from the paper's Table 3, same judge:

| model | route | our 15-repo mean | paper (OpenHands row) | difference |
|---|---|---|---|---|
| GPT-5.1 (`gpt-5.1-2025-11-13`, reasoning none) | CloseAI relay | 69.96 | 70.79 | −0.83 |
| GLM-4.6 (bf16) | OpenRouter → Novita | 79.40 | 70.15 | +9.25 |

GPT-5.1 lands within a point of the published number with the same five-dimension profile, which is what the pipeline
had to show. GLM-4.6 does not reproduce: through Novita it writes 1.7× longer answers, hits the 10-iteration cap on 41 % of
questions and scores higher on every dimension; we report it and do not use it as evidence of parity.

## What is in this package

```
tables/     SWEQA-R5-scores-flat.csv (21,600 rows: repo, arm, run, qid, category, total, five dimensions, votes, behaviour flags)
            SWEQA-R5-arm-run-means.csv (450 rows: the direct input of the run-level permutation tests)
            SWEQA-R5-parity-flat.csv, SWEQA-R5-REPORT.{md,json} (merged report), AUDIT.txt (closing audit, PASS/FAIL per check)
runs/<repo>/            answers-<arm>-<run>.jsonl.gz (every answer with its tool trajectory, files read, cognition-turn trail, usage)
                        scores-<arm>-<run>.jsonl.gz (five raw judge votes per answer and the aggregate)
                        manifest.json, acceptance.json (gate), analysis.json, runlevel-stats.{json,txt}, discards.jsonl, rejudge.jsonl,
                        _superseded-chunk7000/ (passes answered under the wrong chunk setting; kept aside, not used)
parity/<model>/<repo>/  the single-run parity answers and scores
indexes/<repo>/         aoci.txt, aoci.meta.txt, aoci.code.txt (the governed index volumes the MCP server served), aoci-config.json,
                        s3-ledger.jsonl (per-entry authoring ledger: source sha256, tokens), s3-author.log, apply.log
specs/                  the 15 main specs and 30 parity specs; each spec's `_comment` is the preregistration, frozen before the first answer
harness/                the harness files this batch ran (a subset of aocibench-harness, MIT; closing commit b2409be2, the batch ran on d1b6ccbf → b2409be2; see harness/README.md)
logs/                   orchestration, authoring and governed-apply logs
```

## How it was run

1. **Repositories.** The 15 SWE-QA repositories at the benchmark's pinned commits (`repo_commit.txt`), question banks
   taken from the Hugging Face dataset `swe-qa/SWE-QA-Benchmark` as-is (48 per split).
2. **Index scope.** aoci-code's cognition budget caps a whole index at 400k tokens, which rules out the `full` profile for
   django (6,925 files) and also the `production` profile (docs, locale and templates are indexed there; tests are not,
   yet 173 of the 720 questions name a test file). All 15 indexes therefore use one uniform scope: the `full` profile plus
   an `observe **` rule and `index **/*.py, **/*.pyx, **/*.pxd, **/*.pyi` rules — every Python/Cython source including
   tests gets an entry, nothing else does. 13,491 entries in total (django 2,845, the largest). The rules are written into
   `.aoci/config.json` before the first `aoci scan`, and `overview_delivery.chunk_tokens` is set to 24000.
3. **Index authoring.** Entries authored by Claude Sonnet 5.5 (`harness/s3_author.py`, rc20 quota profile) from the
   complete file text, then written through the product's governed path (`aoci_maintain` → `aoci_update_entry`,
   `harness/aoci_apply_entries.py`) until `aoci verify` reports aligned. Empty files are skipped by the product's rule.
   Authoring cost: $460 for 13,491 entries ($0.034 per entry).
4. **Runs.** `harness/run.py specs/sweqa-r5-<repo>-k10.json` (three arms, k = 10, interleaved), the acceptance gate
   (0 leaks, 0 network successes, delivery sha asserted per record, workspace-modified records discarded and re-answered)
   and the judge. `harness/r5_runspec.sh` wraps the resume/discard loop; `harness/r5_orch.sh` sequences the repositories.
5. **Statistics.** `harness/runlevel_stats.py` per repository (exact 10 v 10 permutation), `harness/r5_report.py` for the
   merged and exploratory layers (stratified permutation), `harness/r5_audit.py` for the closing audit.

## Deviations and disclosures

- **Chunk setting.** The governed clones were first built without `chunk_tokens=24000` (default 7000), which split sympy's
  index into 25 chunks and would have cut django's cognition turn at the 30-iteration cap. Fixed on all 15 clones before any
  affected pass was used; the six sympy `oh-aoci` passes and three django records answered under 7000 were set aside
  (`_superseded-chunk7000/`) and re-answered. Nothing from them enters any statistic.
- **Judge quota.** The judge route ran out of credit between 06:22 and 10:05; judging stopped after five refusals and
  resumed later; rows re-judged until they had five valid votes (`rejudge.jsonl`). For one sphinx question the judge kept
  emitting a malformed score object and then a corrected one; the parser accepts the last well-formed five-field object
  (`harness/judge.py`), with a higher retry budget for those rows.
- **Discarded records.** Two bare/CodeGraph answers ended by the host's stuck detector with an empty final answer
  (astropy `oh-7`, reflex `oh-cg-8`); they were discarded as error records and re-answered (`discards.jsonl`). One conan
  record failed on a terminal-initialisation error and was likewise re-answered.
- **Incomplete attestation.** On 4 of the 7,200 AOCI-arm questions (matplotlib 1, scikit-learn 2, sympy 1) the whole index
  was delivered but the model ended its turn before submitting the attestation; these records are kept and counted
  (`AUDIT.txt` lists them as the only failed check).
- **Workspace-dirty answers.** Some GLM-4.6 parity answers and a few main-batch answers changed the workspace
  fingerprint without any write command in the trajectory (view/grep/find only, or a `__pycache__`); the gate treats these
  as benign and keeps them. Real edits would have been discarded.
- **Preregistration text.** The `_comment` strings in the 15 main specs (and the copies embedded in each run's `manifest.json`
  and `analysis.json`) are the frozen preregistration, except that a clause about where the batch would be published was
  removed from this copy.
- **Absolute scores** are not comparable with the earlier four-repository batch (r4: full-profile indexes including
  documentation entries, same model) or with the paper's table (different models); only within-run paired differences are.

## License

Data, tables and reports: CC BY 4.0. Harness code: MIT. The question banks belong to the SWE-QA authors (Apache 2.0).
