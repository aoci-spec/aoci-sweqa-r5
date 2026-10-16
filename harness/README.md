# Harness files used by this batch

A subset of aocibench-harness (MIT, see `LICENSE`) at commit b2409be2: only the files the r5 batch ran. Scripts for other
benchmarks are not included. Paths inside the `r5_*.sh` scripts are the ones used on the evaluation host (`/root/aoci-eval`).

| file | role |
|---|---|
| `run.py` | spec-driven driver: arms × draws interleaved, acceptance gate, judge, analysis; every stage resumable |
| `openhands_ask.py` | one OpenHands 1.1.0 question turn per record (bare / AOCI MCP / CodeGraph MCP): workspace slots, cognition turn, contamination and dirty-path checks |
| `judge.py` | SWE-QA LLM-as-a-judge rubric, five votes, per-dimension majority |
| `accept.py` | acceptance gate (leaks, network, delivery, dirty workspaces, error records) |
| `discard_records.py` | audited discard of error / dirty records before re-answering |
| `runlevel_stats.py` | exact run-level permutation tests |
| `r5_report.py`, `r5_audit.py`, `r5_flat.py` | merged report, closing audit, flat tables |
| `r5_make_specs.py` | writes the 15 main and 30 parity specs |
| `r5_prep.sh`, `r5_gov_prep.sh`, `r5_author_all.sh`, `r5_apply.sh` | pinned clones, governed code-only scope, index authoring, governed apply |
| `r5_runspec.sh`, `r5_runspec_nojudge.sh`, `r5_orch.sh`, `r5_parity.sh`, `r5_rejudge_waiter.sh`, `r5_finish.sh` | orchestration of the main batch, the parity runs, the re-judge after the judge outage, and the closing |
| `s3_author.py` (+ `s2_author.py`, `author.py`, `fras_lint.py`, `outline.py`, `ask.py`, `build_index_artifact.py`) | index authoring and its helpers |
| `aoci_apply_entries.py` | writes authored entries through the product's governed path (`aoci_maintain` → `aoci_update_entry`) |
| `common.py`, `llm.py`, `analyze.py` | shared helpers, model clients, per-run analysis |
| `rubrics/original-v1.txt` | the judge rubric text (sha256 750916e2…, the one every pass used) |
