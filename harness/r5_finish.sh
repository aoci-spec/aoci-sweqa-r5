#!/bin/bash
# r5 finish: run-level stats for every run, final merged report (MC 1e5), audit, flat tables, and a transfer pack.
cd /root/aoci-eval || exit 1; set -a; . config/aoci-agent-deepseek.env; set +a
echo "[finish] start $(date +%T)" >> runs/r5-orch.log
for d in runs/sweqa-r5-*-k10; do
  n=$(cat $d/scores-*.jsonl | grep -c "\"votes\": 5")
  [ "$n" -ge 1440 ] || { echo "[finish] $d not fully judged ($n) — stop" >> runs/r5-orch.log; exit 1; }
  .venv/bin/python3 harness/runlevel_stats.py $d --pairs "oh-aoci:oh=H1_aoci-bare,oh-aoci:oh-cg=H2_aoci-cg,oh-cg:oh=H3_cg-bare" --json $d/runlevel-stats.json > $d/runlevel-stats.txt 2>&1 || echo "[finish] runlevel_stats failed for $d" >> runs/r5-orch.log
done
mkdir -p runs/r5-final
python3 harness/r5_report.py --mc 100000 --json runs/r5-final/SWEQA-R5-REPORT.json --md runs/r5-final/SWEQA-R5-REPORT.md > /dev/null 2> runs/r5-final/report.err
python3 harness/r5_audit.py > runs/r5-final/AUDIT.txt 2>&1
python3 harness/r5_flat.py --out-dir runs/r5-final > runs/r5-final/flat.log 2>&1
# transfer pack: run dirs (answers with traces, scores with raw votes, manifests, gate, stats, discards, superseded), index artifacts, specs, parity runs
tar czf runs/r5-pack.tgz runs/sweqa-r5-*-k10 runs/sweqa-r5p-*-k1 runs/r5-index-artifacts runs/r5-final runs/r5-orch.log runs/r5-author/orch.log runs/r5-author/apply.log runs/r5-author/*.apply.log specs/sweqa-r5-*.json specs/sweqa-r5p-*.json 2> runs/r5-final/pack.err
echo "[finish] DONE $(date +%T) pack=$(stat -c %s runs/r5-pack.tgz) bytes; audit=$(tail -1 runs/r5-final/AUDIT.txt)" >> runs/r5-orch.log
