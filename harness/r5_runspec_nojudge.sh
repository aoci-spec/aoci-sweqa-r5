#!/bin/bash
# r5_runspec_nojudge.sh <repo> — answer-only run (judge route out of quota): same auto-remedy loop, --skip-judge; judge later via r5_rejudge_waiter.sh
cd /root/aoci-eval; set -a; . config/secrets.env; . config/aoci-agent-deepseek.env; set +a
export AOCI_BIN=/root/aoci-eval/bin/aoci-rc20 AOCI_COGNITION_ITERS=30
r=$1; spec=sweqa-r5-$r-k10; run=runs/$spec
for attempt in 1 2 3 4; do
  echo "[orch] $spec (no-judge) attempt $attempt start $(date +%T)" >> runs/r5-orch.log
  .venv/bin/python3 harness/run.py specs/$spec.json --skip-judge >> $run.launch.log 2>&1; rc=$?
  echo "[orch] $spec (no-judge) attempt $attempt rc=$rc $(date +%T)" >> runs/r5-orch.log
  [ $rc -eq 0 ] && break
  tail -8 $run.launch.log | grep -q "acceptance gate failed" || break
  fixable=$(.venv/bin/python3 - "$run" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]+"/acceptance.json")); rows=d if isinstance(d,list) else (d.get("passes") or d.get("rows") or list(d.values()))
out=[]
for x in rows:
    if not (isinstance(x,dict) and x.get("hard_fail")): continue
    other = x.get("leaks") or x.get("net_success") or x.get("delivery_bad") or x.get("extra") or len(x.get("models") or [])>1
    if other: print("STOP",x["pass"]); sys.exit(0)
    out.append(x["pass"])
print(" ".join(out))
PY
)
  case "$fixable" in STOP*) echo "[orch] $spec needs a human: $fixable" >> runs/r5-orch.log; break;; esac
  for p in $fixable; do .venv/bin/python3 harness/discard_records.py $run --pass $p --dirty --error >> runs/r5-orch.log 2>&1; done
done
