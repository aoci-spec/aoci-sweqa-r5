#!/bin/bash
# r5_rejudge_waiter.sh — wait until the judge route (gptnb) answers again, then resume every r5 main run through the normal
# runspec (answers complete → gate → judge resumes rows with <5 votes → analysis), two at a time, small repos first.
cd /root/aoci-eval; set -a; . config/secrets.env; . config/aoci-agent-deepseek.env; set +a
echo "[rejudge] waiter start $(date +%T)" >> runs/r5-orch.log
head -1 runs/sweqa-r5-requests-k10/answers-oh-1.jsonl > runs/r5-judge-probe.jsonl
while :; do
  rm -f runs/r5-judge-probe.scores.jsonl
  timeout 180 .venv/bin/python3 harness/judge.py --answers runs/r5-judge-probe.jsonl --out runs/r5-judge-probe.scores.jsonl --votes 1 --profile gptnb-judge --workers 1 > runs/r5-judge-probe.log 2>&1
  if grep -q "votes=1 *$" runs/r5-judge-probe.log || grep -q "total=[0-9]* votes=1" runs/r5-judge-probe.log; then echo "[rejudge] judge route back $(date +%T)" >> runs/r5-orch.log; break; fi
  sleep 600
done
# wait for any answer-only runs still going
while tmux ls 2>/dev/null | grep -q "^r5-run-.*-nojudge"; do sleep 60; done
ORDER="requests flask pytest xarray reflex streamlink sqlfluff conan sphinx astropy scikit-learn matplotlib sympy pylint django"
for r in $ORDER; do
  while [ "$(tmux ls 2>/dev/null | grep -c "^r5-rejudge-")" -ge 2 ]; do sleep 60; done
  tmux new-session -d -s r5-rejudge-$r "bash /root/aoci-eval/harness/r5_runspec.sh $r"; echo "[rejudge] started $r $(date +%T)" >> runs/r5-orch.log; sleep 20
done
while tmux ls 2>/dev/null | grep -q "^r5-rejudge-"; do sleep 60; done
echo "[rejudge] ALL RESUMED RUNS ENDED $(date +%T)" >> runs/r5-orch.log
