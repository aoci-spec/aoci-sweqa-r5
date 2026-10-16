#!/bin/bash
# r5 main-batch orchestrator: start each repo's three-arm k=10 run once its index is applied, at most 2 runs at a time, small repos first.
cd /root/aoci-eval; ORDER="requests flask pytest xarray reflex streamlink sqlfluff conan sphinx astropy scikit-learn matplotlib sympy pylint django"; MAXC=2
echo "[orch] r5 main orchestrator start $(date +%T)" >> runs/r5-orch.log; started=""
while :; do
  running=$(tmux ls 2>/dev/null | grep -c "^r5-run-")
  for r in $ORDER; do
    case " $started " in *" $r "*) continue;; esac
    [ -f runs/r5-author/$r.applied ] || continue
    [ $running -ge $MAXC ] && break
    tmux new-session -d -s r5-run-$r "bash /root/aoci-eval/harness/r5_runspec.sh $r"; started="$started $r"; running=$((running+1))
    echo "[orch] started $r $(date +%T)" >> runs/r5-orch.log
  done
  [ $(echo $started | wc -w) -eq 15 ] && break
  sleep 60
done
while tmux ls 2>/dev/null | grep -q "^r5-run-"; do sleep 60; done
echo "[orch] all 15 main runs ended $(date +%T)" >> runs/r5-orch.log
