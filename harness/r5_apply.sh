#!/bin/bash
# r5: as each repo's authoring finishes (runs/r5-author/<repo>.done), apply its entries into the governed clone through
# aoci_maintain -> aoci_update_entry, verify aligned, keep the index artifacts; loop until all 15 are applied.
cd /root/aoci-eval || exit 1; set -a; . config/aoci-agent-deepseek.env; set +a; export AOCI_BIN=/root/aoci-eval/bin/aoci-rc20
REPOS="astropy conan django flask matplotlib pylint pytest reflex requests scikit-learn sphinx sqlfluff streamlink sympy xarray"
mkdir -p runs/r5-author runs/r5-index-artifacts
while :; do
  pending=0
  for r in $REPOS; do
    [ -f runs/r5-author/$r.applied ] && continue
    [ -f runs/r5-author/$r.apply-failed ] && continue
    if [ -f runs/r5-author/$r.done ]; then
      echo "$(date +%T) $r apply start" >> runs/r5-author/apply.log
      .venv/bin/python3 harness/aoci_apply_entries.py --repo /root/aoci-eval/repos-r5/$r --entries /root/aoci-eval/repos-deepswe/$r-r5/aoci.code.txt --authoring-root /root/aoci-eval/repos-deepswe/$r-r5 --aoci-bin $AOCI_BIN > runs/r5-author/$r.apply.log 2>&1; rc=$?
      al=$($AOCI_BIN --repo /root/aoci-eval/repos-r5/$r verify --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); g=d.get('governance',{}); print(g.get('governance_aligned'), g.get('code_entry_count'), g.get('code_source_count'))")
      echo "$(date +%T) $r apply rc=$rc verify(aligned entries sources)=$al" >> runs/r5-author/apply.log
      if [ "${al%% *}" = "True" ]; then
        mkdir -p runs/r5-index-artifacts/$r && cp repos-r5/$r/aoci.txt repos-r5/$r/aoci.meta.txt repos-r5/$r/aoci.code.txt runs/r5-index-artifacts/$r/ && cp repos-r5/$r/.aoci/config.json runs/r5-index-artifacts/$r/aoci-config.json && cp deepswe-inject-s3/.ledgers/$r-r5/ledger.jsonl runs/r5-index-artifacts/$r/s3-ledger.jsonl 2>/dev/null; cp runs/r5-author/$r.log runs/r5-index-artifacts/$r/s3-author.log; cp runs/r5-author/$r.apply.log runs/r5-index-artifacts/$r/apply.log
        touch runs/r5-author/$r.applied
      else
        echo "$(date +%T) $r NOT aligned — needs a human" >> runs/r5-author/apply.log; touch runs/r5-author/$r.apply-failed
      fi
    else pending=1; fi
  done
  n=$(ls runs/r5-author/*.applied 2>/dev/null | wc -l); [ $n -eq 15 ] && break
  if grep -q ALL-LANES-DONE runs/r5-author/orch.log 2>/dev/null; then f=$(ls runs/r5-author/*.apply-failed 2>/dev/null | wc -l); [ $((n+f)) -ge 15 ] && break; fi
  sleep 60
done
echo "$(date +%T) APPLY-LOOP-DONE applied=$(ls runs/r5-author/*.applied 2>/dev/null | wc -l)" >> runs/r5-author/apply.log
