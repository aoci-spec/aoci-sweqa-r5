#!/bin/bash
# r5_parity.sh <tag> — run the 15 parity specs (bare OpenHands, single run) for one paper-row model, sequentially.
cd /root/aoci-eval; tag=$1; ORDER="requests flask pytest xarray reflex streamlink sqlfluff conan sphinx astropy scikit-learn matplotlib sympy pylint django"
echo "[parity] $tag start $(date +%T)" >> runs/r5-orch.log
for r in $ORDER; do bash /root/aoci-eval/harness/r5_runspec.sh $r sweqa-r5p-$tag-$r-k1; done
echo "[parity] $tag ALL DONE $(date +%T)" >> runs/r5-orch.log
