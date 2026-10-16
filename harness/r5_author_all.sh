#!/bin/bash
# r5 bulk index authoring: 3 lanes over the 14 remaining repos (largest first), Sonnet 5.5 @ CloseAI, rc20 quota profile, 12 workers each.
# Governed apply is a separate step (r5_apply.sh) once the Managed Scope activation recipe is settled.
cd /root/aoci-eval || exit 1
set -a; . config/secrets.env; . config/aoci-agent-proxy.env; set +a
export AOCI_BIN=/root/aoci-eval/bin/aoci-rc20 AOCI_S_QUOTA_PROFILE=rc20
QUEUE="django pylint sympy conan scikit-learn matplotlib astropy sphinx streamlink sqlfluff reflex pytest xarray flask"
mkdir -p runs/r5-author; echo "$(date +%T) START lanes=3 queue=$QUEUE" >> runs/r5-author/orch.log
lane() {
  for r in "$@"; do
    [ -f runs/r5-author/$r.done ] && continue
    echo "$(date +%T) $r start" >> runs/r5-author/orch.log
    .venv/bin/python3 harness/s3_author.py --repo $r-r5 --profile closeai-sonnet55 --alt-profile closeai-sonnet55 --workers 12 > runs/r5-author/$r.log 2>&1; rc=$?
    if [ $rc -eq 3 ]; then  # SIZE ALARM at start → rerun with --force-size (projection is an estimate; rc20 budget is enforced at apply time)
      echo "$(date +%T) $r size-alarm → force-size" >> runs/r5-author/orch.log
      .venv/bin/python3 harness/s3_author.py --repo $r-r5 --profile closeai-sonnet55 --alt-profile closeai-sonnet55 --workers 12 --force-size --resume >> runs/r5-author/$r.log 2>&1; rc=$?
    fi
    echo "$(date +%T) $r rc=$rc $(grep "s3 done" runs/r5-author/$r.log | tail -1 | cut -c1-200)" >> runs/r5-author/orch.log
    [ $rc -eq 0 ] && touch runs/r5-author/$r.done
  done
}
# three lanes, sizes balanced: django alone; pylint+conan+astropy+streamlink+reflex+xarray; sympy+scikit-learn+matplotlib+sphinx+sqlfluff+pytest+flask
lane django & lane pylint conan astropy streamlink reflex xarray & lane sympy scikit-learn matplotlib sphinx sqlfluff pytest flask &
wait
echo "$(date +%T) ALL-LANES-DONE" >> runs/r5-author/orch.log
