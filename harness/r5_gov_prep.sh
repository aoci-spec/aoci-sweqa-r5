#!/bin/bash
# r5: (re)build the governed clones repos-r5/<repo> from the pristine pinned clones with the code-only Managed Scope written into
# .aoci/config.json BEFORE the first scan (rules + observe_change_policy=informational), so the first Baseline binds the final policy
# and no Scope Change / coverage-reduction review is ever pending. Template rules come from the authoring clone config (same rules).
set -u; AOCI=/root/aoci-eval/bin/aoci-rc20; SRC=/root/sweqa15/repos; cd /root/aoci-eval || exit 1
TPL=/root/aoci-eval/repos-deepswe/requests-r5/.aoci/config.json
for name in $(ls "$SRC"); do
  dst="repos-r5/$name"; rm -rf "$dst"; cp -a "$SRC/$name" "$dst"
  ( cd "$dst" && git clean -fdxq && "$AOCI" init --scope-profile full --repo . --quiet >/dev/null 2>&1
    python3 - "$TPL" <<'PY'
import json,sys
cfg=json.load(open(".aoci/config.json")); ms=cfg["managed_scope"]
ms["rules"]=json.load(open(sys.argv[1]))["managed_scope"]["rules"]; ms["observe_change_policy"]="informational"
json.dump(cfg,open(".aoci/config.json","w"),ensure_ascii=False,indent=2)
PY
    "$AOCI" config set overview_delivery.chunk_tokens 24000 --repo . --quiet >/dev/null 2>&1   # preregistered delivery (r4 parity); default 7000 → 3.4x more chunks
    "$AOCI" scan --repo . --quiet >/dev/null 2>&1; rc=$?
    echo "$(date +%T) $name scan_rc=$rc roles=$(python3 -c "import json,collections;d=json.load(open('.aoci/baseline.json'));print(dict(collections.Counter(v.get('role') for v in d['files'].values())))")" )
done
echo "$(date +%T) GOV-PREP-DONE"
