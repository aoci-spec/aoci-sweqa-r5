#!/bin/bash
# r5: make governed clones (repos-r5/<repo>) and authoring clones (repos-deepswe/<repo>-r5) from the pinned pristine clones,
# apply the code-only Managed Scope (full + observe ** + index Python/Cython) and scan. No model calls.
set -u; AOCI=/root/aoci-eval/bin/aoci-rc20; SRC=/root/sweqa15/repos
cd /root/aoci-eval || exit 1; mkdir -p repos-r5 repos-deepswe
for name in $(ls "$SRC"); do
  for kind in gov auth; do
    if [ "$kind" = gov ]; then dst="repos-r5/$name"; else dst="repos-deepswe/$name-r5"; fi
    if [ -d "$dst/.aoci" ]; then echo "skip $dst"; continue; fi
    rm -rf "$dst"; cp -a "$SRC/$name" "$dst"
    ( cd "$dst" && git clean -fdxq && "$AOCI" init --scope-profile full --repo . --quiet >/dev/null 2>&1
      "$AOCI" scope rule add observe-all --action observe --pattern "**" --pattern-kind glob --order 0 --reason "r5 code-only cognition: non-code files observed" --repo . --quiet >/dev/null 2>&1
      i=10; for g in "**/*.py" "**/*.pyx" "**/*.pxd" "**/*.pyi"; do "$AOCI" scope rule add "index-code-$i" --action index --pattern "$g" --pattern-kind glob --order $i --reason "r5 code-only cognition: Python/Cython sources incl. tests" --repo . --quiet >/dev/null 2>&1; i=$((i+1)); done
      "$AOCI" scope rule add no-agents-md --action exclude --pattern AGENTS.md --pattern-kind file --order 20 --reason "init artifact" --repo . --quiet >/dev/null 2>&1
      "$AOCI" scan --repo . --quiet >/dev/null 2>&1; rc=$?
      n=$(python3 -c "import json;d=json.load(open('.aoci/baseline.json'));print(sum(1 for v in d['files'].values() if v.get('role')=='index'))" 2>/dev/null)
      echo "$(date +%T) $name $kind scan_rc=$rc index=$n" )
  done
done
echo "$(date +%T) PREP-DONE"
