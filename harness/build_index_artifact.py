#!/usr/bin/env python3
"""Build the ONE index file every arm injects, and its provenance sidecar.

Why a built artefact instead of assembling the index at run time:
  - every host (OpenHands adapter, minimal agent, anything else) injects the SAME bytes, so a
    "map on / map off" contrast never hides a difference in what the map looked like;
  - the bytes carry a sha256 that goes into every record, so "how much of the index reached the
    model" is an assertion (delivered == artefact size), not a hope — the SDK's silent 50k
    middle-out truncation (memo §9.36) got through precisely because nothing asserted this;
  - the run-time fixes that were scattered across adapters (absolute directory headers rewritten to
    repo-relative, the F/R/A/S legend the meta volume itself does not state) happen once, here, and
    are recorded.

What goes in, in this order:
  1. a LEGEND — one line stating the entry grammar (name[Tag]: F:function | R:relations | A:interfaces
     | S:constraints). This is the format half of the injection preamble the arms used before; it is
     included because the meta volume defines the tag dictionary but never says what F/R/A/S mean,
     while the competitor's map returns self-describing field names. It is a legend, not a manual:
     no usage advice, no examples beyond the volumes' own. The behavioural sentence ("it is a map,
     use it to decide which files to read…") stays in each arm's same-shape hint, NOT here.
  2. the three volumes verbatim (aoci.txt, aoci.meta.txt, aoci.code.txt), each under a
     `===== <name> =====` separator, exactly as ask.load_index() has always concatenated them;
  3. with every absolute directory header of the authoring machine rewritten to repo-relative
     (`===/abs/repo/xarray/util/===` -> `===xarray/util/===`) and every other occurrence of the
     authoring root removed. Pasted verbatim, absolute headers send an agent out of its own worktree
     (pilot-10: 142 contamination events over 144 questions; pilot-11 after the rewrite: 0).

Self-checks (the build refuses to write on failure):
  - entry count in the artefact == entry count in aoci.code.txt
  - zero occurrences of the authoring root (and of "/home/") remain
  - the volumes' sha256 are recorded; `--check` re-verifies an existing artefact against the volumes
    on disk and exits non-zero if they drifted (call it before every run).

Usage:
  build_index_artifact.py --repo-root repos/xarray --out artifacts/xarray/aoci-index.txt
  build_index_artifact.py --repo-root repos/xarray --out artifacts/xarray/aoci-index.txt --check
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

VOLUME_FILES = ("aoci.txt", "aoci.meta.txt", "aoci.code.txt")
ENTRY_RE = re.compile(r"^\S+\[[A-Z]{1,3}\d[A-Z]?\]: ")

# The legend is the FORMAT half of the preamble every injection arm used before (openhands_ask.INJECT_PREAMBLE,
# agent_ask.INDEX_PREAMBLE_AGENT), verbatim. Do not add usage guidance here — see the module docstring.
LEGEND = ("AOCI cognition index of this repository: one line per file — "
          "name[Tag]: F:function | R:relations | A:interfaces | S:constraints.\n\n")


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_volumes(repo_root):
    vols = {}
    for name in VOLUME_FILES:
        p = os.path.join(repo_root, name)
        if not os.path.exists(p):
            sys.exit(f"missing volume {p}")
        raw = open(p, encoding="utf-8").read()
        vols[name] = {"text": raw, "sha256": sha256_text(raw), "bytes": len(raw.encode("utf-8"))}
    return vols


def relativize(text, repo_root):
    real = os.path.realpath(repo_root)
    text = re.sub(r"^===" + re.escape(real) + r"/?", "===", text, flags=re.M)
    text = text.replace(real + "/", "").replace(repo_root.rstrip("/") + "/", "")
    return text, real


def build(repo_root):
    vols = load_volumes(repo_root)
    body = "\n".join(f"===== {name} =====\n" + vols[name]["text"] for name in VOLUME_FILES)
    body, real = relativize(body, repo_root)
    text = LEGEND + body
    # self-checks
    n_src = sum(1 for l in vols["aoci.code.txt"]["text"].splitlines() if ENTRY_RE.match(l))
    n_art = sum(1 for l in text.splitlines() if ENTRY_RE.match(l))
    problems = []
    if n_src != n_art:
        problems.append(f"entry count drifted: volume {n_src} vs artefact {n_art}")
    if real in text:
        problems.append(f"authoring root still present ({text.count(real)}x)")
    if "/home/" in text:
        problems.append(f"'/home/' still present ({text.count('/home/')}x)")
    if problems:
        sys.exit("refusing to write artefact:\n  " + "\n  ".join(problems))
    meta = {
        "artifact_sha256": sha256_text(text),
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "approx_tokens": len(text) // 3,
        "entries": n_art,
        "legend_chars": len(LEGEND),
        "legend_sha256": sha256_text(LEGEND),
        "volumes": {n: {"sha256": vols[n]["sha256"], "bytes": vols[n]["bytes"]} for n in VOLUME_FILES},
        "repo_commit": common.git_sha(repo_root, short=False, untracked=False),
        "authoring_root_rewritten": real,
        "harness_sha": common.harness_sha(),
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return text, meta


def check(repo_root, out):
    """Re-derive the artefact from the volumes on disk and compare with what is stored."""
    text, meta = build(repo_root)
    side = out[:-4] + ".json" if out.endswith(".txt") else out + ".json"
    if not (os.path.exists(out) and os.path.exists(side)):
        sys.exit(f"artefact or sidecar missing ({out}, {side})")
    stored = json.load(open(side))
    on_disk = open(out, encoding="utf-8").read()
    bad = []
    if sha256_text(on_disk) != stored["artifact_sha256"]:
        bad.append("artefact file no longer matches its own sidecar sha256")
    if meta["artifact_sha256"] != stored["artifact_sha256"]:
        bad.append("volumes on disk no longer produce the stored artefact (index changed since build)")
    for n in VOLUME_FILES:
        if meta["volumes"][n]["sha256"] != stored["volumes"][n]["sha256"]:
            bad.append(f"{n} sha256 drifted")
    if bad:
        sys.exit("STALE ARTEFACT — rebuild before running:\n  " + "\n  ".join(bad))
    print(f"artefact OK  sha256={stored['artifact_sha256'][:16]}  entries={stored['entries']}  chars={stored['chars']:,}")


def external(path):
    """Fingerprint an externally-authored index file (e.g. a maintainer's hand-written index) and write
    the sidecar common.load_artifact() requires, so it can be injected per-arm with the same delivery
    assertions as a built artefact. The file itself is NOT modified — what you ship is what was written."""
    text = open(path, encoding="utf-8").read()
    meta = {
        "artifact_sha256": sha256_text(text),
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "approx_tokens": len(text) // 3,
        "entries": sum(1 for l in text.splitlines() if ENTRY_RE.match(l)),  # best effort on foreign formats
        "volumes": {"external": {"source": os.path.basename(path), "sha256": sha256_text(text),
                                 "bytes": len(text.encode("utf-8"))}},
        "external": True,
        "repo_commit": None,
        "harness_sha": common.harness_sha(),
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    side = path[:-4] + ".json" if path.endswith(".txt") else path + ".json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"external artefact fingerprinted  {path}  ({meta['chars']:,} chars ≈ {meta['approx_tokens']:,} tok, "
          f"{meta['entries']} entry-like lines)")
    print(f"      sha256 {meta['artifact_sha256']}")
    print(f"      sidecar {side}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", help="authoring tree with the AOCI volumes (build/--check modes)")
    ap.add_argument("--out", help="artefact path (.txt); sidecar written next to it as .json")
    ap.add_argument("--check", action="store_true", help="verify an existing artefact against the volumes on disk")
    ap.add_argument("--external", default=None, metavar="FILE",
                    help="fingerprint an existing externally-authored index file and write only its sidecar")
    a = ap.parse_args()
    if a.external:
        external(a.external)
        return
    if not (a.repo_root and a.out):
        ap.error("--repo-root and --out are required unless --external is used")
    if a.check:
        check(a.repo_root, a.out)
        return
    text, meta = build(a.repo_root)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    side = a.out[:-4] + ".json" if a.out.endswith(".txt") else a.out + ".json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}  ({meta['chars']:,} chars ≈ {meta['approx_tokens']:,} tok, {meta['entries']} entries)")
    print(f"      sha256 {meta['artifact_sha256']}")
    print(f"      volumes: " + ", ".join(f"{n} {meta['volumes'][n]['sha256'][:12]}" for n in VOLUME_FILES))
    print(f"      sidecar {side}")


if __name__ == "__main__":
    main()
