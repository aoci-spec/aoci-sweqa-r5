#!/usr/bin/env python3
"""Acceptance gate for a run's ANSWERS — everything that must hold before a single judge token is spent.

  accept.py runs/<id>                 # human table + verdict, exit 1 if any hard check fails
  accept.py runs/<id> --json out.json # machine-readable
  accept.py runs/<id> --arms codex-1,codex-2

Hard checks (a failure blocks judging):
  complete      every pass has exactly the expected questions, each answered once, none missing
  no_errors     no record carries an error
  no_leak       no command names the harness tree, the run directory, the authoring ledger, the
                index artefact or the agent's memory (the codex read-permissive sandbox let four
                conversations do exactly this, 2026-08-21 — memo 9.58)
  no_net_success  no network command that actually returned data (attempts blocked by the jail are fine)
  delivery      every inject record's index sha256 + chars match the artefact and the delivered
                prompt carried at least the whole index
  clean_tree    no record left its workspace modified
  same_model    one profile/model per pass, matching the manifest

Soft checks (reported, never blocking): probes (outside-workspace reads the jail turns into ENOENT),
network attempts, forced/hit-cap answers, refusal or garble re-runs, degenerate answers, zero-source
answers. These are experiment characteristics, not defects, and the result card cites them.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

LEAK_RE = re.compile(r"/aoci-eval/(runs|authoring|artifacts|harness|benchmarks|specs|config|tests)"
                     r"|answers-[\w.-]+\.jsonl|scores-[\w.-]+\.jsonl|manifest\.json"
                     r"|/aoci-code/|/\.claude/|MEMORY\.md|aoci-index\.txt")
NET_RE = re.compile(r"\b(curl|wget|pip3?\s+install|apt(-get)?\s+install|git\s+(clone|fetch|pull)|nc|ssh|scp)\b")


def cmd_of(t):
    """Both adapters record a command; OpenHands puts it under args/full, codex under cmd."""
    if not isinstance(t, dict):
        return ""
    if t.get("cmd"):
        return str(t["cmd"])
    a = t.get("args") or {}
    if isinstance(a, dict):
        return str(a.get("command") or a.get("data") or "")
    return str(a)


def check_pass(name, path, expected_qs, art, spec_arm, in_flight=False):
    r = {"pass": name, "n": 0, "errors": 0, "missing": 0, "extra": 0, "leaks": [], "net_attempts": 0,
         "net_success": [], "probes": 0, "delivery_bad": 0, "delivery_ok": 0, "dirty": 0, "forced": 0,
         "retried": 0, "degenerate": 0, "garbled": 0, "zero_source": 0, "dirty_benign": 0, "models": set(), "profiles": set()}
    recs = common.load_records(path, common.valid_answer)
    r["n"] = len(recs)
    seen = set()
    for q, rec in recs.items():
        seen.add(common.qid(q))
        if rec.get("error"):
            r["errors"] += 1
        if rec.get("model"):
            r["models"].add(rec["model"])
        if rec.get("profile"):
            r["profiles"].add(rec["profile"])
        if rec.get("workspace_dirty"):
            # A python import leaves __pycache__ behind: a side effect of READING the code, not an
            # edit. Only real changes are a hard failure (2026-08-21: 3 such records, all the same
            # question, all `python3 -c "from sqlfluff... import"`).
            paths = rec.get("dirty_paths")
            benign = all(("__pycache__" in p_ or p_.endswith((".pyc", ".pyo"))) for p_ in paths) if paths else None
            if not benign:
                # no path list (older records), or paths that are not pycache: decide by what the agent actually ran.
                # r5 (2026-10-16): file_editor create / str_replace / insert are writes too, not only shell redirections.
                cmds = " ".join(cmd_of(t) for t in (rec.get("trace") or []))
                cmds = re.sub(r"2?>>?\s*/dev/null|2>&1", " ", cmds)   # discarding output is not a write
                benign = not re.search(r">>?\s*\S|sed\s+-i|\btee\b|\brm\b|\bmv\b|\bcp\b|\btouch\b|"
                                       r"\bmkdir\b|\bpatch\b|apply_patch|git\s+(checkout|apply|restore)|"
                                       r"^(create|str_replace|insert|undo_edit)\b|\s(create|str_replace|insert|undo_edit)\s", cmds)
            r["dirty_benign" if benign else "dirty"] += 1
        if rec.get("hit_cap") or str(rec.get("stop_reason", "")).startswith("forced"):
            r["forced"] += 1
        if rec.get("retried"):
            r["retried"] += 1
        if not rec.get("files_read"):
            r["zero_source"] += 1
        if common.degenerate_reason(rec.get("answer") or ""):
            r["degenerate"] += 1
        if common.refusal_reason(rec.get("answer") or ""):
            r["garbled"] += 1   # special-token debris / refusal text that survived the fresh-rerun guard
        # commands: leaks and network, from the trace (authoritative) and the contamination hits
        for t in (rec.get("trace") or []):
            c = cmd_of(t)
            if not c:
                continue
            if LEAK_RE.search(c):
                r["leaks"].append({"q": q[:60], "cmd": c[:160], "chars": t.get("chars")})
            if NET_RE.search(c):
                r["net_attempts"] += 1
                # a network command that returned data is contamination; a blocked one is not
                if t.get("exit") == 0 and (t.get("chars") or 0) > 40:
                    r["net_success"].append({"q": q[:60], "cmd": c[:160], "chars": t.get("chars")})
        for h in (rec.get("contamination") or []):
            if not isinstance(h, dict):
                continue
            why = h.get("why") or []
            c = h.get("cmd") or ""
            if "network" in why and not NET_RE.search(c):
                r["net_attempts"] += 1
            if LEAK_RE.search(c) and not any(l["cmd"][:80] == c[:80] for l in r["leaks"]):
                r["leaks"].append({"q": q[:60], "cmd": c[:160], "chars": None})
            if any(w in ("outside-workspace", "parent-escape", "map-artefact", "env-probe") for w in why):
                r["probes"] += 1
        # index delivery for inject arms
        if rec.get("index_sha256") or rec.get("index_chars"):
            ok = True
            if art.get("sha256") and rec.get("index_sha256") and rec["index_sha256"] != art["sha256"]:
                ok = False
            if art.get("chars") and rec.get("index_chars") and rec["index_chars"] != art["chars"]:
                ok = False
            if rec.get("prompt_chars_delivered") is not None and rec.get("index_chars") \
                    and rec["prompt_chars_delivered"] < rec["index_chars"]:
                ok = False
            r["delivery_ok" if ok else "delivery_bad"] += 1
    want = {common.qid(q) for q in expected_qs}
    r["missing"] = len(want - seen)
    r["extra"] = len(seen - want)
    r["models"] = sorted(r["models"])
    r["profiles"] = sorted(r["profiles"])
    # in-flight: an unfinished pass legitimately misses questions — everything else must already hold
    miss_gate = 0 if in_flight else r["missing"]
    r["hard_fail"] = bool(r["errors"] or miss_gate or r["extra"] or r["leaks"] or r["net_success"]
                          or r["delivery_bad"] or r["dirty"] or len(r["models"]) > 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--arms", default="")
    ap.add_argument("--json", default=None)
    ap.add_argument("--in-flight", action="store_true",
                    help="mid-run integrity check: passes with 0 records are skipped and 'missing' does not gate "
                         "(an unfinished pass legitimately misses questions); every other hard check applies as usual")
    a = ap.parse_args()
    run_dir = a.run.rstrip("/")
    man = common.read_manifest(run_dir) or {}
    spec = man.get("spec") or {}
    qs = [q["question"] for q in common.load_questions(spec["questions"], spec.get("limit") or 0)]
    art = man.get("index_artifact") or {}
    only = {x for x in a.arms.split(",") if x}
    names = sorted(f[len("answers-"):-len(".jsonl")] for f in os.listdir(run_dir)
                   if f.startswith("answers-") and f.endswith(".jsonl"))
    if only:
        names = [n for n in names if n in only]

    arms_spec = spec.get("arms") or []

    def arm_for(pass_name):
        # longest matching arm name wins: "agent-inject-hand-2" must resolve to arm
        # "agent-inject-hand", not its prefix sibling "agent-inject"
        cands = [x for x in arms_spec if pass_name == x["name"] or pass_name.startswith(x["name"] + "-")]
        return max(cands, key=lambda x: len(x["name"])) if cands else {}

    _art_cache = {}

    def art_for(spec_arm):
        # an arm may inject its own artefact (per-arm index_artifact, e.g. a hand-written index);
        # its records must be asserted against THAT artefact's sidecar, not the manifest-level one
        p = (spec_arm or {}).get("index_artifact")
        if not p or (art and p == art.get("path")):
            return art
        if p not in _art_cache:
            side = p[:-4] + ".json" if p.endswith(".txt") else p + ".json"
            try:
                m = json.load(open(side))
                _art_cache[p] = {"path": p, "sha256": m.get("artifact_sha256"), "chars": m.get("chars")}
            except OSError:
                print(f"WARN: per-arm artefact sidecar unreadable, delivery unasserted: {side}")
                _art_cache[p] = {"path": p, "sha256": None, "chars": None}
        return _art_cache[p]

    rows = []
    for n in names:
        sa = arm_for(n)
        row = check_pass(n, os.path.join(run_dir, f"answers-{n}.jsonl"), qs, art_for(sa) or {}, sa, in_flight=a.in_flight)
        if a.in_flight and row["n"] == 0:
            continue   # queued arm, nothing to check yet
        rows.append(row)
    for p, v in sorted((v["path"], v) for v in _art_cache.values()):
        print(f"per-arm artefact  {p}  sha {(v['sha256'] or 'UNRESOLVED')[:12]}")
    if a.in_flight:
        print("IN-FLIGHT mode: missing does not gate; empty passes skipped")
    hard = [r for r in rows if r["hard_fail"]]
    print(f"acceptance gate · {run_dir} · {len(rows)} passes × {len(qs)} questions · artefact "
          f"{(art.get('sha256') or '—')[:12]} {art.get('chars') or ''}")
    print(f"{'pass':26}{'n':>4}{'err':>5}{'miss':>5}{'leak':>5}{'net':>4}{'net!':>5}{'deliv':>8}"
          f"{'dirty':>6}{'probe':>6}{'forced':>7}{'retry':>6}{'degen':>6}{'garb':>6}{'0src':>5}")
    for r in rows:
        deliv = f"{r['delivery_ok']}/{r['delivery_ok'] + r['delivery_bad']}" if (r["delivery_ok"] or r["delivery_bad"]) else "—"
        flag = "✗" if r["hard_fail"] else " "
        print(f"{flag}{r['pass']:25}{r['n']:4}{r['errors']:5}{r['missing']:5}{len(r['leaks']):5}"
              f"{r['net_attempts']:4}{len(r['net_success']):5}{deliv:>8}{r['dirty']:6}{r['probes']:6}"
              f"{r['forced']:7}{r['retried']:6}{r['degenerate']:6}{r['garbled']:6}{r['zero_source']:5}")
    tot = {k: sum(len(r[k]) if isinstance(r[k], list) else r[k] for r in rows)
           for k in ("n", "errors", "missing", "extra", "leaks", "net_attempts", "net_success",
                     "delivery_bad", "dirty", "dirty_benign", "probes", "forced", "retried", "degenerate",
                     "garbled", "zero_source")}
    print(f"\ntotals: {tot['n']} answers · {tot['errors']} errors · {tot['missing']} missing · "
          f"{tot['leaks']} leaks · {tot['net_attempts']} network attempts ({tot['net_success']} returned data) · "
          f"{tot['delivery_bad']} delivery failures · {tot['dirty']} dirty workspaces")
    print(f"soft: {tot['probes']} probes (jail: ENOENT) · {tot['dirty_benign']} __pycache__-only workspaces · {tot['forced']} forced answers · "
          f"{tot['retried']} refusal/garble re-runs · {tot['degenerate']} degenerate · {tot['garbled']} garbled-final · {tot['zero_source']} zero-source")
    for r in hard:
        for l in r["leaks"][:3]:
            print(f"  LEAK  [{r['pass']}] {l['cmd'][:120]}")
        for l in r["net_success"][:3]:
            print(f"  NET   [{r['pass']}] {l['cmd'][:120]} → {l['chars']} chars")
    if a.json:
        json.dump({"run": run_dir, "passes": rows, "totals": tot, "ok": not hard},
                  open(a.json, "w"), indent=1, default=list)
    if hard:
        print(f"\nREFUSING: {len(hard)} pass(es) fail a hard check — judging would spend money on tainted data.")
        sys.exit(1)
    print("\nACCEPTED: every hard check passes; judging may proceed.")


if __name__ == "__main__":
    main()
