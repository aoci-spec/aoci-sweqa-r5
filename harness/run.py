#!/usr/bin/env python3
"""Run a whole experiment from a spec file: preflight → arms → judge → analyze.

A spec is JSON (see specs/*.json):
{
  "run": "pilot-3",                       # → runs/pilot-3/
  "repo": "xarray", "repo_root": "repos/xarray",
  "questions": "benchmarks/swe-qa/xarray.jsonl",
  "limit": 0, "workers": 7,               # per-stage default (1..20)
  "arms": [
    {"name": "direct",       "kind": "direct",       "profile": "xingyun-sonnet5"},
    {"name": "inject",       "kind": "inject",       "profile": "xingyun-sonnet5", "preamble": "v2"},
    {"name": "agent",        "kind": "agent",        "profile": "xingyun-sonnet5", "max_steps": 8},
    {"name": "agent-inject", "kind": "agent-inject", "profile": "xingyun-sonnet5", "max_steps": 8}
  ],
  "judge": {"profile": "xingyun-sonnet4.5", "votes": 5, "workers": 7},
  "reuse": {"direct": "runs/pilot-3-xarray"},   # optional: copy answers/scores of an arm from an earlier run
  "max_usd_per_arm": 40,                        # optional: running-estimate cap passed to every arm
  "require_complete_index": true,
  "index_artifact": "artifacts/xarray/aoci-index.txt",   # built by build_index_artifact.py; --check'd in preflight,
                                                        # copied into the run dir, passed to every *-inject arm
  "draws": 10                                    # optional: expand every arm into name-1..name-k and run them
                                                 # INTERLEAVED BY DRAW (draw 1 of every arm, then draw 2, ...)
}
Arm kinds map to scripts: direct/inject → ask.py, agent/agent-inject → agent_ask.py,
oh/oh-mcp/oh-inject → openhands_ask.py (run with the .venv-oh interpreter, OPENHANDS_PY to override).
Extra keys on an arm are passed through as --key value (max_tokens, temperature,
preamble, max_steps, workers, ...). Every stage is resumable: re-running the same
spec only does what is missing. The run directory gets manifest.json (spec, repo
commit, index fingerprint, per-arm/judge parameters), copies of the index volumes,
and analysis.txt/analysis.json at the end.

Usage:
  run.py specs/pilot-3.json [--only-arms agent,agent-inject] [--skip-judge] [--skip-analyze] [--dry-run] [--allow-dirty]
run.py refuses to run with uncommitted tracked harness changes (reproducibility) unless --allow-dirty;
everything printed also goes to runs/<id>/run.log.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import llm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
AOCI_BIN = os.environ.get("AOCI_BIN", "aoci")
KIND_SCRIPT = {"direct": "ask.py", "inject": "ask.py", "agent": "agent_ask.py", "agent-inject": "agent_ask.py",
               "agent-cg": "agent_ask.py", "agent-thin": "agent_ask.py", "agent-inject-cg": "agent_ask.py",
               "oh": "openhands_ask.py", "oh-aoci": "openhands_ask.py", "oh-mcp": "openhands_ask.py",
               "oh-cg": "openhands_ask.py", "oh-inject": "openhands_ask.py", "oh-file": "openhands_ask.py",
               "oh-inject-cg": "openhands_ask.py", "oh-aoci-cg": "openhands_ask.py",
               "codex": "codex_ask.py", "codex-inject": "codex_ask.py",
               "codex-cg": "codex_ask.py", "codex-inject-cg": "codex_ask.py",
               "loc-direct": "locate.py", "loc-inject": "locate.py",
               "loc-agent": "agent_locate.py", "loc-agent-inject": "agent_locate.py", "loc-agent-cg": "agent_locate.py",
               "loc-agent-inject-cg": "agent_locate.py"}
NEEDS_INDEX = {"inject", "agent-inject", "oh-aoci", "oh-mcp", "oh-inject", "agent-thin", "oh-file", "agent-inject-cg", "oh-inject-cg", "oh-aoci-cg", "codex-inject", "codex-inject-cg"}
NEEDS_REPO = {"agent", "agent-inject", "agent-cg", "agent-thin", "agent-inject-cg", "oh", "oh-aoci", "oh-mcp", "oh-cg", "oh-inject", "oh-file", "oh-inject-cg", "oh-aoci-cg", "codex", "codex-inject", "codex-cg", "codex-inject-cg",
              "loc-agent", "loc-agent-inject", "loc-agent-cg", "loc-agent-inject-cg"}
OPENHANDS_PY = os.environ.get("OPENHANDS_PY", os.path.join(os.path.dirname(HERE), ".venv-oh", "bin", "python3"))


def interpreter_for(kind):
    if kind.startswith("oh"):
        if not os.path.exists(OPENHANDS_PY):
            sys.exit(f"OpenHands arms need the separate venv interpreter {OPENHANDS_PY} (see requirements-openhands.txt) or OPENHANDS_PY=...")
        return OPENHANDS_PY
    return PY


LOG = None


def out(line):
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


def sh(cmd, dry):
    out("+ " + " ".join(cmd))
    if dry:
        return 0
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        out(line.rstrip("\n"))
    return p.wait()


def sh_buffered(cmd, tag):
    """Run one pass, buffer its output, print it as a block on completion (concurrent passes would
    interleave line-by-line otherwise). Live progress is on the dashboard via progress-/answers- files."""
    out(f"+ [{tag}] " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out(f"=== [{tag}] rc={p.returncode} ===")
    for line in (p.stdout or "").splitlines():
        out(f"[{tag}] {line}")
    return p.returncode


SPEC_KEYS = {"_comment", "run", "repo", "repo_root", "questions", "limit", "workers", "arms", "judge", "pairs", "baseline", "task",
             "require_complete_index", "reuse", "max_usd_per_arm", "index_artifact", "draws", "concurrent_passes"}
ARM_KEYS = {"name", "kind", "profile", "workers", "preamble", "output", "max_steps", "min_steps", "max_tokens", "temperature", "effort", "max_usd",
            "max_iterations", "index_artifact", "reading_policy", "read_lines", "max_output_tokens", "thin_k", "fields"}
INJECT_KINDS = {"agent-inject", "oh-inject", "agent-thin", "agent-inject-cg", "oh-inject-cg", "codex-inject", "codex-inject-cg",
                "loc-inject", "loc-agent-inject", "loc-agent-inject-cg"}   # kinds that consume the built index artefact


def validate_spec(spec):
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        sys.exit(f"spec: unknown top-level keys {sorted(unknown)} (known: {sorted(SPEC_KEYS)})")
    for k in ("run", "repo", "questions", "arms"):
        if k not in spec:
            sys.exit(f"spec: missing required key {k!r}")
    names = set()
    for x in spec["arms"]:
        if x.get("kind") not in KIND_SCRIPT:
            sys.exit(f"spec: arm {x.get('name')!r} has unknown kind {x.get('kind')!r} (known: {sorted(KIND_SCRIPT)})")
        bad = set(x) - ARM_KEYS
        if bad:
            sys.exit(f"spec: arm {x.get('name')!r} has unknown keys {sorted(bad)} (known: {sorted(ARM_KEYS)})")
        if x["name"] in names:
            sys.exit(f"spec: duplicate arm name {x['name']!r}")
        names.add(x["name"])
    if not os.path.exists(spec["questions"]):
        sys.exit(f"spec: questions file not found: {spec['questions']}")


def guide_complete(repo_root):
    """True/False from `aoci index agent guide --json`, or None if the binary is unavailable."""
    if not os.path.exists(AOCI_BIN):
        return None
    try:
        out = subprocess.run([AOCI_BIN, "--repo", repo_root, "index", "agent", "guide", "--agent", "claude-code", "--json"],
                             capture_output=True, text=True, timeout=120).stdout
        return bool(json.loads(out).get("complete"))
    except Exception:  # noqa: BLE001
        return None


def preflight(profiles, dry):
    for name in sorted(profiles):
        p = llm.load_profile(name)
        if dry:
            print(f"preflight (dry) profile={name} model={p['model']} key={'set' if p['_has_key'] else 'MISSING'}")
            continue
        if not p["_has_key"]:
            sys.exit(f"profile {name}: key missing ({p['api_key_env']})")
        try:
            r = llm.complete(p, [{"type": "text", "text": "You are a terse assistant."}], "Reply with exactly: OK", max_tokens=8)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"preflight failed for profile {name}: {e.__class__.__name__}: {str(e)[:300]}")
        print(f"preflight ok profile={name} model={p['model']} protocol={r['protocol']}", flush=True)



def rejudge(spec, a):
    """Re-judge existing answers under an alternate rubric (judge-robustness experiments).

    Writes scores into <run_dir>/<out_subdir>/ and leaves the original scores untouched; the
    subdirectory gets a manifest and answer symlinks so dashboard.py lists it like any other run.
    """
    sub = spec.get("out_subdir") or "rejudge"
    rubric = spec["rubric_file"]
    j = spec.get("judge", {})
    draws = spec.get("runs_subset") or []
    jobs, made = [], []
    for run_dir, arms in spec["scope"].items():
        odir = os.path.join(run_dir, sub)
        for arm in arms:
            for d in draws:
                ans = os.path.join(run_dir, f"answers-{arm}-{d}.jsonl")
                if not os.path.exists(ans):
                    sys.exit(f"missing answers: {ans}")
                cmd = [PY, os.path.join(HERE, "judge.py"), "--answers", ans,
                       "--out", os.path.join(odir, f"scores-{arm}-{d}.jsonl"),
                       "--votes", str(j.get("votes", 1)), "--workers", str(j.get("workers", 48)),
                       "--rubric-file", rubric]
                if j.get("profile"):
                    cmd += ["--profile", j["profile"]]
                jobs.append((f"{os.path.basename(run_dir)}/{arm}-{d}", cmd))
                made.append((odir, run_dir, arm, d))
    out(f"rejudge: {len(jobs)} files, rubric={rubric} sha={common.sha256_file(rubric)[:16]} "
        f"profile={j.get('profile')} votes={j.get('votes')}")
    if a.dry_run:
        for name, cmd in jobs:
            out("+ " + " ".join(cmd))
        return
    # dashboard.py discovers runs by manifest.json and reads answers-/scores- pairs: give the
    # subdirectory both so the one dashboard shows re-judged scores with no dashboard changes.
    for odir, run_dir, arm, d in made:
        os.makedirs(odir, exist_ok=True)
        link = os.path.join(odir, f"answers-{arm}-{d}.jsonl")
        if not os.path.exists(link):
            os.symlink(os.path.relpath(os.path.join(run_dir, f"answers-{arm}-{d}.jsonl"), odir), link)
    for odir in sorted({x[0] for x in made}):
        base = {}
        src = os.path.join(os.path.dirname(odir), "manifest.json")
        if os.path.exists(src):
            base = json.load(open(src, encoding="utf-8"))
        scope_arms = spec["scope"][os.path.dirname(odir)]
        if isinstance(base.get("arms"), dict):    # keep only the arms this re-judge covers
            base["arms"] = {k: v for k, v in base["arms"].items() if k in scope_arms}
        # arm_passes drive the panel grid: restrict the parent's to this re-judge's coverage
        # (synthesize when the parent predates arm_passes). The parent's "spec" dict must stay —
        # the panel takes expected question counts and judge presence from it.
        pp = base.get("arm_passes") or []
        pp = [x for x in pp if (x.get("base") or x.get("name")) in scope_arms and int(x.get("draw") or 1) in draws]
        if not pp:
            pp = [{"name": f"{arm}-{d}", "base": arm, "draw": d, "kind": arm, "order": i}
                  for i, (arm, d) in enumerate((arm, d) for arm in scope_arms for d in draws)]
        base["arm_passes"] = pp
        base.update({"draws": len(draws),          # the panel sizes its grid from these
                     "run": f"{os.path.basename(os.path.dirname(odir))}-{sub}", "rejudge_of": os.path.dirname(odir),
                     "rubric_file": rubric, "rubric_sha256": common.sha256_file(rubric),
                     "judge": j, "rejudge_spec": os.path.basename(a.spec), "harness": common.harness_sha(),
                     "started": common.now_iso()})
        json.dump(base, open(os.path.join(odir, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pool = int(spec.get("concurrent_files", 4))
    from concurrent.futures import ThreadPoolExecutor

    def _job(name, cmd):
        odir = os.path.dirname(cmd[cmd.index("--out") + 1])
        with open(os.path.join(odir, "run.log"), "a", encoding="utf-8") as lf:
            lf.write("+ " + " ".join(cmd) + "\n")
        rc = sh_buffered(cmd, name)
        with open(os.path.join(odir, "run.log"), "a", encoding="utf-8") as lf:
            lf.write(f"=== [{name}] rc={rc} ===\n")
        return rc

    failed = []
    with ThreadPoolExecutor(max_workers=pool) as ex:
        futs = {ex.submit(_job, name, cmd): name for name, cmd in jobs}
        for f, name in list(futs.items()):
            if f.result() != 0:
                failed.append(name)
    if failed:
        sys.exit(f"rejudge failed: {failed}")
    out(f"rejudge done: {len(jobs)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--only-arms", default="", help="comma list of arm names to run (default all)")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--skip-accept", action="store_true", help="judge even if the acceptance gate fails (disclose it)")
    ap.add_argument("--skip-analyze", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, call nothing")
    ap.add_argument("--allow-dirty", action="store_true", help="run even if the harness has uncommitted tracked changes")
    a = ap.parse_args()
    global LOG
    spec = json.load(open(a.spec, encoding="utf-8"))
    os.chdir(common.ROOT)
    if spec.get("rubric_file"):          # re-judge spec: no answering, no acceptance gate
        LOG = None if a.dry_run else open(os.path.join("runs", f"{spec.get('run', 'rejudge')}.log"), "a", encoding="utf-8")
        out(f"=== run.py --rejudge {a.spec} at {common.now_iso()} harness={common.harness_sha()}")
        rejudge(spec, a)
        return
    validate_spec(spec)
    run_dir = os.path.join("runs", spec["run"])
    os.makedirs(run_dir, exist_ok=True)
    hsha = common.harness_sha()
    if hsha and hsha.endswith("-dirty") and not a.dry_run:
        if not a.allow_dirty:
            sys.exit(f"refusing: harness code has uncommitted changes ({hsha}); commit first or pass --allow-dirty")
        print(f"WARNING: harness dirty ({hsha}); manifest will say so", flush=True)
    if not a.dry_run:
        LOG = open(os.path.join(run_dir, "run.log"), "a", encoding="utf-8")
        out(f"=== run.py {a.spec} at {common.now_iso()} harness={hsha}")
    # reuse arms from an earlier run (same questions/harness): copy answers/scores if not present
    for arm, src_run in (spec.get("reuse") or {}).items():
        for kind in ("answers", "scores"):
            src = os.path.join(src_run, f"{kind}-{arm}.jsonl")
            dst = os.path.join(run_dir, f"{kind}-{arm}.jsonl")
            if os.path.exists(src) and not os.path.exists(dst) and not a.dry_run:
                shutil.copy(src, dst)
                out(f"reused {src} -> {dst}")
        if not a.dry_run:
            sm = common.read_manifest(src_run)
            common.write_manifest(run_dir, {"reused": {arm: {"from": src_run, "arm_manifest": (sm.get("arms") or {}).get(arm),
                                                              "judge_manifest": (sm.get("judge") or {}).get(arm)}}})
    workers = common.clamp_workers(spec.get("workers", common.DEFAULT_WORKERS))
    only = {x for x in a.only_arms.split(",") if x}
    reused = set((spec.get("reuse") or {}).keys())
    arms = [x for x in spec["arms"] if (not only or x["name"] in only) and x["name"] not in reused]
    repo_root = spec.get("repo_root")
    # draws: expand each arm into k independent draws and INTERLEAVE them by draw so that no arm is blocked
    # in time (PROTOCOL §3.3: `oh-a → oh-b → oh-cg` produced contrasts monotone in slot distance).
    draws = int(spec.get("draws") or 1)
    if draws > 1:
        expanded = []
        for d in range(1, draws + 1):
            for x in arms:
                y = dict(x)
                y["name"] = f"{x['name']}-{d}"
                y["_base"] = x["name"]
                y["_draw"] = d
                expanded.append(y)
        arms = expanded
        out(f"draws={draws}: {len(arms)} arm-passes, interleaved by draw") if not a.dry_run else print(f"draws={draws}: {len(arms)} arm-passes")
    # index artefact: one file, one sha, every inject arm; refuse if it drifted from the volumes on disk
    artifact = spec.get("index_artifact")
    if artifact:
        if not os.path.exists(artifact):
            sys.exit(f"index_artifact not found: {artifact} (build it with harness/build_index_artifact.py)")
        if repo_root and not a.dry_run:
            rc = subprocess.run([PY, os.path.join(HERE, "build_index_artifact.py"), "--repo-root", repo_root,
                                 "--out", artifact, "--check"], capture_output=True, text=True)
            out(rc.stdout.strip() + rc.stderr.strip())
            if rc.returncode != 0:
                sys.exit("refusing: index artefact is stale relative to the volumes on disk")
            shutil.copy(artifact, os.path.join(run_dir, os.path.basename(artifact)))
            side = artifact[:-4] + ".json"
            if os.path.exists(side):
                shutil.copy(side, os.path.join(run_dir, os.path.basename(side)))
        for x in arms:
            if x["kind"] in INJECT_KINDS and "index_artifact" not in x:
                x["index_artifact"] = artifact
    elif any(x["kind"] in INJECT_KINDS for x in arms):
        sys.exit("spec: *-inject arms need a top-level index_artifact (or per-arm index_artifact)")
    needs_index = any(x["kind"] in NEEDS_INDEX for x in arms)

    if needs_index:
        if not repo_root:
            sys.exit("spec needs repo_root for index arms")
        fp = common.index_fingerprint(repo_root)
        gc = guide_complete(repo_root)
        print(f"index: entries={fp['entries']} guide_complete={gc} commit={str(fp['repo_commit'])[:12]}", flush=True)
        if spec.get("require_complete_index", True) and gc is False:
            sys.exit("refusing: index not complete (Guide complete=False). Finish authoring or set require_complete_index=false for a smoke test.")
        if fp["entries"] == 0:
            sys.exit(f"refusing: no entries in {repo_root}/aoci.code.txt")
        if not a.dry_run:
            for name in ("aoci.txt", "aoci.meta.txt", "aoci.code.txt"):
                src = os.path.join(repo_root, name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(run_dir, name))
    else:
        fp = common.index_fingerprint(repo_root) if repo_root and os.path.isdir(repo_root) else None

    profiles = {x.get("profile") or llm._read().get("default") for x in arms}
    judge = spec.get("judge") or {}
    if judge and not a.skip_judge:
        profiles.add(judge.get("profile") or llm._read().get("default"))
    preflight({p for p in profiles if p}, a.dry_run)

    if not a.dry_run:
        art_meta = json.load(open(artifact[:-4] + ".json")) if artifact and os.path.exists(artifact[:-4] + ".json") else None
        common.write_manifest(run_dir, {"spec_file": a.spec, "spec": spec, "repo": spec.get("repo"),
                                        "repo_commit": fp["repo_commit"] if fp else None, "index": fp,
                                        "index_artifact": {"path": artifact, "sha256": art_meta.get("artifact_sha256"),
                                                           "chars": art_meta.get("chars"), "entries": art_meta.get("entries")} if art_meta else None,
                                        "draws": draws,
                                        "arm_passes": [{"name": x["name"], "base": x.get("_base", x["name"]), "draw": x.get("_draw", 1),
                                                        "kind": x["kind"], "order": i} for i, x in enumerate(arms)],
                                        "questions_sha256": common.sha256_file(spec["questions"])[:16],
                                        "aoci_bin": AOCI_BIN if os.path.exists(AOCI_BIN) else None})

    jobs = []
    for x in arms:
        kind, name = x["kind"], x["name"]
        script = os.path.join(HERE, KIND_SCRIPT[kind])
        cmd = [interpreter_for(kind), script, "--repo-name", spec["repo"], "--questions", spec["questions"],
               "--condition", kind, "--out", os.path.join(run_dir, f"answers-{name}.jsonl"),
               "--workers", str(x.get("workers", workers))]
        if kind in NEEDS_INDEX or kind in NEEDS_REPO:
            cmd += ["--repo-root", repo_root]
        if x.get("profile"):
            cmd += ["--profile", x["profile"]]
        if spec.get("limit"):
            cmd += ["--limit", str(spec["limit"])]
        for k, v in x.items():
            if k in ("name", "kind", "profile", "workers") or k.startswith("_"):
                continue
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        if spec.get("max_usd_per_arm") and "max_usd" not in x:
            cmd += ["--max-usd", str(spec["max_usd_per_arm"])]
        jobs.append((name, cmd))
    # A 48-question pass cannot keep more than the REMAINING question count in flight, so its tail idles the
    # pool ("并发到后面掉下来", 2026-08-20). concurrent_passes=P overlaps P passes: while one tails off the next
    # ramps up, and global in-flight stays near P × per-pass workers. Order is preserved by submission.
    P = max(1, int(spec.get("concurrent_passes") or 1))
    if a.dry_run or P == 1:
        for name, cmd in jobs:
            rc = sh(cmd, a.dry_run)
            if rc != 0:
                sys.exit(f"arm {name} failed (rc={rc})")
    else:
        from concurrent.futures import ThreadPoolExecutor
        out(f"concurrent_passes={P}: overlapping passes, per-pass workers={workers}")
        failed = []
        with ThreadPoolExecutor(max_workers=P) as ex:
            futs = {ex.submit(sh_buffered, cmd, name): name for name, cmd in jobs}
            for f in futs:
                pass
            for f, name in list(futs.items()):
                if f.result() != 0:
                    failed.append(name)
        if failed:
            sys.exit(f"arms failed: {failed}")

    if spec.get("task") == "locate" and not a.dry_run:
        # L batch: judge-free. Same acceptance gate discipline, then the deterministic scorer.
        acc = os.path.join(run_dir, "acceptance.json")
        rc = sh([PY, os.path.join(HERE, "accept.py"), run_dir, "--json", acc], False)
        if rc != 0 and not a.skip_accept:
            sys.exit("acceptance gate failed — fix or discard the offending records, then re-run "
                     "(or pass --skip-accept to score anyway, which the result card must then disclose)")
        rc = sh([PY, os.path.join(HERE, "locscore.py"), "--run", run_dir, "--questions", spec["questions"], "--force"], False)
        if rc != 0:
            sys.exit("locscore failed")

    if judge and not a.skip_judge and not a.dry_run:
        # ACCEPTANCE GATE: judging is the expensive half, so nothing tainted may reach it —
        # missing/errored records, commands that read the harness tree, network commands that
        # returned data, index-delivery mismatches, modified workspaces (harness/accept.py).
        acc = os.path.join(run_dir, "acceptance.json")
        rc = sh([PY, os.path.join(HERE, "accept.py"), run_dir, "--json", acc], False)
        if rc != 0 and not a.skip_accept:
            sys.exit("acceptance gate failed — fix or discard the offending records, then re-run "
                     "(or pass --skip-accept to judge anyway, which the result card must then disclose)")

    if judge and not a.skip_judge:
        judge_arms = arms if draws > 1 else [x for x in spec["arms"] if (not only or x["name"] in only)]
        # judge.py parallelises per RECORD, so one 48-question file can never use more than 48 workers.
        # A spec workers value above the question count therefore means "judge several files at once":
        # total concurrency stays ≈ judge.workers, split as n_files × per-file workers (2026-08-21).
        j_workers = int(judge.get("workers", workers))
        try:
            n_q = sum(1 for l in open(spec["questions"], encoding="utf-8") if l.strip())
            if spec.get("limit"):
                n_q = min(n_q, int(spec["limit"]))
        except (OSError, KeyError):
            n_q = 0
        per_file = min(j_workers, max(1, n_q or j_workers))
        j_files = max(1, min(4, j_workers // per_file))
        j_jobs = []
        for x in judge_arms:
            name = x["name"]
            if not os.path.exists(os.path.join(run_dir, f"answers-{name}.jsonl")) and not a.dry_run:
                out(f"judge: no answers for {name}, skipping")
                continue
            cmd = [PY, os.path.join(HERE, "judge.py"), "--answers", os.path.join(run_dir, f"answers-{name}.jsonl"),
                   "--out", os.path.join(run_dir, f"scores-{name}.jsonl"), "--votes", str(judge.get("votes", 1)),
                   "--workers", str(per_file)]
            if judge.get("profile"):
                cmd += ["--profile", judge["profile"]]
            if judge.get("max_tokens"):
                cmd += ["--max-tokens", str(judge["max_tokens"])]
            if judge.get("allow_self_judge"):
                cmd += ["--allow-self-judge"]
            j_jobs.append((name, cmd))
        if a.dry_run or j_files == 1:
            for name, cmd in j_jobs:
                rc = sh(cmd, a.dry_run)
                if rc != 0:
                    sys.exit(f"judge {name} failed (rc={rc})")
        else:
            from concurrent.futures import ThreadPoolExecutor
            out(f"judge: {j_files} files at once × {per_file} workers each (total ≈ {j_files * per_file})")
            failed = []
            with ThreadPoolExecutor(max_workers=j_files) as ex:
                futs = {ex.submit(sh_buffered, cmd, name): name for name, cmd in j_jobs}
                for f, name in list(futs.items()):
                    if f.result() != 0:
                        failed.append(name)
            if failed:
                sys.exit(f"judges failed: {failed}")

    if spec.get("task") == "locate":
        a.skip_analyze = True   # judge-dim summariser; L-batch stats come from locscore + runlevel_stats
    if not a.skip_analyze and not a.dry_run:
        cmd = [PY, os.path.join(HERE, "analyze.py"), run_dir, "--json", os.path.join(run_dir, "analysis.json")]
        if spec.get("pairs"):
            cmd += ["--pairs", ",".join(spec["pairs"])]
        if spec.get("baseline"):
            cmd += ["--baseline", spec["baseline"]]
        res = subprocess.run(cmd, capture_output=True, text=True)
        with open(os.path.join(run_dir, "analysis.txt"), "w", encoding="utf-8") as f:
            f.write(res.stdout + res.stderr)
        out(res.stdout + res.stderr)


if __name__ == "__main__":
    main()
