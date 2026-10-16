#!/usr/bin/env python3
"""Model-driven index authoring for an AOCI Volumes repository.

The author model is a profile (default: the `author` profile in profiles.json,
falling back to xingyun-opus5), so who writes the index is a recorded parameter,
not the operator. Two modes:

  author   write entries for every file the repository's own verify --json lists
           as Missing (or --paths), batch by batch, until nothing is missing
  retag    re-decide the compact tag (A layer + B module + C importance; E scale
           is computed from line count) for every existing entry, keeping the
           F/R/A/S text unchanged, and resubmit those whose tag changed

Per batch: file outlines (outline.py) + the tag dictionary parsed from the
repository's aoci.meta.txt + FRAS v2 limits + few-shot examples → the model
returns JSON → fras_lint (rune quotas, token bands, tag letters) → up to
--repairs repair rounds fed with the lint messages → `aoci update-entry` per
entry with source-SHA binding (machine acceptance; rejections get one more
repair round) → ledger. Model calls run in parallel (--workers); submissions are
serialised (the CLI rewrites the volume).

Outputs (authoring/<repo>/): author-<mode>-<ts>.jsonl (per-entry ledger) and
manifest-<mode>.json (author model, tokens, est. $, wall, attempts, tag
distribution before/after, lint/machine failure counts) — the paper's
authoring-cost table comes from here.

Usage:
  author.py --repo-root repos/xarray [--profile xingyun-opus5] --mode retag [--limit 20] [--dry-run]
  author.py --repo-root repos/xarray --mode author [--batch 12] [--workers 7] [--paths file.txt]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import fras_lint  # noqa: E402
import llm  # noqa: E402
import outline as outline_mod  # noqa: E402

AOCI_BIN = os.environ.get("AOCI_BIN", "aoci")
DEFAULT_AUTHOR_PROFILE = "xingyun-opus5"
SUBMIT_LOCK = threading.Lock()

# ---------------------------------------------------------------- repository readers

def parse_tagdict(meta_path):
    """Tag dictionary from aoci.meta.txt: {'A': {letter: name}, 'B': {...}, 'E': {...}} for the code kind."""
    d = {"A": {}, "B": {}, "C": {}, "E": {}}
    kind = None
    for line in open(meta_path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.match(r"^#\[Tag dictionary: (\w+)\]", line)
        if m:
            kind = m.group(1)
            continue
        if kind != "code":
            continue
        m = re.match(r"^#([ABCE]) [A-Za-z]+: (.*)$", line)
        if m:
            axis, body = m.group(1), m.group(2)
            for tok in body.split():
                if "-" in tok:
                    k, v = tok.split("-", 1)
                    d[axis][k] = v
    if not d["A"] or not d["B"]:
        sys.exit(f"could not parse a code tag dictionary from {meta_path}")
    return d


def scale_letter(lines):
    return "L" if lines > 400 else "M" if lines >= 200 else "S" if lines >= 100 else "T"


def parse_code_volume(repo_root):
    """path -> {name, tag, F, R, A, S, line} from aoci.code.txt."""
    root = os.path.realpath(repo_root)
    entries, cur = {}, None
    for line in open(os.path.join(repo_root, "aoci.code.txt"), encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("===") and line.endswith("==="):
            d = os.path.realpath(line.strip("=")) if line.strip("=") else root
            cur = os.path.relpath(d, root)
            cur = "" if cur == "." else cur + "/"
            continue
        m = fras_lint.ENTRY_RE.match(line)
        if m and cur is not None:
            path = cur + m.group("name")
            entries[path] = {"name": m.group("name"), "tag": m.group("tag"), "F": m.group("F"), "R": m.group("R"),
                             "A": m.group("A"), "S": m.group("S"), "line": line}
    return entries


def missing_paths(repo_root):
    out = subprocess.run([AOCI_BIN, "--repo", repo_root, "verify", "--json"], capture_output=True, text=True).stdout
    d = json.loads(out)
    miss = d["governance"]["code_drift"]["missing"]
    return sorted(x if isinstance(x, str) else x.get("path") for x in miss)


def baseline_files(repo_root):
    return json.load(open(os.path.join(repo_root, ".aoci", "baseline.json"), encoding="utf-8"))["files"]


def make_entry(name, tag, F, R, A, S):
    return f"{name}[{tag}]: F:{F} | R:{R} | A:{A} | S:{S}"


# ---------------------------------------------------------------- prompts

def dict_text(td):
    a = " ".join(f"{k}-{v}" for k, v in td["A"].items())
    b = " ".join(f"{k}-{v}" for k, v in td["B"].items())
    return (f"Compact tag = A layer + B module + C importance (1-9) + E scale.\n"
            f"A layer letters: {a}\nB module letters: {b}\n"
            f"C importance: 9-highest … 5-medium … 1-lowest (how much a maintainer must know this file to work safely).\n"
            f"E scale is computed from line count by the harness (L>400 M200-400 S100-200 T<100); do not output it.\n"
            f"Choose the A letter by what kind of code the file IS (tests→T, docs/specs→S, build/CI/ops→O or X, config data→F, "
            f"integration/backends/adapters→I, storage→P, core domain logic→D, algorithms/numerics→K, entry points/CLI/API surface→E, "
            f"shared helpers/utilities→C, runtime/platform→R, SDK/library facade→L). Choose B by the domain the file serves; "
            f"G (CrossDomain) only when no single domain fits. Vary tags honestly across files — a repository whose files all get "
            f"the same layer+module has been tagged badly.")


FRAS_RULES = (
    "FRAS v2 entry line: name[TAG]: F:<function> | R:<relations> | A:<interfaces> | S:<constraints>\n"
    "F ≤160 characters, one sentence, what the file does (not how). "
    "R ≤360 characters and ≤8 items, comma-separated canonical refs `code:<repo-relative path>` to files this one materially depends on or "
    "is coupled with; `-` if none. A ≤400 characters and ≤6 items: the public names other files use (functions, classes, CLI, env); `-` if none. "
    "S: constraints/invariants/gotchas that are NOT inferable from F and would prevent an error (S-Admission: non-inferable-and-error-preventing); "
    "`-` if none. S budget depends on C: C8-9 ≤600 chars and ≤200/140 tokens; C4-7 ≤200 chars and ≤80 tokens; C1-3 ≤50 chars and ≤40 tokens "
    "(tokens ≈ bytes/3). No line breaks, no `|` inside fields, plain ASCII punctuation preferred."
)

AUTHOR_SYSTEM = (
    "You are the index author for an AOCI repository-cognition index: one line per file, machine-checked. "
    "You see file outlines, not full files; write only what the outline supports; never invent APIs or relations. "
    "Answer with JSON only."
)


def author_prompt(td, examples, files_outlines):
    ex = "\n".join(examples[:3]) if examples else "(no existing entries yet)"
    body = "\n\n".join(files_outlines)
    return (f"{dict_text(td)}\n\n{FRAS_RULES}\n\nExamples of accepted entries from this repository:\n{ex}\n\n"
            f"Write one entry per file below. Return a JSON array of objects with keys: "
            f"path (exactly as given), layer (A letter), module (B letter), importance (1-9), F, R, I (the A: interfaces field), S. "
            f"Use \"-\" for empty R/I/S.\n\n{body}")


def retag_prompt(td, items):
    body = "\n\n".join(f"path: {p}\ncurrent tag: {e['tag']}\nF: {e['F']}\nR: {e['R']}\nA: {e['A']}\nS: {e['S']}\n{o}"
                       for p, e, o in items)
    return (f"{dict_text(td)}\n\nFor each entry below, decide the correct A layer letter, B module letter and C importance (1-9) "
            f"from the entry text and the outline. Keep the importance unless it is clearly wrong. Return a JSON array of objects "
            f"with keys: path (exactly as given), layer, module, importance.\n\n{body}")


def repair_prompt(problems):
    body = "\n\n".join(f"path: {p}\nproposed: {json.dumps(obj, ensure_ascii=False)}\nproblems: {'; '.join(errs)}" for p, obj, errs in problems)
    return ("Some entries violate the machine limits. Fix ONLY the listed entries and return a JSON array with the same keys "
            "(path, layer, module, importance, F, R, I, S). Shorten text to fit; do not drop required fields.\n\n" + body)


def parse_json_array(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return arr if isinstance(arr, list) else None


# ---------------------------------------------------------------- model + machine

def call_model(prof, system, user, max_tokens=8192):
    r = llm.complete(prof, [{"type": "text", "text": system}] if system else [], user, max_tokens=max_tokens, temperature=0)
    return r["text"], r["usage"]


def submit(repo_root, path, sha, entry, dry):
    if dry:
        return {"status": "dry-run", "exit": 0}
    with SUBMIT_LOCK:
        p = subprocess.run([AOCI_BIN, "--repo", repo_root, "update-entry", "--path", path, "--source-sha256", sha,
                            "--entry", entry, "--json"], capture_output=True, text=True)
    out = p.stdout.strip()
    try:
        j = json.loads(out) if out.startswith("{") else {}
    except json.JSONDecodeError:
        j = {}
    rec = {"status": j.get("status") or ("applied" if p.returncode == 0 else "error"), "exit": p.returncode}
    if p.returncode != 0:
        rec["findings"] = j.get("findings") or (out or p.stderr)[-600:]
    return rec


# ---------------------------------------------------------------- modes

def build_entry_from_obj(obj, path, td, lines):
    name = os.path.basename(path)
    layer, module = str(obj.get("layer", "")).strip()[:1], str(obj.get("module", "")).strip()[:1]
    try:
        imp = int(obj.get("importance", 5))
    except (TypeError, ValueError):
        imp = 5
    imp = max(1, min(9, imp))
    tag = f"{layer}{module}{imp}{scale_letter(lines)}"
    F = str(obj.get("F", "")).strip().replace("|", "/").replace("\n", " ")
    R = str(obj.get("R", "-")).strip().replace(" ", "").replace("|", ",") or "-"
    A = str(obj.get("I", obj.get("A", "-"))).strip().replace("|", "/").replace("\n", " ") or "-"
    S = str(obj.get("S", "-")).strip().replace("|", "/").replace("\n", " ") or "-"
    return tag, make_entry(name, tag, F, R, A, S)


def run_author(a, prof, td, repo_root, ledger_path):
    base = baseline_files(repo_root)
    existing = parse_code_volume(repo_root) if os.path.exists(os.path.join(repo_root, "aoci.code.txt")) else {}
    if a.paths:
        targets = [l.strip() for l in open(a.paths, encoding="utf-8") if l.strip()]
    else:
        targets = missing_paths(repo_root)
    targets = [t for t in targets if t in base]
    if a.limit:
        targets = targets[:a.limit]
    print(f"author: targets={len(targets)} batch={a.batch} workers={a.workers} model={prof['model']}", flush=True)
    examples = [e["line"] for e in list(existing.values())[:3]]
    batches = [targets[i:i + a.batch] for i in range(0, len(targets), a.batch)]
    totals = {"input": 0, "output": 0}
    stats = {"applied": 0, "lint_failed": 0, "machine_rejected": 0, "model_missing": 0, "attempts": 0}
    lock = threading.Lock()

    def do_batch(paths):
        t0 = time.time()
        outl = [outline_mod.outline_for(repo_root, p, max_lines=80) for p in paths]
        text, u = call_model(prof, AUTHOR_SYSTEM, author_prompt(td, examples, outl))
        usage = dict(u)
        objs = {o.get("path"): o for o in (parse_json_array(text) or []) if isinstance(o, dict)}
        results = {}
        pending = {}
        for p in paths:
            o = objs.get(p)
            if not o:
                results[p] = {"status": "model_missing"}
                continue
            tag, entry = build_entry_from_obj(o, p, td, outline_mod.line_count(repo_root, p))
            errs = fras_lint.lint(entry, td)
            if errs:
                pending[p] = (o, errs)
            else:
                results[p] = {"status": "lint_ok", "entry": entry, "tag": tag}
        rounds = 0
        while pending and rounds < a.repairs:
            rounds += 1
            text, u = call_model(prof, AUTHOR_SYSTEM, repair_prompt([(p, o, e) for p, (o, e) in pending.items()]))
            for k in usage:
                usage[k] = usage.get(k, 0) + u.get(k, 0)
            fixed = {o.get("path"): o for o in (parse_json_array(text) or []) if isinstance(o, dict)}
            for p in list(pending):
                o = fixed.get(p)
                if not o:
                    continue
                tag, entry = build_entry_from_obj(o, p, td, outline_mod.line_count(repo_root, p))
                errs = fras_lint.lint(entry, td)
                if errs:
                    pending[p] = (o, errs)
                else:
                    results[p] = {"status": "lint_ok", "entry": entry, "tag": tag}
                    del pending[p]
        for p, (o, e) in pending.items():
            results[p] = {"status": "lint_failed", "problems": e}
        # machine acceptance
        for p, r in results.items():
            if r["status"] != "lint_ok":
                continue
            sub = submit(repo_root, p, base[p]["sha256"], r["entry"], a.dry_run)
            r.update(sub)
            if sub["exit"] != 0:
                r["status"] = "machine_rejected"
            elif not a.dry_run:
                r["status"] = "applied"
        with lock:
            for k in totals:
                totals[k] += usage.get(k, 0)
            stats["attempts"] += 1 + rounds
            for p, r in results.items():
                if r["status"] in ("applied", "dry-run"):
                    stats["applied"] += 1
                elif r["status"] == "lint_failed":
                    stats["lint_failed"] += 1
                elif r["status"] == "machine_rejected":
                    stats["machine_rejected"] += 1
                elif r["status"] == "model_missing":
                    stats["model_missing"] += 1
                common.append_jsonl(ledger_path, {"mode": "author", "path": p, "model": prof["model"], **r,
                                                  "batch_usage": usage, "batch_wall_s": round(time.time() - t0, 1)})
        return results

    def on_result(item, res, i, n):
        ok = sum(1 for r in (res or {}).values() if r.get("status") in ("applied", "dry-run"))
        print(f"[batch {i}/{n}] files={len(item)} applied={ok} " + (f"error={res['error']}" if isinstance(res, dict) and 'error' in res and 'status' not in res else ""), flush=True)

    common.run_parallel(batches, do_batch, workers=a.workers, on_result=on_result, label="author")
    return totals, stats


def run_retag(a, prof, td, repo_root, ledger_path):
    base = baseline_files(repo_root)
    existing = parse_code_volume(repo_root)
    paths = [p for p in existing if p in base]
    if a.paths:
        want = {l.strip() for l in open(a.paths, encoding="utf-8") if l.strip()}
        paths = [p for p in paths if p in want]
    if a.limit:
        paths = paths[:a.limit]
    print(f"retag: entries={len(paths)} batch={a.batch} workers={a.workers} model={prof['model']}", flush=True)
    batches = [paths[i:i + a.batch] for i in range(0, len(paths), a.batch)]
    totals = {"input": 0, "output": 0}
    stats = {"applied": 0, "unchanged": 0, "lint_failed": 0, "machine_rejected": 0, "model_missing": 0, "attempts": 0}
    lock = threading.Lock()

    def do_batch(ps):
        t0 = time.time()
        items = [(p, existing[p], outline_mod.outline_for(repo_root, p, max_lines=25)) for p in ps]
        text, u = call_model(prof, AUTHOR_SYSTEM, retag_prompt(td, items), max_tokens=4096)
        usage = dict(u)
        objs = {o.get("path"): o for o in (parse_json_array(text) or []) if isinstance(o, dict)}
        results = {}
        for p in ps:
            e = existing[p]
            o = objs.get(p)
            if not o:
                results[p] = {"status": "model_missing", "old_tag": e["tag"]}
                continue
            layer, module = str(o.get("layer", "")).strip()[:1], str(o.get("module", "")).strip()[:1]
            old_c = int(re.findall(r"\d", e["tag"])[0])
            try:
                imp = max(1, min(9, int(o.get("importance", old_c))))
            except (TypeError, ValueError):
                imp = old_c
            lines = outline_mod.line_count(repo_root, p)
            new_tag = f"{layer}{module}{imp}{scale_letter(lines)}"
            entry = make_entry(e["name"], new_tag, e["F"], e["R"], e["A"], e["S"])
            errs = fras_lint.lint(entry, td)
            if errs and imp != old_c:  # a lowered importance can shrink the S budget below the existing S: keep the old C
                new_tag = f"{layer}{module}{old_c}{scale_letter(lines)}"
                entry = make_entry(e["name"], new_tag, e["F"], e["R"], e["A"], e["S"])
                errs = fras_lint.lint(entry, td)
            if errs:
                results[p] = {"status": "lint_failed", "old_tag": e["tag"], "new_tag": new_tag, "problems": errs}
                continue
            if new_tag == e["tag"]:
                results[p] = {"status": "unchanged", "old_tag": e["tag"], "new_tag": new_tag}
                continue
            sub = submit(repo_root, p, base[p]["sha256"], entry, a.dry_run)
            r = {"old_tag": e["tag"], "new_tag": new_tag, **sub}
            r["status"] = "applied" if sub["exit"] == 0 and not a.dry_run else ("dry-run" if a.dry_run else "machine_rejected")
            results[p] = r
        with lock:
            for k in totals:
                totals[k] += usage.get(k, 0)
            stats["attempts"] += 1
            for p, r in results.items():
                key = "applied" if r["status"] in ("applied", "dry-run") else r["status"]
                stats[key] = stats.get(key, 0) + 1
                common.append_jsonl(ledger_path, {"mode": "retag", "path": p, "model": prof["model"], **r,
                                                  "batch_usage": usage, "batch_wall_s": round(time.time() - t0, 1)})
        return results

    def on_result(item, res, i, n):
        if isinstance(res, dict) and "status" not in res and "error" in res:
            print(f"[batch {i}/{n}] ERROR {res['error']}", flush=True)
            return
        ch = sum(1 for r in res.values() if r.get("status") in ("applied", "dry-run"))
        same = sum(1 for r in res.values() if r.get("status") == "unchanged")
        print(f"[batch {i}/{n}] entries={len(item)} retagged={ch} unchanged={same}", flush=True)

    common.run_parallel(batches, do_batch, workers=a.workers, on_result=on_result, label="retag")
    return totals, stats


def tag_distribution(repo_root):
    from collections import Counter
    ex = parse_code_volume(repo_root) if os.path.exists(os.path.join(repo_root, "aoci.code.txt")) else {}
    ab = Counter(e["tag"][:2] for e in ex.values())
    return {"entries": len(ex), "layer_module": dict(ab.most_common()), "distinct_layer_module": len(ab)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--mode", choices=["author", "retag"], required=True)
    ap.add_argument("--profile", default=None, help=f"author model profile (default: profiles.json 'author' key or {DEFAULT_AUTHOR_PROFILE})")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=common.DEFAULT_WORKERS)
    ap.add_argument("--repairs", type=int, default=2, help="repair rounds after lint failures (author mode)")
    ap.add_argument("--paths", default=None, help="file with repo-relative paths to restrict to")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="call the model, lint, but do not submit")
    a = ap.parse_args()
    a.workers = common.clamp_workers(a.workers)
    cfg = llm._read()
    name = a.profile or cfg.get("author") or DEFAULT_AUTHOR_PROFILE
    prof = llm.load_profile(name)
    if not prof["_has_key"]:
        sys.exit(f"key missing for profile {prof['name']}")
    repo_root = a.repo_root
    td = parse_tagdict(os.path.join(repo_root, "aoci.meta.txt"))
    repo_name = os.path.basename(os.path.realpath(repo_root))
    out_dir = os.path.join(common.ROOT, "authoring", repo_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ledger = os.path.join(out_dir, f"author-{a.mode}-{ts}.jsonl")
    before = tag_distribution(repo_root)
    t0 = time.time()
    totals, stats = (run_author if a.mode == "author" else run_retag)(a, prof, td, repo_root, ledger)
    wall = round(time.time() - t0, 1)
    after = tag_distribution(repo_root)
    price = None
    try:
        import analyze
        price = analyze.cost_usd(analyze.load_prices(), prof["model"], totals)
    except Exception:  # noqa: BLE001
        pass
    manifest = {"mode": a.mode, "repo": repo_name, "repo_commit": common.git_sha(repo_root, short=False, untracked=False),
                "author_profile": prof["name"], "author_model": prof["model"], "harness_sha": common.harness_sha(),
                "batch": a.batch, "workers": a.workers, "repairs": a.repairs, "dry_run": a.dry_run, "when": ts,
                "wall_s": wall, "usage": totals, "est_cost_usd": price, "stats": stats,
                "tags_before": before, "tags_after": after, "ledger": os.path.relpath(ledger, common.ROOT)}
    with open(os.path.join(out_dir, f"manifest-{a.mode}-{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("tags_before", "tags_after")}, ensure_ascii=False))
    print("tags before:", before["distinct_layer_module"], "distinct layer+module;", "after:", after["distinct_layer_module"])
    if not a.dry_run:
        v = subprocess.run([AOCI_BIN, "--repo", repo_root, "verify", "--json"], capture_output=True, text=True).stdout
        try:
            d = json.loads(v)
            print(f"verify: structure_valid={d.get('structure_valid')} governance_aligned={d.get('governance_aligned')} "
                  f"missing={len(d['governance']['code_drift']['missing'])}")
        except Exception:  # noqa: BLE001
            print("verify: could not parse output")


if __name__ == "__main__":
    main()
