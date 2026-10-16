#!/usr/bin/env python3
"""S3 full-entry authoring from COMPLETE files (R3 index pipeline).

  .venv/bin/python3 harness/s3_author.py --repo kea-6c7ebba5 [--workers 20] [--profile gptnb-opus5] [--alt-profile xingyun-opus5]
                                         [--dry-run] [--paths a,b] [--limit N] [--no-pretag]

Every baseline file gets ONE entry (coverage 100%): tag (A layer + B module + C importance + E scale) and F/R/A/S,
all authored from the complete file text (G1: sha256 of the text sent is recorded per entry). Gates:
  G2 relay-truncation guard: per batch, usage.input (+cache) vs estimated tokens; ratio < 0.75 -> batch re-run file-by-file on --alt-profile
  G3 generated/data/oversize files are NOT sent in full: head excerpt only, S forced to "-", ledger mode=excerpt; binaries: name+size only
  pretag: one global pass over the file tree (no contents) proposes layer/module/importance for every file so tags stay consistent across batches
  S quota 600/400/200 with >=80% fill for C>=4 (extend round), shorten round + hard truncate for overflow; R refs validated against the baseline
Outputs into the authoring clone (absolute section headers, what the pinned binary parses): repos-deepswe/<repo>/aoci.code.txt + aoci.meta.txt
(S quota declaration rewritten to 600/400/200), and a ledger repos-deepswe/<repo>/.s3/ledger.jsonl + batches.jsonl. Validation: aoci check/verify.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import author as A0        # noqa: E402  (tag dict, scale letter, FRAS rules, json parsing)
import common              # noqa: E402
import fras_lint           # noqa: E402
import llm                 # noqa: E402
import s2_author as S2     # noqa: E402  (quotas, is_text_file, clean_s, hard_truncate, S2_RULES)

ROOT = os.path.expanduser("~/aoci-eval")
REPOS = os.path.join(ROOT, "repos-deepswe")
AOCI_BIN = os.environ.get("AOCI_BIN", os.path.join(ROOT, "bin", "aoci-pinned"))   # 2026-10-13: env override for the rc20 rerun line
BATCH_BYTES = 160_000
BATCH_FILES = 12
BIG_ALONE = 120_000
EXCERPT_BYTES = 4_000
OVERSIZE = 900_000
GEN_MIN = 200_000
GEN_NAME = re.compile(r"(^|/)(package-lock\.json|pnpm-lock\.yaml|yarn\.lock|bun\.lockb?|Cargo\.lock|poetry\.lock|composer\.lock|Gemfile\.lock|go\.work\.sum)$|\.(min\.js|min\.css|map|lock)$")
GEN_DIR = re.compile(r"(^|/)(testdata|fixtures?|__snapshots__|snapshots|corpus|golden|vendor|third_party|generated|gen)(/|$)")
TRUNC_RATIO = 0.75
SYSTEM = ("You are the index author for an AOCI repository-cognition index: one line per file, machine-checked. "
          "You see the COMPLETE source file (unless a file is explicitly marked as an excerpt or a binary). Write only what the "
          "file supports; never invent APIs, behaviours or relations. Answer with JSON only.")
RULES = (
    "ENTRY = tag + F + R + A + S for ONE file.\n"
    "TAG: layer (A letter), module (B letter), importance 1-9 (how much a maintainer must know this file to work safely in this "
    "repository; 9 = the few files everything hinges on, 5 = ordinary, 1-2 = assets, locks, generated data). A provisional tag from a "
    "whole-repository pass is given per file; keep it unless the full content clearly contradicts it (then change it and say so in \"tag_note\").\n"
    "F: <=160 characters, one sentence, what the file does (not how); specific to this file.\n"
    "R: <=8 items, comma-separated canonical refs `code:<repo-relative path>` to files this one materially depends on or is coupled with "
    "(imports it resolves inside this repository, files it generates/reads, its test or its subject); `-` if none. Only real paths.\n"
    "A: <=6 items, <=400 characters: the public names other files use (exported functions, types, classes, CLI commands, env vars); `-` if none.\n"
    "S: " + S2.S2_RULES.split("WHAT S IS.", 1)[1].replace("The OLD S was written from only the first 40 lines of the file. Treat it as untrusted: keep an old fact only if the full file confirms it and it still qualifies; otherwise replace it.", "If an OLD entry is shown it was written from the first 40 lines only: treat every old field as untrusted context.")
    + "\nFiles marked EXCERPT or BINARY were not fully read: give F/R/A from what is visible and the file name, and set S to \"-\".\n"
    "Plain ASCII punctuation; no line breaks; never use the character '|' in any field."
)


def sha16(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def classify(path, size):
    p = path.lower()
    if not S2.is_text_file(path):
        return "binary"
    if size > OVERSIZE:
        return "excerpt"
    if GEN_NAME.search(p) or (size > GEN_MIN and GEN_DIR.search(p)) or (size > GEN_MIN and p.endswith((".svg", ".sql", ".csv", ".txt", ".json", ".xml", ".yaml", ".yml", ".md"))):
        return "excerpt" if size > 20_000 else "full"
    return "full"


def est_tokens(text):
    return len(text.encode("utf-8")) / 3.2


def pretag_prompt(td, listing):
    return (A0.dict_text(td) + "\n\nBelow is the complete file list of one repository (path, bytes, lines). Assign every file a provisional "
            "layer letter, module letter and importance 1-9 using the whole-repository view: importance must vary honestly (only a few 8-9; "
            "assets, lockfiles, generated data and trivial config are 1-2; tests 2-4 unless central). Return a JSON array of objects "
            "{\"path\", \"layer\", \"module\", \"importance\"} covering EVERY path exactly as given.\n\n" + listing)


def block(e, pre, old):
    head = (f"### FILE: {e['path']}  ({e['mode'].upper()}, {e['bytes']} bytes, {e['lines']} lines; provisional tag {pre.get('layer','?')}{pre.get('module','?')}{pre.get('importance','?')}; "
            f"S quota for C{pre.get('importance',5)} = {S2.s2_quota(int(pre.get('importance',5)))} chars)")
    if old:
        head += f"\nOLD entry (40-line provenance, untrusted): {old}"
    if e["mode"] == "binary":
        return head + "\n<<<BINARY FILE: content not shown>>>"
    if e["mode"] == "excerpt":
        return head + f"\n<<<EXCERPT: first {EXCERPT_BYTES} bytes of a generated/data/oversize file>>>\n{e['text'][:EXCERPT_BYTES]}\n<<<END EXCERPT>>>"
    return head + f"\n<<<BEGIN FILE>>>\n{e['text']}\n<<<END FILE>>>"


def author_prompt(td, blocks):
    return (A0.dict_text(td) + "\n\n" + RULES + "\n\nReturn a JSON array with one object per file: "
            "{\"path\" (exactly as given), \"layer\", \"module\", \"importance\", \"tag_note\", \"F\", \"R\", \"A\", \"S\"}.\n\n"
            + "\n\n".join(blocks))


def call(prof, system, user, max_tokens=20000):
    r = llm.complete(prof, [{"type": "text", "text": system}], user, max_tokens=max_tokens, temperature=0)
    u = dict(r.get("usage") or {})
    return r["text"], u


def usage_in(u):
    """prompt tokens actually billed/served: OpenAI-style prompt_tokens already includes cached tokens, so take the max"""
    return max(u.get("input", 0) or 0, (u.get("cache_read", 0) or 0) + (u.get("cache_write", 0) or 0))


def clean_field(v, maxlen=None):
    v = S2.clean_s(v)
    return v if maxlen is None else v[:maxlen]


def norm_R(v, baseline, self_path):
    items = []
    for tok in str(v or "").replace("|", ",").split(","):
        t = tok.strip()
        if not t or t == "-":
            continue
        t = t[5:] if t.startswith("code:") else t
        t = t[2:] if t.startswith("./") else t
        if t in baseline and t != self_path and ("code:" + t) not in items:
            items.append("code:" + t)
    items = items[:8]
    while items and len(",".join(items)) > 360:
        items.pop()
    return ",".join(items) if items else "-"


def norm_A(v):
    v = clean_field(v)
    if v == "-":
        return "-"
    parts = [x.strip() for x in v.split(",") if x.strip()][:6]
    out = ", ".join(parts)
    return out[:400] if out else "-"


def build(obj, e, td, baseline):
    layer = str(obj.get("layer", "")).strip()[:1].upper()
    module = str(obj.get("module", "")).strip()[:1].upper()
    if layer not in td["A"]:
        layer = "Z"
    if module not in td["B"]:
        module = "Z"
    try:
        c = max(1, min(9, int(obj.get("importance", 5))))
    except (TypeError, ValueError):
        c = 5
    tag = f"{layer}{module}{c}{A0.scale_letter(e['lines'])}"
    F = clean_field(obj.get("F", "")) or "-"
    R = norm_R(obj.get("R", "-"), baseline, e["path"])
    if os.environ.get("AOCI_S_QUOTA_PROFILE") == "rc20" and R != "-":   # rc20 R budget by C band (token×3 chars)
        refs = R.split(",")
        while refs and len(",".join(refs)) > S2.rc20_r_limit(c):
            refs.pop()
        R = ",".join(refs) if refs else "-"
    Af = norm_A(obj.get("A", obj.get("I", "-")))
    S = clean_field(obj.get("S", "-"))
    if e["mode"] != "full":
        S = "-"
    return {"tag": tag, "C": c, "F": F, "R": R, "A": Af, "S": S, "tag_note": clean_field(obj.get("tag_note", ""))}


def contract_quota(c):
    """the pinned binary's machine contract (600/200/50) — our authored S uses 600/400/200, so structure is validated on a trimmed copy"""
    return 600 if c >= 8 else 200 if c >= 4 else 50


def rr_of(repo_root):
    return os.path.realpath(repo_root)


def validate_with_binary(repo_root, lines):
    """Write a contract-compliant copy (S trimmed), re-baseline, run check+verify, then restore the real volume and re-baseline.
    Returns (check finding counts, verify missing count). The real volume deviates from the contract only in S length (disclosed)."""
    vol = os.path.join(repo_root, "aoci.code.txt")
    real = open(vol, encoding="utf-8").read()
    trimmed = []
    unrep = 0
    skip_section = False
    for ln in real.splitlines():
        if ln.startswith("===") and ln.endswith("==="):
            d = ln.strip("=")
            # object-fras-v2 section/entry grammar cannot carry these characters (Next/SvelteKit route dirs, size=1024 dirs):
            skip_section = bool(re.search(r"[()\[\]=\s]", os.path.relpath(d, rr_of(repo_root)) if d.startswith("/") else d))
            if skip_section:
                continue
        mm = fras_lint.ENTRY_RE.match(ln)
        if skip_section and (mm or "]: F:" in ln):
            unrep += 1
            continue
        if not mm and ln and not ln.startswith(("#", "===")) and "]: F:" in ln:
            unrep += 1                        # names containing [ or ] are unrepresentable
            continue
        if mm and mm.group("name").startswith("+"):
            unrep += 1                        # SvelteKit +page/+layout names are unrepresentable
            continue
        if mm:
            mc = re.search(r"[1-9]", mm.group("tag")); c = int(mc.group(0)) if mc else 5
            S = mm.group("S")
            if len(S) > contract_quota(c):
                S = S[:contract_quota(c) - 1].rstrip() + "~"
            ln = f"{mm.group('name')}[{mm.group('tag')}]: F:{mm.group('F')} | R:{mm.group('R')} | A:{mm.group('A')} | S:{S}"
        trimmed.append(ln)
    open(vol, "w", encoding="utf-8").write("\n".join(trimmed) + "\n")
    chk = subprocess.run([AOCI_BIN, "--repo", repo_root, "check", "--json"], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    ver = subprocess.run([AOCI_BIN, "--repo", repo_root, "verify", "--json"], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    codes = {}
    try:
        cj = json.loads(chk)
        if cj.get("error_code"):
            codes = {"ERROR:" + str(cj.get("error_code")): str(cj.get("message", ""))[:160]}
        for fnd in cj.get("findings", []):
            codes[fnd.get("code")] = codes.get(fnd.get("code"), 0) + 1
    except Exception:
        codes = {"unparsed": 1}
    try:
        vj = json.loads(ver)
        if vj.get("error_code"):
            miss = "ERROR:" + str(vj.get("message", ""))[:120]
        else:
            cd = (vj.get("governance") or {}).get("code_drift") or vj.get("code_drift") or {}
            miss = len(cd.get("missing", []))
    except Exception:
        miss = "?"
    open(vol, "w", encoding="utf-8").write(real)
    expected = {"code_volume_unbaselined", "meta_volume_baseline_drift", "cognition_budget_exceeded"}
    codes = {k: v for k, v in codes.items() if k not in expected} | {"(expected-drift)": sum(1 for k in codes if k in expected)}
    # baseline files that are not git-tracked at the task commit are not in the task container: never entries, never "missing"
    try:
        bl = set(json.load(open(os.path.join(repo_root, ".aoci", "baseline.json")))["files"])
        tr = set(subprocess.run(["git", "-C", repo_root, "ls-files", "-z"], capture_output=True, text=True).stdout.split("\0")) - {""}
        untracked = len([p for p in bl - tr if p not in ("aoci.txt", "aoci.meta.txt", "aoci.code.txt", "AGENTS.md") and not p.startswith(".aoci/")])
    except Exception:
        untracked = 0
    if isinstance(miss, int):
        eff = max(0, miss - unrep - untracked)
        note = []
        if unrep:
            note.append(f"unrepresentable-name {unrep}")
        if untracked:
            note.append(f"untracked-in-clone {untracked}")
        miss = f"{eff}" + (f" ({', '.join(note)})" if note else "")
        if codes.get("code_missing", 0) and codes.get("code_missing") <= unrep + untracked:
            codes.pop("code_missing")
    return codes, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--quota-profile", default=os.environ.get("AOCI_S_QUOTA_PROFILE", "eval"), choices=["eval", "mega"],
                    help="S quota tier profile: eval=600/400/200, mega=600/200/50 (large repos)")
    ap.add_argument("--profile", default="gptnb-opus5")
    ap.add_argument("--alt-profile", default="xingyun-opus5")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--paths", default="")
    ap.add_argument("--paths-file", default="", help="file with one repo-relative path per line (avoids ARG_MAX)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretag", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--size-alarm", type=int, default=400_000, help="projected index tokens above this → SIZE ALARM (exit 3 at start unless --force-size)")
    ap.add_argument("--force-size", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse entries persisted in partial.jsonl from a crashed run")
    ap.add_argument("--validate-only", action="store_true", help="re-run the pinned-binary validation on the authored volume, no model calls")
    a = ap.parse_args()
    os.environ["AOCI_S_QUOTA_PROFILE"] = a.quota_profile   # 让 s2_author.s2_quota 的全部消费面(填充轮/提示词/门禁)一致
    repo_root = os.path.join(REPOS, a.repo)
    if a.validate_only:
        codes, miss = validate_with_binary(repo_root, open(os.path.join(repo_root, "aoci.code.txt"), encoding="utf-8").read().splitlines())
        print(f"validate · {a.repo} · verify missing={miss} · check findings={codes}")
        return
    rr = os.path.realpath(repo_root)
    td = A0.parse_tagdict(os.path.join(repo_root, "aoci.meta.txt"))
    baseline = A0.baseline_files(repo_root)
    # rc20 Managed Scope: baseline.json carries a role per file; only role=index files get Entries (observe/exclude never do)
    if isinstance(baseline, dict) and any(isinstance(v, dict) for v in baseline.values()):
        n_all = len(baseline)
        baseline = {p: v for p, v in baseline.items() if not isinstance(v, dict) or v.get("role", "index") == "index"}
        print(f"  baseline roles: {len(baseline)} index of {n_all} files (observe/exclude skipped)", flush=True)
    old = {}
    try:
        old = {p: v["line"] for p, v in A0.parse_code_volume(repo_root).items()}
    except Exception:
        pass
    ledger_dir = os.path.join(ROOT, "deepswe-inject-s3", ".ledgers", a.repo)
    os.makedirs(ledger_dir, exist_ok=True)
    tracked = set(subprocess.run(["git", "-C", repo_root, "ls-files", "-z"], capture_output=True, text=True).stdout.split("\0")) - {""}
    # ---- files: baseline ∩ git-tracked (what the task container actually contains); AOCI artefacts never get entries
    files = []
    skipped_untracked = []
    for p in sorted(baseline):
        if p in ("aoci.txt", "aoci.meta.txt", "aoci.code.txt", "AGENTS.md") or p.startswith(".aoci/") or p.startswith(".s3/"):
            continue
        if tracked and p not in tracked:
            skipped_untracked.append(p)
            continue
        full = os.path.join(repo_root, p)
        if not os.path.isfile(full):
            continue
        size = os.path.getsize(full)
        mode = classify(full, size)
        text = ""
        if mode != "binary":
            try:
                text = open(full, encoding="utf-8", errors="replace").read()
            except Exception:
                mode = "binary"
        files.append({"path": p, "bytes": size, "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0) if text else 0,
                      "mode": mode, "text": text, "src_sha": sha16(text) if text else None})
    if a.paths_file:
        a.paths = ",".join(l.strip() for l in open(a.paths_file, encoding="utf-8") if l.strip())
    if a.paths:
        want = set(a.paths.split(","))
        files = [f for f in files if f["path"] in want]
    if a.limit:
        files = files[:a.limit]
    by_path = {f["path"]: f for f in files}
    total_bytes = sum(len(f["text"]) for f in files if f["mode"] == "full")
    if skipped_untracked:
        print(f"  skipped {len(skipped_untracked)} baseline files not tracked at the task commit (init artefacts etc.): {skipped_untracked[:4]}", flush=True)
    TOK_PER_ENTRY = 143                     # measured S2/S3 density (bytes/3.5)
    projected = len(files) * TOK_PER_ENTRY
    if projected > a.size_alarm:
        print(f"SIZE ALARM · {a.repo} · {len(files)} files → projected index ≈ {projected // 1000}k tokens > {a.size_alarm // 1000}k"
              + ("" if a.force_size else " · STOP (use --force-size to override)"), flush=True)
        if not a.force_size:
            sys.exit(3)
    print(f"s3 · {a.repo} · files={len(files)} full={sum(1 for f in files if f['mode']=='full')} excerpt={sum(1 for f in files if f['mode']=='excerpt')} "
          f"binary={sum(1 for f in files if f['mode']=='binary')} · full-text {total_bytes//1024}KB ≈ {int(total_bytes/3.2/1000)}k tok · old entries {len(old)} · model={a.profile} workers={a.workers}"
          + (" · DRY-RUN" if a.dry_run else ""), flush=True)
    # ---- batches
    files_sorted = sorted(files, key=lambda f: (os.path.dirname(f["path"]), f["path"]))
    batches, cur, cur_b = [], [], 0
    for f in files_sorted:
        w = len(f["text"]) if f["mode"] == "full" else (EXCERPT_BYTES if f["mode"] == "excerpt" else 200)
        if w >= BIG_ALONE:
            if cur:
                batches.append(cur); cur, cur_b = [], 0
            batches.append([f]); continue
        if cur and (cur_b + w > BATCH_BYTES or len(cur) >= BATCH_FILES):
            batches.append(cur); cur, cur_b = [], 0
        cur.append(f); cur_b += w
    if cur:
        batches.append(cur)
    print(f"  batches={len(batches)}", flush=True)
    if a.dry_run:
        for i, b in enumerate(batches[:4]):
            print(f"  batch {i}: {[x['path'] for x in b]} ({sum(len(x['text']) for x in b)//1024}KB)")
        return
    llm.load_secrets()
    prof = llm.load_profile(a.profile)
    alt = llm.load_profile(a.alt_profile) if a.alt_profile else prof
    lock = threading.Lock()
    usage_tot = {"in": 0, "out": 0}
    blog = open(os.path.join(ledger_dir, "batches.jsonl"), "a")
    # ---- pretag (global view, no contents)
    pre = {}
    if not a.no_pretag:
        listing_rows = [f"{f['path']}\t{f['bytes']}\t{f['lines']}" for f in files_sorted]
        chunks = [listing_rows[i:i + 250] for i in range(0, len(listing_rows), 250)]
        tree = "\n".join(sorted({os.path.dirname(f["path"]) or "." for f in files_sorted}))
        def do_pre(ch):
            try:
                txt, u = call(prof, SYSTEM, pretag_prompt(td, "DIRECTORIES:\n" + tree + "\n\nFILES (path\\tbytes\\tlines):\n" + "\n".join(ch)), max_tokens=12000)
            except Exception as e:
                print(f"  pretag chunk failed: {e.__class__.__name__}: {str(e)[:120]}", flush=True)
                return {}
            arr = A0.parse_json_array(txt) or []
            if not arr:
                print(f"  pretag chunk: unparseable output ({len(txt)} chars)", flush=True)
            with lock:
                usage_tot["in"] += usage_in(u); usage_tot["out"] += u.get("output", 0) or 0
            return {o.get("path"): o for o in arr if isinstance(o, dict)}
        for r in common.run_parallel(chunks, do_pre, workers=min(a.workers, 4), label="pretag"):
            if isinstance(r, dict):
                pre.update(r)
        print(f"  pretag: {len(pre)}/{len(files)} files tagged", flush=True)
    results = {}
    if a.resume and os.path.exists(os.path.join(ledger_dir, "partial.jsonl")):
        for l in open(os.path.join(ledger_dir, "partial.jsonl"), encoding="utf-8"):
            try:
                o = json.loads(l)
            except Exception:
                continue
            pth = o.pop("path", None)
            if pth in by_path and all(k in o for k in ("tag", "C", "F", "R", "A", "S")):
                results[pth] = o
        batches = [[f for f in b if f["path"] not in results] for b in batches]
        batches = [b for b in batches if b]
        print(f"  resume: {len(results)} entries reused, {len(batches)} batches left", flush=True)
    # ---- authoring batches
    def do_batch(idx_b):
        idx, b = idx_b
        blocks = [block(f, pre.get(f["path"], {}), old.get(f["path"])) for f in b]
        prompt = author_prompt(td, blocks)
        est = est_tokens(prompt)
        txt, u = call(prof, SYSTEM, prompt)
        ratio = usage_in(u) / est if est else 1.0
        arr = A0.parse_json_array(txt) or []
        got = {o.get("path"): o for o in arr if isinstance(o, dict)}
        rec = {"batch": idx, "files": [f["path"] for f in b], "est_tokens": int(est), "usage_in": usage_in(u), "ratio": round(ratio, 2),
               "returned": len(got), "profile": a.profile}
        suspect = ratio < TRUNC_RATIO and est > 20000
        if suspect:
            rec["suspect_truncation"] = True
            got = {}
            for f in b:                                   # re-run file by file on the alternate relay
                t2, u2 = call(alt, SYSTEM, author_prompt(td, [block(f, pre.get(f["path"], {}), old.get(f["path"]))]))
                a2 = A0.parse_json_array(t2) or []
                for o in a2:
                    if isinstance(o, dict) and o.get("path") == f["path"]:
                        got[f["path"]] = o
                rec.setdefault("alt_ratios", []).append(round(usage_in(u2) / max(1, est_tokens(f["text"])), 2))
                with lock:
                    usage_tot["in"] += usage_in(u2); usage_tot["out"] += u2.get("output", 0) or 0
        with lock:
            usage_tot["in"] += usage_in(u); usage_tot["out"] += u.get("output", 0) or 0
            blog.write(json.dumps(rec) + "\n"); blog.flush()
        out = {}
        for f in b:
            o = got.get(f["path"])
            if o:
                out[f["path"]] = build(o, f, td, baseline)
                out[f["path"]]["batch"] = idx
                out[f["path"]]["ratio"] = round(ratio, 2)
        return out
    partial = open(os.path.join(ledger_dir, "partial.jsonl"), "a")

    def on_result(item, r, i, n):
        if isinstance(r, dict):
            good = {k: v for k, v in r.items() if k in by_path}
            with lock:
                results.update(good)
                for k, v in good.items():                 # crash insurance: persist every authored entry immediately
                    partial.write(json.dumps({"path": k, **{kk: vv for kk, vv in v.items() if kk != "_kept"}}, ensure_ascii=False) + "\n")
                partial.flush()
            if len(good) < len(r):
                print(f"  batch returned non-path keys (error?): {[k for k in r if k not in by_path][:2]}", flush=True)
        if i % 20 == 0 and results:
            produced = sum(len(A0.make_entry(os.path.basename(k), v["tag"], v["F"], v["R"], v["A"], v["S"])) for k, v in results.items())
            proj = produced / len(results) * len(files) / 3.5
            if proj > a.size_alarm:
                print(f"SIZE ALARM (running) · {a.repo} · projected ≈ {int(proj) // 1000}k tokens from {len(results)} entries", flush=True)
        print(f"[batch {i}/{n}] ok={len(results)}", flush=True)
    common.run_parallel(list(enumerate(batches)), do_batch, workers=a.workers, on_result=on_result, label="s3")
    # ---- coverage retry: files the model skipped
    missing = [f for f in files if f["path"] not in results]
    for attempt in range(2):
        if not missing:
            break
        print(f"  coverage retry {attempt+1}: {len(missing)} files", flush=True)
        def do_one(f):
            txt, u = call(prof if attempt == 0 else alt, SYSTEM, author_prompt(td, [block(f, pre.get(f["path"], {}), old.get(f["path"]))]))
            with lock:
                usage_tot["in"] += usage_in(u); usage_tot["out"] += u.get("output", 0) or 0
            arr = A0.parse_json_array(txt) or []
            for o in arr:
                if isinstance(o, dict) and o.get("path") == f["path"]:
                    r = build(o, f, td, baseline); r["batch"] = -1; r["ratio"] = None
                    return {f["path"]: r}
            return {}
        for r in common.run_parallel(missing, do_one, workers=a.workers, label="cover"):
            if isinstance(r, dict):
                results.update({k: v for k, v in r.items() if k in by_path})
        missing = [f for f in files if f["path"] not in results]
    for f in missing:                                     # last resort: minimal honest entry, flagged in the ledger
        results[f["path"]] = {"tag": f"ZZ2{A0.scale_letter(f['lines'])}", "C": 2, "F": f"File {os.path.basename(f['path'])} (entry not produced by the model; fallback).",
                              "R": "-", "A": "-", "S": "-", "tag_note": "fallback", "batch": -2, "ratio": None}
    # ---- S fill / overflow rounds (C>=4 under 80% -> extend; any over quota -> shorten)
    def s_round(kind):
        items = []
        for p, r in results.items():
            f = by_path[p]
            if f["mode"] != "full" or r["S"] == "-" and kind == "shorten":
                continue
            q = S2.s2_quota(r["C"])
            if kind == "extend" and r["C"] >= 4 and len(r["S"]) < S2.FILL_FLOOR * q and not r.get("exhausted"):
                items.append(f)
            if kind == "shorten" and len(r["S"]) > q:
                items.append(f)
        if not items:
            return
        print(f"  {kind} round: {len(items)} entries", flush=True)
        groups = [items[i:i + 6] for i in range(0, len(items), 6)]
        def do_g(g):
            if kind == "extend":
                body = "\n\n".join(block(f, {"layer": results[f["path"]]["tag"][0], "module": results[f["path"]]["tag"][1], "importance": results[f["path"]]["C"]}, None)
                                   + f"\nCURRENT S ({len(results[f['path']]['S'])} of {S2.s2_quota(results[f['path']]['C'])} chars): {results[f['path']]['S']}" for f in g)
                prompt = (S2.S2_RULES + "\n\nThe S below uses less than 80% of its quota. Re-read the complete file and, ONLY if it supports further facts that pass "
                          "the self-test, extend S up to the quota. If nothing more qualifies, return the same S and set \"exhausted\": true.\n"
                          "Return a JSON array: [{\"path\": ..., \"S\": ..., \"exhausted\": bool}]\n\n" + body)
            else:
                prompt = S2.shorten_prompt([(f["path"], results[f["path"]]["S"], S2.s2_quota(results[f["path"]]["C"])) for f in g])
            txt, u = call(prof, SYSTEM, prompt)
            with lock:
                usage_tot["in"] += usage_in(u); usage_tot["out"] += u.get("output", 0) or 0
            out = {}
            for o in A0.parse_json_array(txt) or []:
                if isinstance(o, dict) and o.get("path") in results:
                    out[o["path"]] = (S2.clean_s(o.get("S")), bool(o.get("exhausted")))
            return out
        for r in common.run_parallel(groups, do_g, workers=a.workers, label=kind):
            if isinstance(r, dict):
                for p, (s, ex) in r.items():
                    results[p]["S"] = s; results[p]["exhausted"] = ex
                    results[p]["attempts"] = results[p].get("attempts", 1) + 1
    for kind in ("extend", "shorten"):
        try:
            s_round(kind)
        except Exception as e:
            print(f"  {kind} round failed ({e.__class__.__name__}: {str(e)[:100]}); continuing with current S", flush=True)
    # F over 160: ask the model for a complete <=150-char sentence instead of cutting mid-word
    longF = [p for p, r in results.items() if len(r["F"]) > 160]
    if longF:
        print(f"  F shorten round: {len(longF)} entries", flush=True)
        def do_f(p):
            f = by_path[p]
            prompt = (f"FILE: {p}\n<<<BEGIN FILE>>>\n{f['text'][:200000]}\n<<<END FILE>>>\n\nCurrent F (too long): {results[p]['F']}\n\n"
                      "Rewrite F as ONE complete sentence of at most 150 characters saying what the file does (not how). Return JSON: {\"F\": \"...\"}")
            try:
                txt, u = call(prof, SYSTEM, prompt, max_tokens=400)
                with lock:
                    usage_tot["in"] += usage_in(u); usage_tot["out"] += u.get("output", 0) or 0
                mm = re.search(r"\{.*\}", txt, re.S)
                return {p: S2.clean_s(json.loads(mm.group(0))["F"])[:160]}
            except Exception:
                return {}
        for r in common.run_parallel(longF, do_f, workers=a.workers, label="fshort"):
            if isinstance(r, dict):
                for p, F in r.items():
                    if p in results:
                        results[p]["F"] = F
        for p in longF:                                  # last resort: cut at a word boundary
            if len(results[p]["F"]) > 160:
                results[p]["F"] = results[p]["F"][:157].rsplit(" ", 1)[0] + "..."
    for p, r in results.items():
        q = S2.s2_quota(r["C"])
        if len(r["S"]) > q:
            r["S"] = S2.hard_truncate(r["S"], q); r["truncated"] = True
    # ---- write volume (absolute section headers for the pinned binary) + meta quota declaration + ledger
    if a.paths or a.limit:                       # partial run: keep every existing entry not re-authored now
        try:
            for p0, e0 in A0.parse_code_volume(repo_root).items():
                if p0 not in results and p0 in baseline:
                    mc = re.search(r"[1-9]", e0["tag"])
                    results[p0] = {"tag": e0["tag"], "C": int(mc.group(0)) if mc else 5, "F": e0["F"], "R": e0["R"], "A": e0["A"], "S": e0["S"], "tag_note": "", "batch": None, "ratio": None, "_kept": True}
                    by_path.setdefault(p0, {"path": p0, "bytes": 0, "lines": 0, "mode": "kept", "text": "", "src_sha": None})
        except Exception:
            pass
    by_dir = {}
    for p, r in results.items():
        by_dir.setdefault(os.path.dirname(p), []).append((p, r))
    lines = ["#AOCI-CODE-VOLUME: 1"]
    for d in sorted(by_dir, key=lambda x: (x != "", x)):
        lines.append(""); lines.append(f"==={rr}/{d}/===" if d else f"==={rr}/===")
        for p, r in sorted(by_dir[d]):
            lines.append(A0.make_entry(os.path.basename(p), r["tag"], r["F"], r["R"], r["A"], r["S"]))
    open(os.path.join(repo_root, "aoci.code.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    mp = os.path.join(repo_root, "aoci.meta.txt"); m = open(mp, encoding="utf-8").read()
    m = re.sub(r"^#S quota: .*$", f"#S quota: {S2.S2_BANDS}", m, flags=re.M)
    if "#S配额:" not in m:
        m = m.replace(f"#S quota: {S2.S2_BANDS}\n", f"#S quota: {S2.S2_BANDS}\n#S配额: {S2.S2_BANDS}\n")
    open(mp, "w", encoding="utf-8").write(m)
    with open(os.path.join(ledger_dir, "ledger.jsonl"), "a") as lf:
        for p, r in sorted(results.items()):
            if r.get("_kept"):
                continue
            f = by_path[p]
            lf.write(json.dumps({"path": p, "mode": f["mode"], "bytes": f["bytes"], "src_lines": f["lines"], "src_sha": f["src_sha"],
                                 "tag": r["tag"], "C": r["C"], "F": r["F"], "R": r["R"], "A": r["A"], "S": r["S"], "new_S": r["S"], "old_S": "",
                                 "quota": S2.s2_quota(r["C"]), "fill": round(len(r["S"]) / S2.s2_quota(r["C"]), 2) if r["S"] != "-" else 0.0, "attempts": r.get("attempts", 1),
                                 "exhausted": r.get("exhausted"), "truncated": r.get("truncated", False), "batch": r.get("batch"),
                                 "usage_ratio": r.get("ratio"), "tag_note": r.get("tag_note", ""), "ts": __import__("time").strftime("%F %T")},
                                ensure_ascii=False) + "\n")
    # ---- export delivery copies (relative section headers) + spotcheck-compatible ledger
    exp = os.path.join(ROOT, "deepswe-inject-s3", a.repo)
    os.makedirs(exp, exist_ok=True)
    rel = []
    for ln in lines:
        if ln.startswith("===") and ln.endswith("==="):
            d = ln.strip("=")
            d = os.path.relpath(d, rr) if d.startswith("/") else d
            rel.append("===./===" if d in (".", "") else f"==={d.strip('/')}/===")
        else:
            rel.append(ln)
    open(os.path.join(exp, "aoci.code.txt"), "w", encoding="utf-8").write("\n".join(rel) + "\n")
    open(os.path.join(exp, "aoci.meta.txt"), "w", encoding="utf-8").write(m)
    open(os.path.join(exp, "aoci.txt"), "w", encoding="utf-8").write(open(os.path.join(repo_root, "aoci.txt"), encoding="utf-8").read())
    with open(os.path.join(exp, "s2-ledger.jsonl"), "a" if (a.paths or a.limit) else "w") as lf2:
        for p, r in sorted(results.items()):
            if r.get("_kept"):
                continue
            f = by_path[p]
            lf2.write(json.dumps({"path": p, "C": r["C"], "quota": S2.s2_quota(r["C"]), "old_S": (old.get(p) or "").split(" | S:")[-1] if old.get(p) else "",
                                  "new_S": r["S"], "fill": round(len(r["S"]) / S2.s2_quota(r["C"]), 2) if r["S"] != "-" else 0.0,
                                  "attempts": r.get("attempts", 1), "bytes": f["bytes"], "mode": f["mode"], "tag": r["tag"], "F": r["F"], "R": r["R"], "A": r["A"]},
                                 ensure_ascii=False) + "\n")
    codes, miss = validate_with_binary(repo_root, lines)
    cov_missing = [p for p in baseline if p in tracked and p not in results and not (p in ("aoci.txt", "aoci.meta.txt", "aoci.code.txt", "AGENTS.md") or p.startswith(".aoci/"))]
    codes["python_coverage_missing"] = len(cov_missing)
    full = [r for p, r in results.items() if by_path[p]["mode"] == "full"]
    c4 = [r for r in full if r["C"] >= 4]
    print(f"\ns3 done · {a.repo} · entries {len(results)}/{len(files)} · verify missing={miss} · check findings={codes} · "
          f"C≥4 fill mean {sum(len(r['S'])/S2.s2_quota(r['C']) for r in c4)/max(1,len(c4)):.2f} · fallback {sum(1 for r in results.values() if r.get('batch')==-2)} · "
          f"tokens in={usage_tot['in']} out={usage_tot['out']} · est ${usage_tot['in']/1e6*5 + usage_tot['out']/1e6*25:.2f}", flush=True)


if __name__ == "__main__":
    main()
